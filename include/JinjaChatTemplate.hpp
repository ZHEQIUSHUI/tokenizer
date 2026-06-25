#pragma once
// Jinja-based chat templating: render a model's own HuggingFace `chat_template`
// (from tokenizer_config.json / chat_template.jinja) with minja, instead of the
// hand-written per-model apply_chat_template string concatenation.
//
// Optional feature (pulls in third_party/minja, which uses <regex> on the small
// template string only). Build with -I third_party and define the include path.
//
// Text models: drop-in. Multimodal needs an extra pad-token expansion step after
// rendering (the template emits a single placeholder per image; the per-image
// token count is a runtime concern, exactly as in HF's processor) -- see apply().

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

#include "BaseTokenizer.hpp"
#include "minja/chat-template.hpp"

class JinjaChatTemplate
{
public:
    using json = nlohmann::ordered_json;

    // Construct directly from a template source string.
    bool load(const std::string &chat_template, const std::string &bos_token, const std::string &eos_token)
    {
        if (chat_template.empty())
            return false;
        try {
            tmpl_ = std::make_shared<minja::chat_template>(chat_template, bos_token, eos_token);
        } catch (const std::exception &) {
            tmpl_ = nullptr;
            return false;
        }
        return tmpl_ != nullptr;
    }

    // Load from a HuggingFace tokenizer directory (reads tokenizer_config.json,
    // and chat_template.jinja if the config doesn't carry the template inline).
    bool load_from_dir(const std::string &dir)
    {
        json cfg;
        if (!read_json(join(dir, "tokenizer_config.json"), cfg))
            return false;

        std::string ct = extract_chat_template(cfg);
        if (ct.empty()) {
            // Newer repos store the template in a sidecar file.
            std::ifstream jf(join(dir, "chat_template.jinja"));
            if (jf) {
                std::stringstream ss; ss << jf.rdbuf();
                ct = ss.str();
            }
        }
        const std::string bos = token_str(cfg, "bos_token");
        const std::string eos = token_str(cfg, "eos_token");
        return load(ct, bos, eos);
    }

    bool ready() const { return tmpl_ != nullptr; }

    // Override the placeholder token a model's template emits per image/video/audio.
    // Defaults follow the Qwen convention; set these for other model families.
    void set_media_pads(const std::string &image, const std::string &video, const std::string &audio)
    {
        image_pad_ = image;
        video_pad_ = video;
        audio_pad_ = audio;
    }

    // Render the conversation. `extra_context` can carry template flags such as
    // {"enable_thinking": false}; `tools` is the optional tool/function list that
    // tool-calling templates iterate over.
    std::string apply(const std::vector<Content> &contents,
                      bool add_generation_prompt = true,
                      const json &extra_context = json::object(),
                      const json &tools = json()) const
    {
        if (!tmpl_)
            return {};
        minja::chat_template_inputs in;
        in.messages = to_messages(contents);
        in.add_generation_prompt = add_generation_prompt;
        in.extra_context = extra_context;
        in.tools = tools;
        // Deterministic clock so renders are reproducible (matches minja's test mode).
        in.now = std::chrono::system_clock::from_time_t(0);

        minja::chat_template_options opt;
        opt.apply_polyfills = false;   // render the template verbatim, like HF tokenize=False
        opt.use_bos_token = false;     // the template emits bos/eos itself where it wants them
        opt.use_eos_token = false;
        std::string out = tmpl_->apply(in, opt);

        // The template emits one placeholder per media item; expand each to the
        // item's num_media_tokens copies (the per-image token count is a runtime
        // concern, exactly as HF's processor does it after apply_chat_template).
        expand_media_pads(out, contents);
        return out;
    }

private:
    std::shared_ptr<minja::chat_template> tmpl_;
    // Per-modality placeholder tokens (Qwen convention by default).
    std::string image_pad_ = "<|image_pad|>";
    std::string video_pad_ = "<|video_pad|>";
    std::string audio_pad_ = "<|audio_pad|>";

    // Replace each single placeholder the template emitted with num_media_tokens
    // copies, in render order (which equals Content order). Text-only -> no-op.
    void expand_media_pads(std::string &s, const std::vector<Content> &contents) const
    {
        struct Job { const std::string *pad; int count; };
        std::vector<Job> jobs;
        for (const auto &c : contents) {
            if (c.type == TEXT || c.num_media_tokens <= 0)
                continue;
            const std::string *pad = c.type == IMAGE ? &image_pad_
                                   : c.type == VIDEO ? &video_pad_ : &audio_pad_;
            if (pad->empty())
                continue;
            const int items = c.num_media > 0 ? c.num_media : 1;
            for (int i = 0; i < items; ++i)
                jobs.push_back({pad, c.num_media_tokens});
        }
        if (jobs.empty())
            return;
        size_t pos = 0;
        for (const auto &job : jobs) {
            const std::string &pad = *job.pad;
            const size_t p = s.find(pad, pos);
            if (p == std::string::npos)
                break; // template emitted fewer placeholders than expected
            std::string rep;
            rep.reserve(pad.size() * (size_t)job.count);
            for (int i = 0; i < job.count; ++i)
                rep += pad;
            s.replace(p, pad.size(), rep);
            pos = p + rep.size();
        }
    }

    static std::string role_name(RoleType r)
    {
        switch (r) {
            case SYSTEM: return "system";
            case ASSISTANT: return "assistant";
            case USER: default: return "user";
        }
    }

    // Map Content[] -> HF messages JSON. Text content becomes a plain string;
    // media content becomes the list-of-parts shape VLM templates iterate over.
    static json to_messages(const std::vector<Content> &contents)
    {
        json messages = json::array();
        for (const auto &c : contents) {
            json msg;
            msg["role"] = role_name(c.role);
            if (c.type == TEXT) {
                msg["content"] = c.data;
            } else {
                json parts = json::array();
                const char *t = c.type == IMAGE ? "image" : c.type == VIDEO ? "video" : "audio";
                for (int i = 0; i < (c.num_media > 0 ? c.num_media : 1); ++i)
                    parts.push_back({{"type", t}});
                if (!c.data.empty())
                    parts.push_back({{"type", "text"}, {"text", c.data}});
                msg["content"] = parts;
            }
            messages.push_back(std::move(msg));
        }
        return messages;
    }

    static std::string join(const std::string &a, const std::string &b)
    {
        if (a.empty()) return b;
        return a.back() == '/' ? a + b : a + "/" + b;
    }

    static bool read_json(const std::string &path, json &out)
    {
        std::ifstream f(path);
        if (!f) return false;
        std::stringstream ss; ss << f.rdbuf();
        try { out = json::parse(ss.str()); } catch (...) { return false; }
        return true;
    }

    // chat_template may be a string, or a list of {name, template} objects.
    static std::string extract_chat_template(const json &cfg)
    {
        auto it = cfg.find("chat_template");
        if (it == cfg.end()) return {};
        if (it->is_string()) return it->get<std::string>();
        if (it->is_array()) {
            for (const auto &e : *it)
                if (e.contains("name") && e["name"] == "default" && e.contains("template"))
                    return e["template"].get<std::string>();
            if (!it->empty() && (*it)[0].contains("template"))
                return (*it)[0]["template"].get<std::string>();
        }
        return {};
    }

    // bos_token/eos_token may be a string or an {content: "..."} object.
    static std::string token_str(const json &cfg, const char *key)
    {
        auto it = cfg.find(key);
        if (it == cfg.end() || it->is_null()) return {};
        if (it->is_string()) return it->get<std::string>();
        if (it->is_object() && it->contains("content")) return (*it)["content"].get<std::string>();
        return {};
    }
};
