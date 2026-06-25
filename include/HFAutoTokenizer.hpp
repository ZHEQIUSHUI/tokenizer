#pragma once
// General "load a HuggingFace tokenizer directory and just use it" entry point.
//
//   encode/decode        -> wangzhaode/tokenizer.cpp (reads tokenizer.json directly,
//                           handles BPE/WordPiece/Unigram + normalizers, no conversion)
//   apply_chat_template  -> minja, rendering the model's own chat_template (more robust
//                           than the bundled jinja; see JinjaChatTemplate.hpp)
//
// This is the "general leg": no per-model C++ for new models. The only thing that
// still needs per-model handling is multimodal pad-token expansion (the chat template
// emits one placeholder per image; expanding it to N is a runtime concern) -- see
// encode_chat() below.
//
// Optional: requires TOKENIZER_WITH_HF_LOADER (pulls in third_party/tokenizer.cpp +
// oniguruma). Off-by-default builds keep the lean .txt + hand-written path.

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "BaseTokenizer.hpp"        // Content
#include "tokenizer.hpp"            // wangzhaode: tokenizer::AutoTokenizer
#include "JinjaChatTemplate.hpp"    // minja-based chat templating

class HFAutoTokenizer
{
public:
    using json = JinjaChatTemplate::json;

    // Load tokenizer + chat template from a HuggingFace tokenizer directory.
    bool from_pretrained(const std::string &dir)
    {
        tok_ = tokenizer::AutoTokenizer::from_pretrained(dir);
        if (!tok_)
            return false;
        chat_.load_from_dir(dir);   // optional; ok if the model has no chat_template
        detect_media_pads(dir);     // zero-config multimodal: read pad tokens from config.json
        return true;
    }

    bool ready() const { return tok_ != nullptr; }
    bool has_chat_template() const { return chat_.ready(); }

    std::vector<int> encode(const std::string &text, bool add_special_tokens = true) const
    {
        return tok_ ? tok_->encode(text, add_special_tokens) : std::vector<int>{};
    }

    std::string decode(const std::vector<int> &ids, bool skip_special_tokens = true) const
    {
        return tok_ ? tok_->decode(ids, skip_special_tokens) : std::string{};
    }

    // Render the conversation to a prompt string via the model's own jinja template.
    // extra_context passes template flags (e.g. {"enable_thinking": false}); tools is
    // the optional function list for tool-calling templates.
    std::string apply_chat_template(const std::vector<Content> &contents,
                                    bool add_generation_prompt = true,
                                    const json &extra_context = json::object(),
                                    const json &tools = json()) const
    {
        return chat_.apply(contents, add_generation_prompt, extra_context, tools);
    }

    // Render + tokenize. add_special_tokens=false because the chat template already
    // emits the special tokens as text; wangzhaode still recognises them as single ids.
    std::vector<int> encode_chat(const std::vector<Content> &contents,
                                 bool add_generation_prompt = true,
                                 const json &extra_context = json::object(),
                                 const json &tools = json()) const
    {
        return encode(apply_chat_template(contents, add_generation_prompt, extra_context, tools),
                      /*add_special_tokens=*/false);
    }

private:
    std::shared_ptr<tokenizer::PreTrainedTokenizer> tok_;
    JinjaChatTemplate chat_;

    // Auto-discover the per-modality placeholder tokens from config.json so multimodal
    // works with no per-model setup. VL configs carry e.g. image_token_id / video_token_id
    // (Qwen) or image_token_index (Llava); we decode those ids to their token strings.
    // Falls back to the JinjaChatTemplate defaults (Qwen convention) when absent.
    void detect_media_pads(const std::string &dir)
    {
        json cfg;
        {
            std::ifstream f(dir + "/config.json");
            if (!f) return;
            std::stringstream ss; ss << f.rdbuf();
            try { cfg = json::parse(ss.str()); } catch (...) { return; }
        }
        // Search top level and one level of nesting (e.g. text_config / vision_config).
        auto find_id = [&](std::initializer_list<const char *> keys) -> int {
            for (const auto &kv : cfg.items()) {
                const json &v = kv.value();
                for (const char *k : keys) {
                    if (kv.key() == k && v.is_number_integer()) return v.get<int>();
                    if (v.is_object() && v.contains(k) && v[k].is_number_integer())
                        return v[k].get<int>();
                }
            }
            return -1;
        };
        auto pad_for = [&](std::initializer_list<const char *> keys, const std::string &dflt) {
            int id = find_id(keys);
            if (id < 0) return dflt;
            std::string s = tok_->decode({id}, /*skip_special_tokens=*/false);
            return s.empty() ? dflt : s;
        };
        chat_.set_media_pads(
            pad_for({"image_token_id", "image_token_index"}, "<|image_pad|>"),
            pad_for({"video_token_id", "video_token_index"}, "<|video_pad|>"),
            pad_for({"audio_token_id", "audio_token_index"}, "<|audio_pad|>"));
    }
};
