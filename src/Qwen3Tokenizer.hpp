#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"
#include "utils/json.hpp"

template <ContentType... Types>
class Qwen3Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string video_pad_token = "<|video_pad|>";
    std::string image_pad_token = "<|image_pad|>";
    std::string audio_pad_token = "<|audio_pad|>";

    std::string img_start_token = "<|vision_start|>";
    std::string img_end_token = "<|vision_end|>";
    std::string audio_start_token = "<|audio_start|>";
    std::string audio_end_token = "<|audio_end|>";

public:
    bool load(const std::string tokenizer_path) override
    {
        if (!BaseMixinTokenizer<Types...>::load(tokenizer_path)) {
            return false;
        }
        adapt_placeholder_tokens_for_runtime();
        return true;
    }

    static bool starts_with_jina_embedding_prefix(const std::string &text)
    {
        return text.rfind("Query: ", 0) == 0 || text.rfind("Document: ", 0) == 0;
    }

private:
    bool has_single_token_id(const std::string &text) const
    {
        auto ids = this->tokenizer->encode(text);
        return ids.size() == 1;
    }

    void adapt_placeholder_tokens_for_runtime()
    {
        if (!has_single_token_id(image_pad_token) && has_single_token_id("<image>")) {
            image_pad_token = "<image>";
        }
        if (!has_single_token_id(video_pad_token)) {
            if (has_single_token_id("<video>")) {
                video_pad_token = "<video>";
            } else if (has_single_token_id(image_pad_token)) {
                video_pad_token = image_pad_token;
            }
        }
        if (!has_single_token_id(img_start_token)) {
            img_start_token.clear();
        }
        if (!has_single_token_id(img_end_token)) {
            img_end_token.clear();
        }
    }

public:

    // Render the request's tools (Qwen3 format) into the system-prompt tool section.
    std::string render_tools_section() const
    {
        if (this->tools_json_.empty()) return {};
        nlohmann::json tools;
        try { tools = nlohmann::json::parse(this->tools_json_); } catch (...) { return {}; }
        if (!tools.is_array() || tools.empty()) return {};
        std::string s;
        s += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n";
        s += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto &t : tools) s += "\n" + t.dump();
        s += "\n</tools>\n\n";
        s += "For each function call, return a json object with function name and arguments within "
             "<tool_call></tool_call> XML tags:\n<tool_call>\n"
             "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";
        return s;
    }

    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        // check contents type
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        const std::string tools_section = render_tools_section();
        bool has_system = false;
        for (const auto &c : contents) if (c.role == SYSTEM) { has_system = true; break; }

        std::stringstream text;
        if (!tools_section.empty() && !has_system)
            text << "<|im_start|>system\nYou are a helpful assistant." << tools_section << "<|im_end|>\n";
        for (const auto &content : contents)
        {
            if (content.role == SYSTEM)
            {
                text << "<|im_start|>system\n"
                     << content.data;
                if (!tools_section.empty()) text << tools_section;
                text << "<|im_end|>\n";
            }
            else if (content.role == USER)
            {
                switch (content.type)
                {
                case TEXT:
                    text << "<|im_start|>user\n"
                         << content.data << "<|im_end|>\n";
                    break;
                case IMAGE:
                    text << "<|im_start|>user\n"
                         << (starts_with_jina_embedding_prefix(content.data) ? content.data : std::string())
                         << img_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << image_pad_token;
                    }
                    text << img_end_token
                         << (!starts_with_jina_embedding_prefix(content.data) ? content.data : std::string())
                         << "<|im_end|>\n";
                    break;
                case VIDEO:
                    text << "<|im_start|>user\n"
                         << (starts_with_jina_embedding_prefix(content.data) ? content.data : std::string())
                         << img_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << video_pad_token;
                    }
                    text << img_end_token
                         << (!starts_with_jina_embedding_prefix(content.data) ? content.data : std::string())
                         << "<|im_end|>\n";
                    break;
                case AUDIO:
                    text << "<|im_start|>user\n"
                         << (starts_with_jina_embedding_prefix(content.data) ? content.data : std::string())
                         << audio_start_token;
                    for (int i = 0; i < content.num_media * content.num_media_tokens; i++)
                    {
                        text << audio_pad_token;
                    }
                    text << audio_end_token
                         << (!starts_with_jina_embedding_prefix(content.data) ? content.data : std::string())
                         << "<|im_end|>\n";
                    break;
                default:
                    break;
                }
            }
            else if (content.role == ASSISTANT)
            {
                if (!this->think_in_prompt)
                {
                    auto cleaned_data = this->remove_thinking(content.data);
                    text << "<|im_start|>assistant\n"
                         << cleaned_data << "<|im_end|>\n";
                }
                else
                {
                    text << "<|im_start|>assistant\n"
                         << content.data << "<|im_end|>\n";
                }
            }
        }

        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
            // Honor the thinking toggle, matching the official Qwen3 chat
            // template: enable_thinking=false injects an empty think block so the
            // model skips reasoning; think/unspecified keeps native behavior.
            if (this->generation_thinking_mode == ThinkingMode::NoThink)
            {
                text << "<think>\n\n</think>\n\n";
            }
        }

        // ALOGD("text: \n%s", text.str().c_str());
        return text.str();
    }

    bool supports_thinking_toggle() const override
    {
        return true;
    }
};
using qwen3vl_tokenizer = Qwen3Tokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(Qwen3VL, qwen3vl_tokenizer)
using qwen3_tokenizer = Qwen3Tokenizer<TEXT>;
REGISTER(Qwen3, qwen3_tokenizer)
using qwen2_5_tokenizer = Qwen3Tokenizer<TEXT>;
REGISTER(Qwen2_5, qwen2_5_tokenizer)
using qwen3omni_tokenizer = Qwen3Tokenizer<TEXT, IMAGE, VIDEO, AUDIO>;
REGISTER(Qwen3Omni, qwen3omni_tokenizer)
using qwen3_5_tokenizer = Qwen3Tokenizer<TEXT>;
REGISTER(Qwen3_5, qwen3_5_tokenizer)
using qwen3_5vl_tokenizer = Qwen3Tokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(Qwen3_5VL, qwen3_5vl_tokenizer)
