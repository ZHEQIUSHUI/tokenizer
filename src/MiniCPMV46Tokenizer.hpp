#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class MiniCPMV46Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string image_pad_token = "<|image_pad|>";
    std::string video_pad_token = "<|video_pad|>";
    std::string image_start_token = "<image>";
    std::string image_end_token = "</image>";
    std::string image_id_start_token = "<image_id>";
    std::string image_id_end_token = "</image_id>";

    static int media_tokens_or_default(const Content &content)
    {
        return std::max(1, content.num_media_tokens);
    }

    static int media_count_or_default(const Content &content)
    {
        return std::max(1, content.num_media);
    }

    std::string render_media_content(const Content &content,
                                     const std::string &pad_token,
                                     bool use_image_id,
                                     bool newline_between_media) const
    {
        std::stringstream ss;
        const int media_count = media_count_or_default(content);
        const int media_tokens = media_tokens_or_default(content);

        for (int media_index = 0; media_index < media_count; ++media_index)
        {
            if (media_index > 0 && newline_between_media) ss << "\n";
            if (use_image_id)
            {
                ss << image_id_start_token
                   << media_index
                   << image_id_end_token;
            }

            ss << image_start_token;
            for (int token_index = 0; token_index < media_tokens; ++token_index)
            {
                ss << pad_token;
            }
            ss << image_end_token;
        }

        if (!content.data.empty())
        {
            if (media_count > 0) ss << "\n";
            ss << content.data;
        }

        return ss.str();
    }

public:
    std::string decode(int id) override
    {
        // MNN gpt2_byte_bpe decode drops these underscore tokens when decoded
        // one-by-one, but axllm streams tokens incrementally. Keep MiniCPM-V 4.6
        // streaming text aligned with HF decoding for common identifier tokens.
        if (id == 62) return "_";
        if (id == 11481) return "_MIN";
        return BaseMixinTokenizer<Types...>::decode(id);
    }

    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        for (const auto &content : contents)
        {
            if (!this->support(content.type))
            {
                ALOGE("unsupport content type: %d", content.type);
                return {};
            }
        }

        std::stringstream text;
        for (const auto &content : contents)
        {
            if (content.role == SYSTEM)
            {
                text << "<|im_start|>system\n"
                     << content.data << "<|im_end|>\n";
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
                         << render_media_content(content, image_pad_token, true, true)
                         << "<|im_end|>\n";
                    break;
                case VIDEO:
                    text << "<|im_start|>user\n"
                         << render_media_content(content, video_pad_token, false, false)
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

        if (!contents.empty() && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n"
                 << "<think>\n\n</think>\n\n";
        }

        return text.str();
    }
};

using minicpmv46_tokenizer = MiniCPMV46Tokenizer<TEXT>;
REGISTER(MiniCPMV46, minicpmv46_tokenizer)

using minicpmv46vl_tokenizer = MiniCPMV46Tokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(MiniCPMV46VL, minicpmv46vl_tokenizer)
