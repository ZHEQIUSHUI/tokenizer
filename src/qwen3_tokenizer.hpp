#pragma once
#include <sstream>

#include "base_tokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"
#include "tokenizer/tokenizer.hpp"

class qwen3_tokenizer : public base_tokenizer
{
private:
    std::shared_ptr<MNN::Transformer::Tokenizer> tokenizer;

public:
    bool load(const std::string tokenizer_path) override
    {
        tokenizer.reset(MNN::Transformer::Tokenizer::createTokenizer(tokenizer_path));
        return tokenizer != nullptr;
    }

    bool support(ContentType type) override
    {
        return type == TEXT;
    }

    bool is_stop(int token) override
    {
        return tokenizer->is_stop(token);
    }

    std::vector<int> encode(const std::vector<Content> &contents) override
    {
        // check contents type
        for (const auto &content : contents)
        {
            if (!support(content.type))
            {
                ALOGE("qwen3_tokenizer only support TEXT type");
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
                text << "<|im_start|>user\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == ASSISTANT)
            {
                if (!think_in_prompt)
                {
                    auto cleaned_data = remove_thinking(content.data);
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
        ALOGD("text: \n%s", text.str().c_str());
        return tokenizer->encode(text.str());
    }
    std::string decode(const std::vector<int> &ids) override
    {
        std::stringstream text;
        for (auto id : ids)
        {
            text << tokenizer->decode(id);
        }
        return text.str();
    }
};
REGISTER(Qwen3, qwen3_tokenizer)