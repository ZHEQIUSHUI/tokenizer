#pragma once
#include <sstream>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

// Phi-4 / Phi-4-mini chat-template tokenizer.
//
// Scope note: only Phi-4 and Phi-4-mini are supported here. Both use an HF
// byte-level BPE (converted to a HUGGINGFACE .txt) whose encode/decode matches
// transformers exactly. Phi-3 / Phi-3.5 (SentencePiece) are intentionally NOT
// supported: the core SentencePiece engine omits the Llama-style dummy-prefix
// space, so its ids diverge from transformers for those models.
//
// The actual encode/decode is handled by the core engine from the converted
// .txt; this class only supplies the per-model chat template and stop tokens:
//   Phi4     : <|im_start|>{role}<|im_sep|>{content}<|im_end|>  gen: <|im_start|>assistant<|im_sep|>
//   Phi4Mini : <|{role}|>{content}<|end|>                       gen: <|assistant|>
enum class PhiChatStyle
{
    Phi4,     // microsoft/phi-4 (im_start / im_sep / im_end)
    Phi4Mini, // microsoft/Phi-4-mini-instruct (<|role|> ... <|end|>)
};

template <PhiChatStyle Style, ContentType... Types>
class PhiTokenizer : public BaseMixinTokenizer<Types...>
{
public:
    bool load(const std::string tokenizer_path) override
    {
        if (!BaseMixinTokenizer<Types...>::load(tokenizer_path))
        {
            return false;
        }
        // End-of-turn / eos markers as stop tokens. add_stop_token(string) is a
        // no-op for any marker that doesn't map to a single id in this vocab.
        if (Style == PhiChatStyle::Phi4)
        {
            this->add_stop_token(std::string("<|im_end|>"));
        }
        else // Phi4Mini
        {
            this->add_stop_token(std::string("<|end|>"));
            this->add_stop_token(std::string("<|endoftext|>"));
        }
        return true;
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
            const char *role = role_name(content.role);
            if (!role)
            {
                continue;
            }

            if (Style == PhiChatStyle::Phi4)
            {
                text << "<|im_start|>" << role << "<|im_sep|>" << content.data << "<|im_end|>";
            }
            else // Phi4Mini
            {
                text << "<|" << role << "|>" << content.data << "<|end|>";
            }
        }

        if (add_generation_prompt)
        {
            if (Style == PhiChatStyle::Phi4)
            {
                text << "<|im_start|>assistant<|im_sep|>";
            }
            else // Phi4Mini
            {
                text << "<|assistant|>";
            }
        }

        return text.str();
    }

private:
    static const char *role_name(RoleType role)
    {
        switch (role)
        {
        case SYSTEM:
            return "system";
        case USER:
            return "user";
        case ASSISTANT:
            return "assistant";
        default:
            return nullptr;
        }
    }
};

using phi4_tokenizer = PhiTokenizer<PhiChatStyle::Phi4, TEXT>;
REGISTER(Phi4, phi4_tokenizer)
using phi4_mini_tokenizer = PhiTokenizer<PhiChatStyle::Phi4Mini, TEXT>;
REGISTER(Phi4Mini, phi4_mini_tokenizer)
