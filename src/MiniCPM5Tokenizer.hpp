#pragma once
#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class MiniCPM5Tokenizer : public BaseMixinTokenizer<Types...>
{
public:
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
        text << "<s>";
        ThinkingMode think_mode = this->generation_thinking_mode;
        int last_user_index = -1;
        for (size_t i = 0; i < contents.size(); ++i)
        {
            if (contents[i].role == USER)
            {
                last_user_index = static_cast<int>(i);
            }
        }

        for (size_t i = 0; i < contents.size(); ++i)
        {
            const auto &content = contents[i];
            if (content.type != TEXT)
            {
                ALOGE("MiniCPM5 only supports text content, got type: %d", content.type);
                return {};
            }

            if (content.role == SYSTEM)
            {
                text << "<|im_start|>system\n"
                     << content.data << "<|im_end|>\n";
            }
            else if (content.role == USER)
            {
                std::string user_text = content.data;
                if (i == contents.size() - 1)
                {
                    strip_think_control(user_text, think_mode, this->generation_thinking_mode == ThinkingMode::Unspecified);
                }
                text << "<|im_start|>user\n"
                     << user_text << "<|im_end|>\n";
            }
            else if (content.role == ASSISTANT)
            {
                // The official MiniCPM5 template strips reasoning from assistant
                // history before the last user turn, and only keeps it for
                // assistant prefill turns after the last user.
                const bool after_last_user = last_user_index >= 0 && static_cast<int>(i) > last_user_index;
                std::string assistant_text = after_last_user ? content.data : this->remove_thinking(content.data);
                text << "<|im_start|>assistant\n"
                     << assistant_text << "<|im_end|>\n";
            }
        }

        if (!contents.empty() && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
            if (think_mode == ThinkingMode::NoThink)
            {
                text << "<think>\n\n</think>\n\n";
            }
            else if (think_mode == ThinkingMode::Think)
            {
                text << "<think>\n";
            }
        }

        return text.str();
    }

private:
    static std::string trim(std::string text)
    {
        auto not_space = [](unsigned char c) { return !std::isspace(c); };
        text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
        text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
        return text;
    }

    static bool erase_marker(std::string &text, const std::string &marker)
    {
        const size_t pos = text.find(marker);
        if (pos == std::string::npos)
        {
            return false;
        }
        text.erase(pos, marker.size());
        text = trim(text);
        return true;
    }

    static void strip_think_control(std::string &text, ThinkingMode &think_mode, bool marker_controls_mode)
    {
        if (erase_marker(text, "/no_think"))
        {
            if (marker_controls_mode)
            {
                think_mode = ThinkingMode::NoThink;
            }
            return;
        }
        if (erase_marker(text, "/think"))
        {
            if (marker_controls_mode)
            {
                think_mode = ThinkingMode::Think;
            }
        }
    }
};

using minicpm5_tokenizer = MiniCPM5Tokenizer<TEXT>;
REGISTER(MiniCPM5, minicpm5_tokenizer)
