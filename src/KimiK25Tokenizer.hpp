#pragma once
#include <sstream>
#include <algorithm>

#include "BaseMixinTokenizer.hpp"
#include "utils/object_register.hpp"
#include "utils/sample_log.h"

template <ContentType... Types>
class KimiK25Tokenizer : public BaseMixinTokenizer<Types...>
{
protected:
    std::string media_begin_token = "<|media_begin|>";
    std::string media_content_token = "<|media_content|>";
    std::string media_pad_token = "<|media_pad|>";
    std::string media_end_token = "<|media_end|>";
    std::string video_placeholder = "<|kimi_k25_video_placeholder|>";

public:
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

        std::stringstream text;
        
        // Find the last non-tool-call assistant message index
        int last_non_tool_assistant_index = -1;
        for (int i = static_cast<int>(contents.size()) - 1; i >= 0; i--)
        {
            if (contents[i].role == ASSISTANT)
            {
                last_non_tool_assistant_index = i;
                break;
            }
        }

        // Split messages into history (up to and including last non-tool assistant) 
        // and suffix (after last non-tool assistant)
        size_t history_end = (last_non_tool_assistant_index >= 0) 
            ? static_cast<size_t>(last_non_tool_assistant_index + 1) 
            : 0;

        // Process history messages
        for (size_t i = 0; i < history_end && i < contents.size(); i++)
        {
            const auto &content = contents[i];
            
            if (content.role == SYSTEM)
            {
                text << "<|im_system|>system<|im_middle|>" 
                     << content.data << "<|im_end|>";
            }
            else if (content.role == USER)
            {
                text << "<|im_user|>user<|im_middle|>";
                
                switch (content.type)
                {
                case TEXT:
                    text << content.data;
                    break;
                case IMAGE:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << media_begin_token << "image" << media_content_token 
                             << media_pad_token << media_end_token;
                    }
                    text << content.data;
                    break;
                case VIDEO:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << video_placeholder;
                    }
                    text << content.data;
                    break;
                default:
                    text << content.data;
                    break;
                }
                
                text << "<|im_end|>";
            }
            else if (content.role == ASSISTANT)
            {
                text << "<|im_assistant|>assistant<|im_middle|><think></think>";
                
                if (!this->think_in_prompt)
                {
                    auto cleaned_data = this->remove_thinking(content.data);
                    text << cleaned_data;
                }
                else
                {
                    text << content.data;
                }
                
                text << "<|im_end|>";
            }
        }

        // Process suffix messages (after last non-tool assistant)
        for (size_t i = history_end; i < contents.size(); i++)
        {
            const auto &content = contents[i];
            
            if (content.role == SYSTEM)
            {
                text << "<|im_system|>system<|im_middle|>" 
                     << content.data << "<|im_end|>";
            }
            else if (content.role == USER)
            {
                text << "<|im_user|>user<|im_middle|>";
                
                switch (content.type)
                {
                case TEXT:
                    text << content.data;
                    break;
                case IMAGE:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << media_begin_token << "image" << media_content_token 
                             << media_pad_token << media_end_token;
                    }
                    text << content.data;
                    break;
                case VIDEO:
                    for (int j = 0; j < content.num_media; j++)
                    {
                        text << video_placeholder;
                    }
                    text << content.data;
                    break;
                default:
                    text << content.data;
                    break;
                }
                
                text << "<|im_end|>";
            }
            else if (content.role == ASSISTANT)
            {
                text << "<|im_assistant|>assistant<|im_middle|>";
                
                // For suffix messages, preserve reasoning_content if present
                std::string reasoning_content;
                std::string actual_content = content.data;
                
                // Extract reasoning content from <think> tags if present
                size_t think_start = content.data.find("<think>");
                size_t think_end = content.data.find("</think>");
                
                if (think_start != std::string::npos && think_end != std::string::npos && think_end > think_start)
                {
                    reasoning_content = content.data.substr(think_start + 7, think_end - think_start - 7);
                    actual_content = content.data.substr(think_end + 8);
                    // Trim leading whitespace
                    size_t content_start = actual_content.find_first_not_of(" \n\r\t");
                    if (content_start != std::string::npos)
                    {
                        actual_content = actual_content.substr(content_start);
                    }
                }
                
                if (!reasoning_content.empty())
                {
                    text << "<think>" << reasoning_content << "</think>";
                }
                else
                {
                    text << "<think></think>";
                }
                
                if (!this->think_in_prompt)
                {
                    auto cleaned_data = this->remove_thinking(actual_content);
                    text << cleaned_data;
                }
                else
                {
                    text << actual_content;
                }
                
                text << "<|im_end|>";
            }
        }

        // Add generation prompt
        if (contents.size() > 0 && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_assistant|>assistant<|im_middle|><think>";
        }

        return text.str();
    }
};

// Register tokenizer variants - all use the same underlying tokenizer file
using kimik25_tokenizer = KimiK25Tokenizer<TEXT>;
REGISTER(KimiK25, kimik25_tokenizer)

using kimik25vl_tokenizer = KimiK25Tokenizer<TEXT, IMAGE, VIDEO>;
REGISTER(KimiK25VL, kimik25vl_tokenizer)
