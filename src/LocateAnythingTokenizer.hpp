#pragma once
#include "Qwen3Tokenizer.hpp"

// LocateAnything-3B (nvidia/LocateAnything-3B, Qwen2.5-3B backbone) grounding / detection VLM.
// Direct port of infer_locateanything_axengine.py build_locateanything_prompt():
//   <|im_start|>system\n{system}\n<|im_end|>\n
//   <|im_start|>user\n<image N><img><IMG_CONTEXT>*tokens</img>{prompt}<|im_end|>\n
//   <|im_start|>assistant\n
// Image encoder embeddings are injected at the <IMG_CONTEXT> (id 151665) positions.
class LocateAnythingTokenizer : public Qwen3Tokenizer<TEXT, IMAGE>
{
public:
    LocateAnythingTokenizer()
    {
        image_pad_token = "<IMG_CONTEXT>";
        img_start_token = "<img>";
        img_end_token = "</img>";
    }

    std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt) override
    {
        for (const auto &content : contents)
        {
            if (!support(content.type))
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
                // Reference keeps the trailing newline before <|im_end|>.
                text << "<|im_start|>system\n"
                     << content.data << "\n<|im_end|>\n";
            }
            else if (content.role == USER)
            {
                text << "<|im_start|>user\n";
                if (content.type == IMAGE)
                {
                    for (int i = 0; i < content.num_media; i++)
                    {
                        text << "<image " << (i + 1) << ">" << img_start_token;
                        for (int j = 0; j < content.num_media_tokens; j++)
                        {
                            text << image_pad_token;
                        }
                        text << img_end_token;
                    }
                }
                text << content.data << "<|im_end|>\n";
            }
            else if (content.role == ASSISTANT)
            {
                text << "<|im_start|>assistant\n"
                     << content.data << "<|im_end|>\n";
            }
        }

        if (!contents.empty() && contents.back().role == USER && add_generation_prompt)
        {
            text << "<|im_start|>assistant\n";
        }

        return text.str();
    }
};

using locateanything_tokenizer = LocateAnythingTokenizer;
REGISTER(LocateAnything, locateanything_tokenizer)
