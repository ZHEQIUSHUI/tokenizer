#include "BaseTokenizer.hpp"
#include <memory>
#include <algorithm>

#include "utils/object_register.hpp"

#include "Qwen3Tokenizer.hpp"
#include "InternVL3Tokenizer.hpp"
#include "SmolVLM2Tokenizer.hpp"
#include "FastVLMTokenizer.hpp"
#include "HunYuanTokenizer.hpp"
#include "MiniCPM4Tokenizer.hpp"
#include "Gemma3Tokenizer.hpp"
#include "Gemma4Tokenizer.hpp"
#include "SmolLM2Tokenizer.hpp"
#include "SmolLM3Tokenizer.hpp"
#include "GLM5Tokenizer.hpp"
#include "KimiK25Tokenizer.hpp"
#include "MiniMaxM2Tokenizer.hpp"
#include "PaddleOCRVLTokenizer.hpp"
#include "MiniCPMV46Tokenizer.hpp"
#include "MiniCPM5Tokenizer.hpp"

std::shared_ptr<BaseTokenizer> create_tokenizer(ModelType type)
{
    delete_fun destroy_fun = nullptr;
    BaseTokenizer *tokenizer = (BaseTokenizer *)OBJFactory::getInstance().getObjectByID(type, destroy_fun);
    if (!tokenizer)
    {
        fprintf(stderr, "create tokenizer failed");
        return nullptr;
    }
    return std::shared_ptr<BaseTokenizer>(tokenizer, destroy_fun);
}

std::shared_ptr<BaseTokenizer> create_tokenizer(std::string type_name)
{
    delete_fun destroy_fun = nullptr;
    BaseTokenizer *tokenizer = (BaseTokenizer *)OBJFactory::getInstance().getObjectByName(type_name, destroy_fun);
    if (!tokenizer)
    {
        fprintf(stderr, "create tokenizer failed");
        return nullptr;
    }
    return std::shared_ptr<BaseTokenizer>(tokenizer, destroy_fun);
}
