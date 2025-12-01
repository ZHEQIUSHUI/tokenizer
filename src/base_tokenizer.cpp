#include "base_tokenizer.hpp"
#include <memory>
#include "utils/object_register.hpp"
#include "qwen3_tokenizer.hpp"
#include <algorithm>

std::shared_ptr<base_tokenizer> create_tokenizer(ModelType type)
{
    delete_fun destroy_fun = nullptr;
    base_tokenizer *tokenizer = (base_tokenizer *)OBJFactory::getInstance().getObjectByID(type, destroy_fun);
    if (!tokenizer)
    {
        fprintf(stderr, "create tokenizer failed");
        return nullptr;
    }
    return std::shared_ptr<base_tokenizer>(tokenizer, destroy_fun);
}

static std::string trim_copy(const std::string &s)
{
    size_t start = 0;
    while (start < s.size() && std::isspace((unsigned char)s[start]))
        start++;

    size_t end = s.size();
    while (end > start && std::isspace((unsigned char)s[end - 1]))
        end--;

    return s.substr(start, end - start);
}

static inline void trim_inplace(std::string &s)
{
    // 去掉前面空白
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch)
                                    { return !std::isspace(ch); }));
    // 去掉后面空白
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch)
                         { return !std::isspace(ch); })
                .base(),
            s.end());
}

std::string base_tokenizer::remove_thinking(const std::string &text, std::string end_think_token, bool trim)
{
    // 查找 token
    size_t pos = text.find(end_think_token);

    // 找不到 token：返回原文本（按需可更改）
    if (pos == std::string::npos)
    {
        std::string result = text;
        if (trim)
            trim_inplace(result);
        return result;
    }

    // 提取 token 之后的内容
    std::string result = text.substr(pos + end_think_token.size());

    // 可选 trim
    if (trim)
        trim_inplace(result);

    return result;
}
