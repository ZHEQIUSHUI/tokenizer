#include "base_tokenizer.hpp"
#include <memory>
#include "utils/object_register.hpp"
#include "qwen3_tokenizer.hpp"

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

std::string base_tokenizer::remove_thinking(const std::string &text, std::string start_think_token, std::string end_think_token, bool trim)
{
    std::string result;
    result.reserve(text.size());

    size_t pos = 0;
    while (true)
    {
        size_t start = text.find(start_think_token, pos);
        if (start == std::string::npos)
        {
            result.append(text.substr(pos));
            break;
        }

        // append before <think>
        result.append(text.substr(pos, start - pos));

        // find </think>
        size_t end = text.find(end_think_token, start + start_think_token.length());
        if (end == std::string::npos)
        {
            // No closing tag → keep rest
            break;
        }

        // skip the whole <think>...</think>
        pos = end + end_think_token.length();
    }

    // trim
    if (trim)
    {
        return trim_copy(result);
    }
    return result;
}
