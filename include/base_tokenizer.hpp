#pragma once
#include <string>
#include <vector>
#include <memory>

enum ModelType
{
    Qwen3,
};

enum RoleType
{
    SYSTEM,
    ASSISTANT,
    USER,
};

enum ContentType
{
    TEXT,
    AUDIO,
    IMAGE,
    VIDEO,
};

struct Content
{
    RoleType role;
    ContentType type;
    std::string data;
};

class base_tokenizer
{
protected:
    // 是否在prompt中包含think标签, 默认为false
    bool think_in_prompt = false;

    // 从文本中移除 start_think_token 和 end_think_token 之间的内容,并 trim 空格和换行符
    std::string remove_thinking(const std::string &text, std::string start_think_token = "<think>", std::string end_think_token = "</think>", bool trim = true);

public:
    virtual bool load(const std::string tokenizer_path) = 0;
    virtual bool support(ContentType type) = 0;
    virtual void set_think_in_prompt(bool think_in_prompt)
    {
        this->think_in_prompt = think_in_prompt;
    }

    virtual bool is_stop(int token) = 0;

    virtual std::vector<int> encode(const std::vector<Content> &contents) = 0;
    virtual std::string decode(const std::vector<int> &ids) = 0;
};

std::shared_ptr<base_tokenizer> create_tokenizer(ModelType type);
