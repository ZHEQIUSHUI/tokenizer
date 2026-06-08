#pragma once
#include <string>
#include <vector>
#include <memory>

enum ModelType
{
    Qwen2_5,      // 0
    Qwen3,        // 1
    Qwen3VL,      // 2
    InternVL3,    // 3
    HunYuan,      // 4
    SmolVLM2,     // 5
    FastVLM,      // 6
    MiniCPM4,     // 7
    MiniCPMV4,    // 8
    Gemma3,       // 9
    Gemma3VL,     // 10
    SmolLM2,      // 11
    SmolLM3,      // 12
    GLM5,         // 13
    GLM5VL,       // 14
    KimiK25,      // 15
    KimiK25VL,    // 16
    Qwen3Omni,    // 17
    Qwen3_5,      // 18, reserved for future text-only version
    Qwen3_5VL,    // 19
    MiniMaxM2,    // 20
    MiniMaxM2VL,  // 21
    MiniCPMO4_5,  // 22
    InternVL3_5,  // 23
    PaddleOCRVL,  // 24
    Gemma4,       // 25 (google/gemma-4-*-it)
    Gemma4VL,     // 26
    MiniCPMV46,   // 27
    MiniCPMV46VL, // 28
    MiniCPM5,     // 29
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

enum class ThinkingMode
{
    Unspecified,
    Think,
    NoThink,
};

struct Content
{
    RoleType role;
    ContentType type;
    std::string data;
    int num_media = 0;        // 多模态数量
    int num_media_tokens = 0; // 多模态token pad数量
};

class BaseTokenizer
{
protected:
    // 是否在上下文中保留thinking内容, 默认为false
    bool think_in_prompt = false;
    ThinkingMode generation_thinking_mode = ThinkingMode::Unspecified;

public:
    virtual bool load(const std::string tokenizer_path) = 0;
    virtual bool support(ContentType type) = 0;
    virtual void set_think_in_prompt(bool think_in_prompt)
    {
        this->think_in_prompt = think_in_prompt;
    }
    virtual void set_generation_thinking_mode(ThinkingMode mode)
    {
        this->generation_thinking_mode = mode;
    }

    virtual bool is_stop(int token) = 0;
    virtual void add_stop_token(int token) = 0;
    virtual bool add_stop_token(std::string stop_token) = 0;
    virtual void clear_addition_stop_tokens() = 0;
    virtual std::vector<int> get_stop_tokens() = 0;

    virtual std::vector<int> encode(const std::vector<Content> &contents, bool add_generation_prompt = true)
    {
        return encode(apply_chat_template(contents, add_generation_prompt));
    }

    virtual std::string apply_chat_template(const std::vector<Content> &contents, bool add_generation_prompt = true) = 0;
    virtual std::vector<int> encode(const std::string &text) = 0;
    virtual std::string decode(const std::vector<int> &ids) = 0;
    virtual std::string decode(int id) = 0;
};

std::shared_ptr<BaseTokenizer> create_tokenizer(ModelType type);
std::shared_ptr<BaseTokenizer> create_tokenizer(std::string type_name);
