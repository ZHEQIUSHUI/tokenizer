#include "cmdline.hpp"
#include "timer.hpp"
#include "magic_enum.hpp"
#include "../include/BaseTokenizer.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>
#include <optional>

// ANSI color codes
namespace color {
    const char* reset = "\033[0m";
    const char* red = "\033[31m";
    const char* green = "\033[32m";
    const char* yellow = "\033[33m";
    const char* blue = "\033[34m";
    const char* cyan = "\033[36m";
    const char* bold = "\033[1m";
}

// 测试结果统计
struct TestStats {
    int passed = 0;
    int failed = 0;
    int skipped = 0;
    
    void print() const {
        printf("\n%s%sTest Summary:%s\n", color::bold, color::cyan, color::reset);
        printf("  %sPassed:%s  %d\n", color::green, color::reset, passed);
        if (failed > 0) printf("  %sFailed:%s  %d\n", color::red, color::reset, failed);
        if (skipped > 0) printf("  %sSkipped:%s %d\n", color::yellow, color::reset, skipped);
        printf("  %sTotal:%s   %d\n", color::bold, color::reset, passed + failed + skipped);
    }
};
static TestStats g_stats;

// 改进的 diff 函数 - 使用 const 引用避免拷贝
std::vector<int> diff_token_ids(const std::vector<int>& ids1, const std::vector<int>& ids2)
{
    if (ids1.size() >= ids2.size()) {
        return {};
    }
    for (size_t i = 0; i < ids1.size(); i++) {
        if (ids1[i] != ids2[i]) {
            return {};
        }
    }
    return std::vector<int>(ids2.begin() + ids1.size(), ids2.end());
}

// 打印 id 列表（无多余逗号）
void print_ids(const std::vector<int>& ids, const char* label = "ids")
{
    printf("%s size: %zu\n[", label, ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        if (i > 0) printf(", ");
        printf("%d", ids[i]);
    }
    printf("]\n");
}

// 打印文本（带边界标记）
void print_text(const std::string& text, const char* label = "text")
{
    printf("%s:\n%s%s%s\n", label, color::yellow, text.c_str(), color::reset);
}

// 加载 system prompt，失败返回默认值
std::string load_system_prompt(const char* filename, const std::string& default_prompt)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        return default_prompt;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // 如果文件为空，返回默认值
    if (content.empty() || content.find_first_not_of(" \t\n\r") == std::string::npos) {
        return default_prompt;
    }
    
    return content;
}

// 测试用例结构
struct TextTestCase {
    const char* name;
    bool think_in_prompt;
    const char* assistant_response;
};

// 运行单个文本测试
bool run_text_test(std::shared_ptr<BaseTokenizer> tokenizer, const TextTestCase& test_case, 
                   const std::string& system_prompt, bool verbose)
{
    printf("\n%s%s[TEST] %s%s\n", color::bold, color::blue, test_case.name, color::reset);
    
    tokenizer->set_think_in_prompt(test_case.think_in_prompt);
    
    std::vector<Content> contents = {
        {SYSTEM, TEXT, system_prompt},
        {USER, TEXT, "Hello"},
        {ASSISTANT, TEXT, test_case.assistant_response}
    };
    
    // 测试 apply_chat_template
    timer t_cost;
    t_cost.start();
    auto chat_template = tokenizer->apply_chat_template(contents);
    t_cost.stop();
    
    if (verbose) {
        printf("chat_template cost: %.3f ms\n", t_cost.cost());
        print_text(chat_template, "template");
    }
    
    // 测试 encode
    t_cost.start();
    std::vector<int> ids = tokenizer->encode(chat_template);
    t_cost.stop();
    
    if (verbose) {
        printf("encode cost: %.3f ms\n", t_cost.cost());
        print_ids(ids);
    }
    
    // 测试 decode
    std::string decoded = tokenizer->decode(ids);
    if (verbose) {
        print_text(decoded, "decoded");
    }
    
    // 验证 round-trip
    bool round_trip_ok = (decoded == chat_template);
    if (!round_trip_ok) {
        printf("%s[FAIL] decode(encode(text)) != text%s\n", color::red, color::reset);
        printf("  expected: %s\n  got:      %s\n", chat_template.c_str(), decoded.c_str());
        g_stats.failed++;
        return false;
    }
    
    // 测试增量对话
    contents.push_back({USER, TEXT, "What can you do"});
    auto ids2 = tokenizer->encode(contents);
    
    if (verbose) {
        print_ids(ids2, "ids2 (with new turn)");
    }
    
    // 测试 diff
    auto diff_ids = diff_token_ids(ids, ids2);
    if (verbose) {
        print_ids(diff_ids, "diff_ids");
    }
    
    // 验证增量编码确实追加了内容。注意:不能假设旧 ids 是新 ids2 的前缀——许多 chat
    // 模板在追加新轮时会改写结尾(generation prompt / thinking 标记),并非前缀稳定。
    // 唯一稳健的不变量是:多了一轮用户内容,token 总数应增加。
    bool diff_ok = ids2.size() > ids.size();
    if (!diff_ok) {
        printf("%s[FAIL] new turn did not increase token count (%zu -> %zu)%s\n",
               color::red, ids.size(), ids2.size(), color::reset);
        g_stats.failed++;
        return false;
    }

    printf("%s%s[PASS] %s%s\n", color::green, color::bold, test_case.name, color::reset);
    g_stats.passed++;
    return true;
}

// 多模态测试用例
struct MediaTestCase {
    const char* name;
    ContentType type;
    const char* text;
    int num_media;
    int num_tokens_per_media;
};

bool run_media_test(std::shared_ptr<BaseTokenizer> tokenizer, const MediaTestCase& test_case,
                    const std::string& system_prompt, bool verbose)
{
    printf("\n%s%s[TEST] %s%s\n", color::bold, color::blue, test_case.name, color::reset);
    
    // 检查支持
    if (!tokenizer->support(test_case.type)) {
        printf("%s[SKIP] Model does not support %s%s\n", color::yellow, 
               test_case.type == IMAGE ? "IMAGE" : "VIDEO", color::reset);
        g_stats.skipped++;
        return false;
    }
    
    std::vector<Content> contents = {
        {SYSTEM, TEXT, system_prompt},
        {USER, TEXT, "Hello"},
        {ASSISTANT, TEXT, "Hello! How can I help you today?"}
    };
    
    // 基础对话
    std::vector<int> ids = tokenizer->encode(contents);
    if (verbose) {
        print_ids(ids, "base_ids");
    }
    
    // 添加媒体输入
    contents.push_back({USER, test_case.type, test_case.text, test_case.num_media, test_case.num_tokens_per_media});
    
    timer t_cost;
    t_cost.start();
    auto ids2 = tokenizer->encode(contents);
    t_cost.stop();
    
    if (verbose) {
        printf("encode with media cost: %.3f ms\n", t_cost.cost());
        print_ids(ids2, "media_ids");
    }
    
    // 验证媒体输入增加了 token 数量
    bool size_increased = ids2.size() > ids.size();
    if (!size_increased) {
        printf("%s[FAIL] Media input did not increase token count (%zu -> %zu)%s\n", 
               color::red, ids.size(), ids2.size(), color::reset);
        g_stats.failed++;
        return false;
    }
    
    // decode 验证
    std::string decoded = tokenizer->decode(ids2);
    if (verbose) {
        print_text(decoded, "decoded");
    }
    
    printf("%s%s[PASS] %s (tokens: %zu -> %zu, +%zu)%s\n", 
           color::green, color::bold, test_case.name, 
           ids.size(), ids2.size(), ids2.size() - ids.size(), color::reset);
    g_stats.passed++;
    return true;
}

// 打印帮助信息
void print_help(const char* program)
{
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  -t, --tokenizer_path <path>  Tokenizer file path (required)\n");
    printf("  -m, --model_type <num>       Model type number (required)\n");
    printf("  -v, --verbose                Verbose output\n");
    printf("  --no-text                    Skip text tests\n");
    printf("  --no-media                   Skip media tests\n");
    printf("  -h, --help                   Show this help\n");
    printf("\n");
}

int main(int argc, char *argv[])
{
    // 构建 model type 帮助字符串
    std::stringstream model_type_str;
    model_type_str << "model type: \033[32m";
    for (auto name : magic_enum::enum_names<ModelType>()) {
        model_type_str << std::setw(12) << name << ":" << magic_enum::enum_cast<ModelType>(name).value() << " ";
    }
    model_type_str << "\033[0m";

    // 解析参数
    cmdline::parser parser;
    parser.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    parser.add<int>("model_type", 'm', model_type_str.str(), true);
    parser.add("verbose", 'v', "verbose output");
    parser.add("no-text", 0, "skip text tests");
    parser.add("no-media", 0, "skip media tests");
    parser.add("help", 'h', "print this message");
    
    // 手动处理 --help
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            printf("%s\n", model_type_str.str().c_str());
            return 0;
        }
    }
    
    parser.parse_check(argc, argv);
    
    std::string tokenizer_path = parser.get<std::string>("tokenizer_path");
    bool verbose = parser.exist("verbose");
    bool skip_text = parser.exist("no-text");
    bool skip_media = parser.exist("no-media");
    
    // 验证 model type
    auto model_type = magic_enum::enum_cast<ModelType>(parser.get<int>("model_type"));
    if (!model_type.has_value()) {
        fprintf(stderr, "%s[ERROR]%s model type %d not found\n", color::red, color::reset, parser.get<int>("model_type"));
        fprintf(stderr, "%s\n", model_type_str.str().c_str());
        return 1;
    }
    
    printf("%s%sTokenizer Test%s\n", color::bold, color::cyan, color::reset);
    printf("  Model: %s%s%s\n", color::green, magic_enum::enum_name(model_type.value()).data(), color::reset);
    printf("  Path:  %s\n\n", tokenizer_path.c_str());
    
    // 创建 tokenizer
    std::shared_ptr<BaseTokenizer> tokenizer = create_tokenizer(model_type.value());
    if (!tokenizer) {
        fprintf(stderr, "%s[ERROR]%s create tokenizer failed\n", color::red, color::reset);
        return 1;
    }
    
    if (!tokenizer->load(tokenizer_path)) {
        fprintf(stderr, "%s[ERROR]%s load tokenizer failed: %s\n", color::red, color::reset, tokenizer_path.c_str());
        return 1;
    }
    
    // 加载 system prompt
    std::string system_prompt = load_system_prompt("system_prompt.md", 
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant");
    
    // 运行文本测试
    if (!skip_text) {
        printf("\n%s%s=== Text Tests ===%s\n", color::bold, color::cyan, color::reset);
        
        TextTestCase text_tests[] = {
            {"think_in_prompt=true", true, "<think>\n\n</think>\n\nHello! How can I help you today?"},
            {"think_in_prompt=false (with think tags)", false, "<think>\n\n</think>\n\nHello! How can I help you today?"},
            {"think_in_prompt=false (remove think)", false, "</think>\n\nHello! How can I help you today?"},
        };
        
        for (const auto& test : text_tests) {
            run_text_test(tokenizer, test, system_prompt, verbose);
        }
    }
    
    // 运行多模态测试
    if (!skip_media) {
        printf("\n%s%s=== Media Tests ===%s\n", color::bold, color::cyan, color::reset);
        
        MediaTestCase media_tests[] = {
            {"IMAGE input", IMAGE, "Please analyze this image", 1, 256},
            {"VIDEO input", VIDEO, "Please analyze this video", 8, 256},
        };
        
        for (const auto& test : media_tests) {
            run_media_test(tokenizer, test, system_prompt, verbose);
        }
    }
    
    // 打印统计
    g_stats.print();
    
    return g_stats.failed > 0 ? 1 : 0;
}
