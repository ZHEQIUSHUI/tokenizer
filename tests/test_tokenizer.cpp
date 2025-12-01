#include "cmdline.hpp"
#include "../include/base_tokenizer.hpp"

// ids1: {1,2,3}
// ids2: {1,2,3,4,5}
// diff_ids: {4,5}
// 这是用来比较两个 token id 序列的差异的函数，当新一轮对话用户输入prompt的时候，编码出来的token id 序列与上一轮的差异，
// 得到新增的token id 序列，这样历史对话的token id 序列就不需要重复计算kvcached了
std::vector<int> diff_token_ids(std::vector<int> ids1, std::vector<int> ids2)
{
    if (ids1.size() >= ids2.size())
    {
        return {};
    }
    for (int i = 0; i < ids1.size(); i++)
    {
        if (ids1[i] != ids2[i])
        {
            return {};
        }
    }
    std::vector<int> diff_ids(ids2.begin() + ids1.size(), ids2.end());
    return diff_ids;
}

int main(int argc, char *argv[])
{
    std::string tokenizer_path = "../tests/assets/qwen3_tokenizer.txt";
    cmdline::parser a;
    a.add<std::string>("tokenizer_path", 't', "tokenizer path", true);
    a.parse_check(argc, argv);
    tokenizer_path = a.get<std::string>("tokenizer_path");

    std::shared_ptr<base_tokenizer> tokenizer = create_tokenizer(Qwen3);
    if (!tokenizer->load(tokenizer_path))
    {
        fprintf(stderr, "load tokenizer failed");
        return -1;
    }

    // 保留 thinking 内容
    {
        tokenizer->set_think_in_prompt(true);
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, TEXT, "你能做什么"});
        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());

        auto diff_ids = diff_token_ids(ids, ids2);
        printf("diff_ids size: %ld\n{", diff_ids.size());
        for (auto id : diff_ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");
    }

    // 不保留 thinking 内容
    {
        tokenizer->set_think_in_prompt(false);
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, TEXT, "你能做什么"});
        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());

        auto diff_ids = diff_token_ids(ids, ids2);
        printf("diff_ids size: %ld\n{", diff_ids.size());
        for (auto id : diff_ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");
    }

    // 不保留 thinking 内容
    {
        tokenizer->set_think_in_prompt(false);
        std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "</think>\n\n你好！有什么我可以帮助你的吗？"}};

        std::vector<int> ids = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids.size());
        for (auto id : ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        std::string text = tokenizer->decode(ids);
        printf("text: \n%s\n", text.c_str());

        contents.push_back({USER, TEXT, "你能做什么"});
        auto ids2 = tokenizer->encode(contents);
        printf("ids size: %ld\n{", ids2.size());
        for (auto id : ids2)
        {
            printf("%d, ", id);
        }
        printf("}\n");

        text = tokenizer->decode(ids2);
        printf("text: \n%s\n", text.c_str());

        auto diff_ids = diff_token_ids(ids, ids2);
        printf("diff_ids size: %ld\n{", diff_ids.size());
        for (auto id : diff_ids)
        {
            printf("%d, ", id);
        }
        printf("}\n");
    }

    return 0;
}