# tokenizer
方便好用的纯纯 C++ tokenizer 封装

## 支持的模型

| ModelType | 名称 | 说明 |
|-----------|------|------|
| 0 | Qwen2_5 | Qwen2.5 系列 |
| 1 | Qwen3 | Qwen3 系列 |
| 2 | Qwen3VL | Qwen3-VL 多模态 |
| 3 | InternVL3 | InternVL3 多模态 |
| 4 | HunYuan | 混元系列 |
| 5 | SmolVLM2 | SmolVLM2 多模态 |
| 6 | FastVLM | FastVLM 多模态 |
| 7 | MiniCPM4 | MiniCPM4 文本 |
| 8 | MiniCPMV4 | MiniCPM-V4 多模态 |
| 9 | Gemma3 | Gemma3 文本 (1b) |
| 10 | Gemma3VL | Gemma3 多模态 (4b/12b/27b) |
| 11 | SmolLM2 | SmolLM2 系列 |
| 12 | SmolLM3 | SmolLM3 系列 |
| 13 | GLM5 | GLM-5 文本 |
| 14 | GLM5VL | GLM-5 多模态 |
| 15 | KimiK25 | Kimi-K2.5 文本 |
| 16 | KimiK25VL | Kimi-K2.5 多模态 |
| 17 | Qwen3Omni | Qwen3-Omni 多模态 |
| 18 | Qwen3_5 | Qwen3.5 文本（预留） |
| 19 | Qwen3_5VL | Qwen3.5 多模态 |
| 20 | MiniMaxM2 | MiniMax-M2 文本 |
| 21 | MiniMaxM2VL | MiniMax-M2 多模态 |
| 22 | MiniCPMO4_5 | MiniCPM-o-4_5 多模态 |
| 23 | InternVL3_5 | InternVL3.5 多模态（1B/2B 共用） |
| 24 | PaddleOCR_VL | Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing |
| 25 | Gemma4 | Gemma4 多模态 (google/gemma-4-*-it) |
| 26 | Gemma4VL | Gemma4 多模态 |
| 27 | MiniCPMV46 | MiniCPM-V4.6 多模态 |
| 28 | MiniCPMV46VL | MiniCPM-V4.6 多模态 |
| 29 | MiniCPM5 | MiniCPM5 文本 |

详见 [ModelType](include/BaseTokenizer.hpp#L6)

## 导出
导出模型的 `hf` 的路径的 `tokenizer` 到 `txt` 文件，方便在 C++ 中使用
```shell
cd tests
python convert_tokenizer.py --tokenizer_path Qwen/Qwen3-1.7B --dst_path assets/qwen3_tokenizer.txt 
```

## 使用
在 C++ 中使用导出的 `tokenizer` 文件，参考 `tests/main.cpp`
```C++
// 加载导出的 tokenizer 文件
std::shared_ptr<BaseTokenizer> tokenizer = create_tokenizer((ModelType)model_type.value());
if (!tokenizer->load("assets/qwen3_tokenizer.txt"))
{
    fprintf(stderr, "load tokenizer failed");
    return -1;
}

// 构建输入的 contents
std::vector<Content> contents = {
            {SYSTEM, TEXT, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {USER, TEXT, "你好"},
            {ASSISTANT, TEXT, "<think>\n\n</think>\n\n你好！有什么我可以帮助你的吗？"}};
std::vector<int> ids = tokenizer->encode(contents);
std::string text = tokenizer->decode(ids);
printf("text: \n%s\n", text.c_str());
```
输出
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮助你的吗？<|im_end|>
```
加入新的对话内容
```C++
// 添加新的 USER 输入
contents.push_back({USER, TEXT, "你能做什么"});
auto ids2 = tokenizer->encode(contents);
text = tokenizer->decode(ids2);
printf("text: \n%s\n", text.c_str());
```
输出
```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么我可以帮助你的吗？<|im_end|>
<|im_start|>user
你能做什么<|im_end|>
<|im_start|>assistant
```
VL模型如下
```C++
// 添加新的 IMAGE 输入，图片的数量为1，图片的 token 数量为 256
contents.push_back({USER, IMAGE, "帮我看看这张图片", 1, 256});

// 添加新的 VIDEO 输入，视频的图像数量为8，视频的 token 数量为 256
contents.push_back({USER, VIDEO, "帮我看看这个视频", 8, 256});
```
输出
```
过于先进，不便展示。
```


## 直接加载 HuggingFace 目录（通用路径，可选）

除了上面“转换成 `.txt` 再用”的精简路径，还提供一条**通用路径**：直接读取 HuggingFace 的 tokenizer 目录（`tokenizer.json` / `tokenizer_config.json` / `config.json`），**无需转换，也无需为新模型写代码**。底层分词用 [wangzhaode/tokenizer.cpp](https://github.com/wangzhaode/tokenizer.cpp)，对话模板用 [minja](https://github.com/google/minja) 渲染模型自带的 jinja 模板。

> 这条路径会引入 oniguruma / utf8proc 等依赖，**默认关闭**；边缘 / 嵌入式构建保持精简的 `.txt` 路径即可，完全不受影响。

### 构建
```shell
git submodule update --init                 # 拉取 third_party/tokenizer.cpp
cmake -DTOKENIZER_WITH_HF_LOADER=ON ..
make -j
```

### 使用
```C++
#include "HFAutoTokenizer.hpp"

HFAutoTokenizer tok;
tok.from_pretrained("/path/to/hf/tokenizer/dir");   // 直接读 HF 目录，无需转换

// 纯分词
std::vector<int> ids = tok.encode("你好");
std::string text     = tok.decode(ids);

// 对话模板（用模型自带 jinja，自动正确）
std::vector<Content> contents = {
    {SYSTEM, TEXT, "You are a helpful assistant."},
    {USER,   TEXT, "你好"},
};
std::string prompt    = tok.apply_chat_template(contents);  // 拼好的字符串
std::vector<int> cids = tok.encode_chat(contents);          // 渲染 + 编码

// thinking / 工具调用：通过 extra_context / tools 传入
tok.apply_chat_template(contents, true, {{"enable_thinking", false}});

// 多模态：pad token 自动从 config.json 读取，零配置
contents.push_back({USER, IMAGE, "看看这张图", /*num_media=*/1, /*num_media_tokens=*/256});
auto mids = tok.encode_chat(contents);
```

### 两条路径对比
| | 精简 `.txt` 路径 | 通用 HF 目录路径 |
|---|---|---|
| 入口 | `create_tokenizer(type)->load(".txt")` | `HFAutoTokenizer::from_pretrained(dir)` |
| 需要转换 | 是（`convert_tokenizer.py`） | 否 |
| 新模型 | 需注册 / 手写模板 | 多数零代码（仅特殊拼接逻辑例外） |
| 依赖 | 极少 | + oniguruma / utf8proc |
| 适合 | 边缘 / 嵌入式 | 服务器 / 开发，快速接入新模型 |
| CMake | 默认 | `-DTOKENIZER_WITH_HF_LOADER=ON` |

两套机制并存，互不影响；现有手写模板逻辑保持不变。

### 只要 jinja 对话模板
如果只想用 jinja 渲染对话模板（不更换分词引擎），用 `JinjaChatTemplate`（仅依赖 header-only 的 minja，`-DTOKENIZER_WITH_JINJA=ON`，默认开启）：
```C++
#include "JinjaChatTemplate.hpp"
JinjaChatTemplate ct;
ct.load_from_dir("/path/to/hf/dir");        // 或 ct.load(template_str, bos, eos)
std::string prompt = ct.apply(contents);
```

## reference
- [mnn-llm](https://github.com/wangzhaode/mnn-llm)
- [llm-export](https://github.com/wangzhaode/llm-export)
- [tokenizer.cpp](https://github.com/wangzhaode/tokenizer.cpp)
- [minja](https://github.com/google/minja)
