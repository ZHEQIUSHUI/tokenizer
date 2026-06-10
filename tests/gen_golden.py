#!/usr/bin/env python3
"""Generate golden token-id baselines for the HF-parity regression test.

For each supported model we record the *reference* token ids that the upstream
HuggingFace tokenizer produces for tests/golden/corpus.txt. These golden files
are committed, so tests/test_hf_parity.cpp can verify the C++ engine reproduces
HuggingFace exactly without needing transformers / network at test time.

This script is a developer tool: it needs the HF tokenizer.json for each model.
Edit MODELS below to point at local tokenizer.json paths (HF cache snapshots,
or a dir produced by `AutoTokenizer.from_pretrained(...).save_pretrained(...)`).

Run from the repo root:  python3 tests/gen_golden.py
"""
import os
import sys

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("pip install tokenizers")

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/huggingface/hub")

def snap(repo, name):
    base = os.path.join(HUB, f"models--{repo}", "snapshots")
    if os.path.isdir(base):
        for s in os.listdir(base):
            p = os.path.join(base, s, name)
            if os.path.exists(p):
                return p
    return None

# asset basename (without .txt) -> tokenizer.json path
MODELS = {
    "qwen3_5_tokenizer":     snap("Qwen--Qwen3.5-35B-A3B-GPTQ-Int4", "tokenizer.json"),
    "qwen3_omni_tokenizer":  "/tmp/qwen3_omni_dir/tokenizer.json",
    "gpt_oss_20b_tokenizer": snap("openai--gpt-oss-20b", "tokenizer.json"),
    "minicpmo4_5_tokenizer": snap("openbmb--MiniCPM-o-4_5", "tokenizer.json"),
    "minimaxm2_tokenizer":   snap("MiniMaxAI--MiniMax-M2.1", "tokenizer.json"),
    "glm5_tokenizer":        "/tmp/glm5_tokenizer.json",
    "gemma4_tokenizer":      snap("google--gemma-4-31B-it", "tokenizer.json"),
}

def unescape(s):
    out = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s) and s[i+1] in 'ntr\\':
            out.append({'n': '\n', 't': '\t', 'r': '\r', '\\': '\\'}[s[i+1]])
            i += 2
            continue
        out.append(s[i])
        i += 1
    return ''.join(out)

def main():
    corpus_path = os.path.join(HERE, "golden", "corpus.txt")
    corpus = [unescape(l.rstrip('\n')) for l in open(corpus_path, encoding='utf-8')]
    manifest = []
    for name, jpath in MODELS.items():
        if not jpath or not os.path.exists(jpath):
            print(f"SKIP {name}: tokenizer.json not found ({jpath})")
            continue
        tok = Tokenizer.from_file(jpath)
        out_path = os.path.join(HERE, "golden", name + ".golden")
        with open(out_path, "w") as f:
            for text in corpus:
                ids = tok.encode(text, add_special_tokens=False).ids
                f.write(" ".join(str(i) for i in ids) + "\n")
        manifest.append(name)
        print(f"OK   {name}: {len(corpus)} lines -> {os.path.relpath(out_path, HERE)}")
    with open(os.path.join(HERE, "golden", "manifest.txt"), "w") as f:
        for n in manifest:
            f.write(n + "\n")
    print(f"manifest: {len(manifest)} models")

if __name__ == "__main__":
    main()
