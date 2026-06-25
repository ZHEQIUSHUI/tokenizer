#!/usr/bin/env python3
"""Generate the golden chat-template renders for tests/test_jinja_parity.cpp.

For each model we record its HuggingFace `chat_template`, bos/eos tokens, and the
string that `apply_chat_template` produces for a fixed conversation. The committed
golden lets the C++ test verify minja renders identically to HuggingFace without
needing transformers / network at test time.

Developer tool: needs the HF tokenizers locally (cache snapshots, or dirs made by
`AutoTokenizer.from_pretrained(...).save_pretrained(...)`). Edit MODELS as needed.

Run from the repo root:  python3 tests/gen_jinja_golden.py
"""
import json
import os
import glob
import datetime

# Pin the clock so templates that embed the current date (e.g. gpt-oss) render
# deterministically. We freeze to 1970-01-01 UTC to match the epoch-0 clock used
# by JinjaChatTemplate::apply in the C++ test (run there with TZ=UTC).
os.environ["TZ"] = "UTC"
try:
    import time
    time.tzset()
except Exception:
    pass

from transformers import AutoTokenizer

def _freeze_dates():
    # transformers' chat-template `strftime_now` is a nested function that calls
    # `datetime.now()` resolved against this module's `datetime` symbol (it does
    # `from datetime import datetime`). Swap that symbol for a frozen stand-in.
    import transformers.utils.chat_template_utils as ctu
    class _FrozenDatetime:
        @staticmethod
        def now(*args, **kwargs):
            return datetime.datetime(1970, 1, 1)
    ctu.datetime = _FrozenDatetime

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/huggingface/hub")

def snap(repo):
    d = glob.glob(f"{HUB}/models--{repo}/snapshots/*/")
    return d[0] if d else None

# name -> local model dir (tokenizer files)
MODELS = {
    "qwen3_5": snap("Qwen--Qwen3.5-35B-A3B-GPTQ-Int4"),
    "minimax": snap("MiniMaxAI--MiniMax-M2.1"),
    "minicpm": snap("openbmb--MiniCPM-o-4_5"),
    "gpt_oss": snap("openai--gpt-oss-20b"),
}

# Fixed text-only conversation exercising system + multi-turn + generation prompt.
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "Thanks! And 3+3?"},
]

def main():
    _freeze_dates()
    out = []
    for name, d in MODELS.items():
        if not d or not os.path.isdir(d):
            print(f"SKIP {name}: no dir")
            continue
        try:
            tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
        except Exception as e:
            print(f"SKIP {name}: load failed {str(e)[:80]}")
            continue
        if tok.chat_template is None:
            print(f"SKIP {name}: no chat_template")
            continue
        try:
            rendered = tok.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            print(f"SKIP {name}: apply failed {str(e)[:80]}")
            continue
        out.append({
            "name": name,
            "chat_template": tok.chat_template,
            "bos": tok.bos_token or "",
            "eos": tok.eos_token or "",
            "hf_render": rendered,
        })
        print(f"OK {name}: {len(rendered)} chars")

    dst = os.path.join(HERE, "golden", "chat_templates.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"messages": MESSAGES, "models": out}, f, ensure_ascii=False, indent=0)
    print(f"wrote {os.path.relpath(dst, HERE)} with {len(out)} models")

if __name__ == "__main__":
    main()
