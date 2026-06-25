// Multimodal old-vs-new parity check.
//
// Compares, for one image turn, the hand-written apply_chat_template (old path,
// loaded from a converted .txt asset) against the general jinja path (new path,
// HFAutoTokenizer reading the HF dir) -- both the rendered prompt and the token
// ids. Confirms the new path's automatic pad-token expansion reproduces the old
// vision block exactly.
//
// Needs a VL model dir + its converted .txt asset, so it is a developer/manual
// test: with no args it skips (exit 0) and CI stays green.
//
//   usage: test_mm_compare <hf_model_dir> <old_asset.txt> <ModelType_name>
//   (ModelType_name is the enum name, e.g. "Qwen3VL" -- the string create_tokenizer
//    registers via REGISTER's #clsid, not the lower-case class alias.)
//   e.g.   test_mm_compare /path/Qwen3-VL-dir qwen3vl.txt Qwen3VL
#include "BaseTokenizer.hpp"
#include "HFAutoTokenizer.hpp"
#include <cstdio>
#include <string>
#include <vector>

static std::string sh(const std::string &s){
    std::string o; for(char c: s){ if(c=='\n') o+="\\n"; else o+=c; } return o;
}

int main(int argc, char** argv){
    if (argc < 4) {
        printf("[SKIP] test_mm_compare: needs <model_dir> <old_asset> <registered_name>\n");
        return 0;
    }
    auto oldt = create_tokenizer(std::string(argv[3]));
    if (!oldt || !oldt->load(argv[2])) { fprintf(stderr, "old load failed\n"); return 1; }

    HFAutoTokenizer nw;
    if (!nw.from_pretrained(argv[1])) { fprintf(stderr, "new load failed\n"); return 1; }

    // One image turn; explicit system isolates the multimodal block from any
    // model-default system message the official template may add.
    std::vector<Content> msgs = {
        {SYSTEM, TEXT,  "You are helpful.",        0, 0},
        {USER,   IMAGE, "What is in this image?",  1, 4},
    };

    const std::string os = oldt->apply_chat_template(msgs, true);
    const std::string ns = nw.apply_chat_template(msgs, true);
    const auto oi = oldt->encode(os);
    const auto ni = nw.encode(ns, /*add_special_tokens=*/false);

    const bool render_ok = (os == ns);
    const bool token_ok  = (oi == ni);
    printf("render old==new: %s\n", render_ok ? "YES" : "no");
    printf("tokens old==new: %s  (old %zu, new %zu)\n", token_ok ? "YES" : "no", oi.size(), ni.size());
    if (!render_ok) {
        printf("  old: %s\n  new: %s\n", sh(os).c_str(), sh(ns).c_str());
    }
    const bool ok = render_ok && token_ok;
    printf("%s multimodal old-vs-new parity\n", ok ? "[ OK ]" : "[FAIL]");
    return ok ? 0 : 1;
}
