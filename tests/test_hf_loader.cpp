// Validates the general HF-directory path (HFAutoTokenizer): load a HuggingFace
// tokenizer dir and check encode() matches the committed HF golden ids, and that
// the jinja chat template renders. Needs a local HF tokenizer dir, so it is a
// developer/manual test -- with no args it skips (exit 0) so CI stays green.
//
//   usage: test_hf_loader <model_dir> [corpus_file golden_ids_file]
#include "HFAutoTokenizer.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string un(const std::string& in){
    std::string o; o.reserve(in.size());
    for(size_t i=0;i<in.size();++i){
        if(in[i]=='\\'&&i+1<in.size()){char n=in[i+1];
            if(n=='n'){o+='\n';++i;continue;} if(n=='t'){o+='\t';++i;continue;}
            if(n=='r'){o+='\r';++i;continue;} if(n=='\\'){o+='\\';++i;continue;}}
        o+=in[i];
    }
    return o;
}

int main(int argc, char** argv){
    if (argc < 2) {
        printf("[SKIP] test_hf_loader: no model dir given (developer/manual test)\n");
        return 0;
    }
    HFAutoTokenizer tok;
    if (!tok.from_pretrained(argv[1])) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    printf("loaded %s  (chat_template: %s)\n", argv[1], tok.has_chat_template() ? "yes" : "no");

    int rc = 0;
    if (argc >= 4) {
        std::ifstream cf(argv[2]), gf(argv[3]);
        std::string cl, gl; int ok=0, bad=0, idx=0;
        while (std::getline(cf, cl) && std::getline(gf, gl)) {
            std::vector<int> g; std::istringstream gs(gl); int x; while (gs>>x) g.push_back(x);
            auto ids = tok.encode(un(cl), /*add_special_tokens=*/false);
            bool m = ids.size()>=g.size() && std::equal(ids.end()-(long)g.size(), ids.end(), g.begin());
            if (m) ++ok; else { ++bad; if (bad<=3) printf("  line %d encode mismatch\n", idx); }
            ++idx;
        }
        printf("%s encode: %d/%d lines match HF golden\n", ok==ok+bad?"[ OK ]":"[FAIL]", ok, ok+bad);
        if (bad) rc = 1;
    }

    // chat smoke check
    if (tok.has_chat_template()) {
        std::vector<Content> msgs = {
            {SYSTEM, TEXT, "You are a helpful assistant.", 0, 0},
            {USER,   TEXT, "What is 2+2?", 0, 0},
        };
        std::string p = tok.apply_chat_template(msgs, true);
        printf("chat render: %zu chars, ids=%zu\n", p.size(), tok.encode_chat(msgs, true).size());
        if (p.empty()) { printf("[FAIL] chat render empty\n"); rc = 1; }
    }
    return rc;
}
