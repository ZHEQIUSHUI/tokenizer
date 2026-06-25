// Jinja chat-template parity test.
//
// Renders each model's own HuggingFace chat_template through JinjaChatTemplate
// (minja) on a fixed conversation and checks the output matches the golden render
// captured from HF's apply_chat_template (tests/golden/chat_templates.json, made by
// tests/gen_jinja_golden.py). No transformers / network needed at test time.
//
//   usage: test_jinja_parity [tests_dir]   (default "tests")
#include "JinjaChatTemplate.hpp"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using json = JinjaChatTemplate::json;

static std::string show(const std::string& s){
    std::string o; for(char c: s){ if(c=='\n') o+="\\n"; else o+=c; } return o;
}
static RoleType role_of(const std::string& r){
    if (r=="system") return SYSTEM;
    if (r=="assistant") return ASSISTANT;
    return USER;
}

int main(int argc, char** argv){
    // Templates that embed the current date (via strftime_now) render through
    // minja's std::localtime. Pin the zone so the fixed epoch-0 clock used in
    // JinjaChatTemplate::apply yields a stable "1970-01-01" matching the golden.
    // setenv is POSIX-only; MSVC/MinGW use _putenv_s / _tzset.
#ifdef _WIN32
    _putenv_s("TZ", "UTC");
    _tzset();
#else
    setenv("TZ", "UTC", 1);
    tzset();
#endif
    const std::string base = (argc > 1) ? argv[1] : "tests";
    std::ifstream f(base + "/golden/chat_templates.json");
    if (!f) { fprintf(stderr, "cannot read %s/golden/chat_templates.json\n", base.c_str()); return 2; }
    std::stringstream ss; ss << f.rdbuf();
    json data = json::parse(ss.str());

    std::vector<Content> contents;
    for (auto& m : data["messages"])
        contents.push_back({role_of(m["role"]), TEXT, m["content"].get<std::string>(), 0, 0});

    int pass = 0, total = 0;
    for (auto& m : data["models"]) {
        ++total;
        const std::string name = m["name"];
        JinjaChatTemplate tmpl;
        if (!tmpl.load(m["chat_template"].get<std::string>(),
                       m["bos"].get<std::string>(), m["eos"].get<std::string>())) {
            printf("[FAIL] %-12s template failed to parse\n", name.c_str());
            continue;
        }
        const std::string got = tmpl.apply(contents, /*add_generation_prompt=*/true);
        const std::string exp = m["hf_render"];
        if (got == exp) { printf("[ OK ] %-12s render == HF (%zu chars)\n", name.c_str(), got.size()); ++pass; }
        else {
            printf("[FAIL] %-12s render != HF\n", name.c_str());
            printf("   HF : %s\n", show(exp).c_str());
            printf("   got: %s\n", show(got).c_str());
        }
    }
    printf("\n%d/%d chat templates render identically to HuggingFace\n", pass, total);
    return pass == total ? 0 : 1;
}
