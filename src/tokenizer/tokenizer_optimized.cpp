#include "tokenizer_optimized.hpp"
#include "unicode_categories.hpp"
#include <fstream>
#include <sstream>
#include <queue>
#include <functional>
#include <random>
#include <codecvt>
#include <set>
#include <climits>
#include <cctype>
#include <cstring>
namespace MNN {
namespace Transformer {

// base64
static const std::string base64_chars =
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz"
"0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

static inline size_t one_char_len(const char *src) {
    return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

static inline void replace_all(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

static inline bool parse_hex_byte_token(const std::string& token, unsigned char* value) {
    if (token.size() != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x' || token[5] != '>') {
        return false;
    }
    auto hex2val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    const int hi = hex2val(token[3]);
    const int lo = hex2val(token[4]);
    if (hi < 0 || lo < 0) {
        return false;
    }
    *value = (unsigned char)((hi << 4) | lo);
    return true;
}

static std::string base64_decode(const std::string& str) {
    int in_len = str.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && ( str[in_] != '=') && is_base64(str[in_])) {
        char_array_4[i++] = str[in_]; in_++;
        if (i ==4) {
            for (i = 0; i <4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]);
            }
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (i = 0; (i < 3); i++) {
                ret.push_back(char_array_3[i]);
            }
            i = 0;
        }
    }
    if (i) {
        for (j = i; j < 4; j++) {
            char_array_4[j] = 0;
        }
        for (j = 0; j < 4; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]);
        }
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        for (j = 0; (j < i - 1); j++) {
            ret.push_back(char_array_3[j]);
        }
    }
    return ret;
}

static std::string maybe_decode_b64_token(const std::string& line) {
    static const std::string prefix = "b64:";
    if (line.rfind(prefix, 0) != 0) {
        return line;
    }
    return base64_decode(line.substr(prefix.size()));
}

static bool parse_merge_line(const std::string& line, std::string* left, std::string* right) {
    static const std::string prefix = "b64:";
    if (line.rfind(prefix, 0) == 0) {
        const size_t sep = line.find(' ');
        if (sep == std::string::npos) {
            return false;
        }
        const std::string lhs = line.substr(0, sep);
        const std::string rhs = line.substr(sep + 1);
        if (rhs.rfind(prefix, 0) != 0) {
            return false;
        }
        *left = base64_decode(lhs.substr(prefix.size()));
        *right = base64_decode(rhs.substr(prefix.size()));
        return true;
    }

    const size_t sep = line.find(' ');
    if (sep == std::string::npos) {
        return false;
    }
    *left = line.substr(0, sep);
    *right = line.substr(sep + 1);
    return true;
}

static inline void to_lower_case(std::string& str) {
    for (auto &c : str) {
        if (c >= 'A' && c <= 'Z') {
            c = tolower(static_cast<unsigned char>(c));
        }
    }
}

Tokenizer* Tokenizer::createTokenizer(const std::string& filename) {
    Tokenizer* tokenizer = nullptr;
    // check file
    std::ifstream tok_file(filename);
    if (!tok_file.good()) {
        printf("Failed: can't load tokenzier from: %s.\n", filename.c_str());
        return tokenizer;
    }
    // check tokenizer info
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int magic_number, tokenizer_type;
    line_str >> magic_number;
    if (magic_number != MAGIC_NUMBER) {
        printf("Failed: magic number is wrong from: %s.\n", filename.c_str());
        return tokenizer;
    }
    line_str >> tokenizer_type;
    printf("tokenizer_type = %d\n", tokenizer_type);
    // create tokenizer
    switch (tokenizer_type)
    {
        case SENTENCEPIECE:
            tokenizer = new Sentencepiece();
            break;
        case TIKTOIKEN:
            tokenizer = new Tiktoken();
            break;
        case BERT:
            tokenizer = new BertTokenizer();
            break;
        case HUGGINGFACE:
            tokenizer = new HuggingfaceTokenizer();
            break;
        default:
            return tokenizer;
    }
    // load special tokens
    tokenizer->load_special(tok_file);
    // load vocabs
    tokenizer->load_vocab(tok_file);
    tok_file.close();
    return tokenizer;
}

bool Tokenizer::is_stop(int token) {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
}

bool Tokenizer::is_special(int token) {
    return std::find(special_tokens_.begin(), special_tokens_.end(), token) != special_tokens_.end();
}

std::vector<int> Tokenizer::get_stop_tokens() {
    return stop_tokens_;
}

void Tokenizer::load_special(std::ifstream& tok_file) {
    std::string line;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    int special_num, stop_num, prefix_num;
    line_str >> special_num >> stop_num >> prefix_num;
    std::getline(tok_file, line);
    std::istringstream specail_line(line);
    if (special_num) {
        // load special tokens
        special_tokens_.resize(special_num);
        for (int i = 0; i < special_num; i++) {
            specail_line >> special_tokens_[i];
        }
    }
    if (stop_num) {
        // load stop tokens
        stop_tokens_.resize(stop_num);
        for (int i = 0; i < stop_num; i++) {
            specail_line >> stop_tokens_[i];
        }
    }
    if (prefix_num) {
        // load prefix tokens
        prefix_tokens_.resize(prefix_num);
        for (int i = 0; i < prefix_num; i++) {
            specail_line >> prefix_tokens_[i];
        }
    }
    // Special tokens list updated; invalidate decoded-cache.
    special_cache_ready_ = false;
}

std::vector<int> Tokenizer::encode(const std::string& str) {
    std::vector<int> ids = prefix_tokens_;
    // Heuristic reserve to reduce re-allocations on long inputs.
    ids.reserve(ids.size() + str.size() / 2);

    if (special_tokens_.empty()) {
        encode(str, ids);
        return ids;
    }

    ensure_special_cache();
    const std::string& text = str;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        bool matched = false;
        for (const auto& p : special_cache_) {
            const int special_id = p.first;
            const std::string& tok = p.second;
            if (tok.empty()) continue;
            const size_t L = tok.size();
            if (i + L <= text.size() && text.compare(i, L, tok) == 0) {
                if (i > start) {
                    encode(text.substr(start, i - start), ids);
                }
                ids.push_back(special_id);
                start = i + L;
                i = start - 1;
                matched = true;
                break;
            }
        }
        if (matched) {
            continue;
        }
    }
    if (start < text.size()) {
        encode(text.substr(start), ids);
    }
    return ids;
}

void Tokenizer::ensure_special_cache() const {
    if (special_cache_ready_) {
        return;
    }
    special_cache_.clear();
    special_cache_.reserve(special_tokens_.size());
    for (int id : special_tokens_) {
        special_cache_.push_back({id, decode(id)});
    }
    special_cache_ready_ = true;
}

bool Sentencepiece::load_vocab(std::ifstream& tok_file) {
    std::string line, token;
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    float score;
    int type;
    sentence_pieces_.resize(vocab_len);
    for (int index = 0; index < vocab_len; index++) {
        std::getline(tok_file, line);
        std::istringstream line_str(line);
        line_str >> token >> score >> type;
        token = base64_decode(token);
        auto piece_type = static_cast<PieceType>(type);
        SentencePiece piece = {token, score, piece_type};
        sentence_pieces_[index] = std::move(piece);
        if (piece_type == PieceType::NORMAL) {
            pieces_.insert({token, index});
        } else {
            reserved_id_map_.insert({token, index});
            if (piece_type == PieceType::UNKNOWN) {
                unk_id_ = index;
            }
        }
    }
    return true;
}

int Sentencepiece::piece_to_id(const std::string& piece) const {
    auto it = reserved_id_map_.find(piece);
    if (it != reserved_id_map_.end()) {
        return it->second;
    }
    auto it2 = pieces_.find(piece);
    if (it2 != pieces_.end()) {
        return it2->second;
    }
    return unk_id_;
}

std::string Sentencepiece::byte_to_piece(unsigned char c) const {
    const int len = ::snprintf(nullptr, 0, "<0x%02X>", c);
    std::string s;
    s.resize(len);
    ::snprintf(&s[0], s.size() + 1, "<0x%02X>", c);
    return s;
}

// ref: https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
Sentencepiece::EncodeResult Sentencepiece::bpe_encode(string_view_ normalized, float alpha) {
    // util class begin
    struct SymbolPair {
        int left;     // left index of this pair
        int right;    // right index of this pair
        float score;  // score of this pair. large is better.
        size_t size;  // length of this piece
    };

    class SymbolPairComparator {
    public:
        const bool operator()(SymbolPair *h1, SymbolPair *h2) {
            return (h1->score < h2->score || (h1->score == h2->score && h1->left > h2->left));
        }
    };

    struct Symbol {
        int prev;     // prev index of this symbol. -1 for BOS.
        int next;     // next index of tihs symbol. -1 for EOS.
        bool freeze = false;  // this symbol is never be merged.
        string_view_ piece;
    };
    // util class end

    using Agenda = std::priority_queue<SymbolPair *, std::vector<SymbolPair *>, SymbolPairComparator>;
    Agenda agenda;
    std::vector<Symbol> symbols;
    symbols.reserve(normalized.size());
    // Reverse merge rules. key: merged symbol, value: pair of original symbols.
    std::unordered_map<string_view_, std::pair<string_view_, string_view_>> rev_merge;
    // SymbolPair holder.
    std::vector<std::unique_ptr<SymbolPair>> symbol_pair_holder;
    // Lookup new symbol pair at [left, right] and inserts it to agenda.
    auto MaybeAddNewSymbolPair = [this, &symbol_pair_holder, &symbols, &agenda, &rev_merge](int left, int right) {
        if (left == -1 || right == -1 || symbols[left].freeze || symbols[right].freeze) {
            return;
        }
        const string_view_ piece(symbols[left].piece.data(), symbols[left].piece.size() + symbols[right].piece.size());
        std::string piece_str(piece.to_string());
        const auto it = pieces_.find(piece_str);
        if (it == pieces_.end()) {
            return;
        }
        symbol_pair_holder.emplace_back(new SymbolPair);
        auto *h = symbol_pair_holder.back().get();
        h->left = left;
        h->right = right;
        h->score = get_score(it->second);
        h->size = piece.size();
        agenda.push(h);

        // Makes `rev_merge` for resegmentation.
        if (is_unused(it->second)) {
            rev_merge[piece] = std::make_pair(symbols[left].piece, symbols[right].piece);
        }
    };
    // Splits the input into character sequence
    int index = 0;
    while (!normalized.empty()) {
        Symbol s;
        // const int mblen = matcher_->PrefixMatch(normalized, &s.freeze);
        int mblen = std::min<int>(normalized.size(), one_char_len(normalized.data()));
        s.piece = string_view_(normalized.data(), mblen);
        s.prev = index == 0 ? -1 : index - 1;
        normalized.remove_prefix(mblen);
        s.next = normalized.empty() ? -1 : index + 1;
        ++index;
        symbols.emplace_back(s);
    }

    if (symbols.empty()) {
        return {};
    }
    // Lookup all bigrams.
    for (size_t i = 1; i < symbols.size(); ++i) {
        MaybeAddNewSymbolPair(i - 1, i);
    }

    // BPE-dropout: https://arxiv.org/pdf/1910.13267.pdf
    // std::mt19937 *rand_gen = nullptr;
    std::mt19937 rand_gen;
    auto skip_merge = [&]() {
        if (alpha <= 0.0) return false;
        if (alpha >= 1.0) return true;
        // if (rand_gen == nullptr) rand_gen = random::GetRandomGenerator();
        std::uniform_real_distribution<> gen(0.0, 1.0);
        return gen(rand_gen) < alpha;
    };

    // Main loop.
    while (!agenda.empty()) {
        SymbolPair *top = agenda.top();
        agenda.pop();

        // `top` is no longer available.
        if (symbols[top->left].piece.empty() || symbols[top->right].piece.empty() ||
            symbols[top->left].piece.size() + symbols[top->right].piece.size() != top->size) {
            continue;
        }

        if (skip_merge()) continue;
        // Replaces symbols with `top` rule.
        symbols[top->left].piece = string_view_(
                                                symbols[top->left].piece.data(),
                                                symbols[top->left].piece.size() + symbols[top->right].piece.size());

        // Updates prev/next pointers.
        symbols[top->left].next = symbols[top->right].next;
        if (symbols[top->right].next >= 0) {
            symbols[symbols[top->right].next].prev = top->left;
        }
        symbols[top->right].piece = string_view_("");

        // Adds new symbol pairs which are newly added after symbol replacement.
        MaybeAddNewSymbolPair(symbols[top->left].prev, top->left);
        MaybeAddNewSymbolPair(top->left, symbols[top->left].next);
    }

    std::function<void(string_view_, EncodeResult*)> resegment;
    resegment = [this, &resegment, &rev_merge](string_view_ w, EncodeResult *output) -> void {
        std::string w_str(w.to_string());
        const int id = piece_to_id(w_str);
        // std::cout << "piece: " << w << ", id = " << id << std::endl;
        if (id == -1 || !is_unused(id)) {
            output->emplace_back(w, id);
            return;
        }
        const auto p = rev_merge.find(w);
        if (p == rev_merge.end()) {
            // This block will never be called, as `rev_merge` stores all the
            // resegmentation info for unused id.
            output->emplace_back(w, id);
            return;
        }
        // Recursively resegment left and right symbols.
        resegment(p->second.first, output);
        resegment(p->second.second, output);
    };
    EncodeResult output;
    for (int index = 0; index != -1; index = symbols[index].next) {
        resegment(symbols[index].piece, &output);
    }
    return output;
}

void Sentencepiece::encode(const std::string& str, std::vector<int>& ids) {
    auto result = bpe_encode(str);
    size_t consumed = 0;
    for (const auto &p : result) {
        const string_view_ w = p.first;   // piece
        const int id = p.second;              // id
        const bool is_unk = (id == unk_id_);
        if (is_unk && byte_fall_back_) {
            // Decomposes an unknown piece into UTF-8 bytes
            for (int i = 0; i < w.size(); ++i) {
                // Create a byte piece
                const char b = w[i];
                const auto piece = byte_to_piece(b);
                auto sp_id = piece_to_id(piece);
                ids.push_back(sp_id);
            }
        } else {
            ids.push_back(id);
        }
    }
}

std::string Sentencepiece::decode(int id) const {
    const auto& sp = sentence_pieces_[id];
    if (sp.type == PieceType::BYTE && sp.piece.size() == 6 &&
        sp.piece[0] == '<' && sp.piece[1] == '0' && sp.piece[2] == 'x' && sp.piece[5] == '>') {
        // Convert <0xHH> to raw byte
        auto hex2val = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return 0;
        };
        char byte = (char)((hex2val(sp.piece[3]) << 4) | hex2val(sp.piece[4]));
        return std::string(1, byte);
    }
    // Replace every U+2581 ("▁", 3 UTF-8 bytes) with a space. The previous code
    // used piece.replace(pos, pos + 3, " ") which (a) only handled the first
    // occurrence and (b) passed pos+3 as the length argument, over-deleting when
    // the marker was not at offset 0 (e.g. "▁▁" indentation tokens). replace_all
    // matches the SpaceReplaceBPE decode path below.
    std::string piece = sp.piece;
    replace_all(piece, u8"▁", " ");
    return piece;
}

float Sentencepiece::get_score(int id) const {
    return sentence_pieces_[id].score;
}

bool Sentencepiece::is_unused(int id) const {
    return sentence_pieces_[id].type == PieceType::UNUSED;
}

bool Sentencepiece::is_control(int id) const {
    return sentence_pieces_[id].type == PieceType::CONTROL;
}

bool Tiktoken::load_vocab(std::ifstream& tok_file) {
    std::string line;
    std::getline(tok_file, line);
    int vocab_len = std::stoi(line);
    // load vocab
    decoder_.resize(vocab_len);
    max_token_len_ = 0;
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        auto token = base64_decode(line);
        encoder_.insert({token, i});
        decoder_[i] = token;
        if (token.size() > max_token_len_) {
            max_token_len_ = token.size();
        }
    }
    build_trie();
    return true;
}

int Tiktoken::trie_find_next(int node, unsigned char c) const {
    const auto& edges = trie_[node].next;
    for (const auto& e : edges) {
        if (e.c == c) return e.next;
    }
    return -1;
}

int Tiktoken::trie_add_next(int node, unsigned char c) {
    auto& edges = trie_[node].next;
    for (const auto& e : edges) {
        if (e.c == c) return e.next;
    }
    const int new_node = (int)trie_.size();
    edges.push_back(TrieEdge{c, new_node});
    trie_.push_back(TrieNode{});
    return new_node;
}

void Tiktoken::build_trie() {
    trie_.clear();
    trie_.reserve(decoder_.size() * 2);
    trie_.push_back(TrieNode{}); // root

    for (int id = 0; id < (int)decoder_.size(); ++id) {
        const std::string& tok = decoder_[id];
        if (tok.empty()) continue;
        int node = 0;
        for (unsigned char c : tok) {
            node = trie_add_next(node, c);
        }
        // First (lowest-id) wins for duplicate token strings, matching the
        // reference engine's encoder_.insert() semantics (insert keeps the first).
        // A plain assignment would let a later duplicate (e.g. an added-token row
        // rendering to the same bytes) shadow the real BPE token id.
        if (trie_[node].id == -1) {
            trie_[node].id = id;
        }
    }
}

void Tiktoken::encode(const std::string& str, std::vector<int>& ids) {
    if (str.empty()) return;
    if (trie_.empty()) {
        // Fallback: should not happen if load_vocab() was called.
        std::cerr << "Error: trie not built." << std::endl;
        return;
    }

    // Rough heuristic to reduce reallocations.
    ids.reserve(ids.size() + str.size() / 4);

    size_t i = 0;
    while (i < str.size()) {
        int node = 0;
        int last_id = -1;
        size_t last_len = 0;

        const size_t limit = std::min(str.size(), i + max_token_len_);
        for (size_t j = i; j < limit; ++j) {
            const unsigned char c = (unsigned char)str[j];
            const int nxt = trie_find_next(node, c);
            if (nxt < 0) break;
            node = nxt;
            if (trie_[node].id != -1) {
                last_id = trie_[node].id;
                last_len = j - i + 1;
            }
        }

        if (last_id == -1) {
            std::cerr << "Error: No encoding found for the sequence starting at position " << i << std::endl;
            return;
        }

        ids.push_back(last_id);
        i += last_len;
    }
}

std::string Tiktoken::decode(int id) const {
    if (id >= decoder_.size()) {
        return "";
    }
    return decoder_[id];
}

std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    auto it = encoder_.find(token);
    if (it != encoder_.end()) {
        return {it->second};
    }
    std::vector<int> ids;
    std::string current = token;
    while (!current.empty()) {
        int match_id = -1;
        size_t match_pos = 0;
        for (int len = current.size(); len > 0; --len) {
            std::string candidate = current.substr(0, len);
            if (!ids.empty()) {
                candidate = "##" + candidate;
            }
            auto it = encoder_.find(candidate);
            if (it != encoder_.end()) {
                match_id = it->second;
                match_pos = len;
                break;
            }
        }
        // [UNK]
        if (match_id == -1) {
            ids.push_back(100);
            break;
        }
        ids.push_back(match_id);
        // not first word, adding ## prefix
        current = current.substr(match_pos);
    }
    return ids;
}

void BertTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;
    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);
        // handle multi-byte UTF-8 characters
        if ((c & 0x80) != 0) {
            unsigned char mask = 0xE0; // 1110 0000 for 3-byte char
            if ((c & mask) == mask) {
                current_token = str.substr(i, 3);
                i += 3;
            } else {
                ++i;
                continue;
            }
        }
        // handle continuous sequence of letters and digits
        else if (isalnum(c)) {
            while (i < str.size() && isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += tolower(str[i]);
                ++i;
            }
        }
        // handle punctuation and symbols
        else if (ispunct(c)) {
            current_token = str[i];
            ++i;
        }
        // handle space, tab, enter
        else if (isspace(c)) {
            ++i;
            continue;
        }
        // handle any other single-byte characters
        else {
            current_token = str[i];
            ++i;
        }
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
    }

    for (auto token : tokens) {
        for (auto id : word_piece(token)) {
            ids.push_back(id);
        }
    }
}

std::wstring utf8_to_wstring(const std::string& str) {
    auto is_cont = [](unsigned char c) { return (c & 0xC0) == 0x80; };

    std::wstring out;
    out.reserve(str.size());

    for (size_t i = 0; i < str.size();) {
        const unsigned char c0 = static_cast<unsigned char>(str[i]);

        if (c0 < 0x80) {
            out.push_back((wchar_t)c0);
            ++i;
            continue;
        }

        uint32_t codepoint = 0;
        size_t advance = 0;

        if ((c0 & 0xE0) == 0xC0 && i + 1 < str.size()) {
            const unsigned char c1 = static_cast<unsigned char>(str[i + 1]);
            if (is_cont(c1)) {
                const uint32_t cp = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(c1 & 0x3F);
                if (cp >= 0x80) {
                    codepoint = cp;
                    advance = 2;
                }
            }
        } else if ((c0 & 0xF0) == 0xE0 && i + 2 < str.size()) {
            const unsigned char c1 = static_cast<unsigned char>(str[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(str[i + 2]);
            if (is_cont(c1) && is_cont(c2)) {
                const uint32_t cp = ((uint32_t)(c0 & 0x0F) << 12) |
                                    ((uint32_t)(c1 & 0x3F) << 6) |
                                    (uint32_t)(c2 & 0x3F);
                if (cp >= 0x800 && !(cp >= 0xD800 && cp <= 0xDFFF)) {
                    codepoint = cp;
                    advance = 3;
                }
            }
        } else if ((c0 & 0xF8) == 0xF0 && i + 3 < str.size()) {
            const unsigned char c1 = static_cast<unsigned char>(str[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(str[i + 2]);
            const unsigned char c3 = static_cast<unsigned char>(str[i + 3]);
            if (is_cont(c1) && is_cont(c2) && is_cont(c3)) {
                const uint32_t cp = ((uint32_t)(c0 & 0x07) << 18) |
                                    ((uint32_t)(c1 & 0x3F) << 12) |
                                    ((uint32_t)(c2 & 0x3F) << 6) |
                                    (uint32_t)(c3 & 0x3F);
                if (cp >= 0x10000 && cp <= 0x10FFFF) {
                    codepoint = cp;
                    advance = 4;
                }
            }
        }

        if (advance != 0) {
            // Windows has 16-bit wchar_t; codepoints > 0xFFFF must be emitted as a UTF-16
            // surrogate pair so that the round-trip through wstring_to_utf8 (which uses
            // codecvt_utf8 and expects surrogates) and BPE table lookups stay consistent.
            // Linux/macOS have 32-bit wchar_t; the else-branch is the common case.
            if (sizeof(wchar_t) >= 4 || codepoint <= 0xFFFF) {
                out.push_back((wchar_t)codepoint);
            } else {
                const uint32_t v = codepoint - 0x10000;
                out.push_back((wchar_t)(0xD800u | (v >> 10)));
                out.push_back((wchar_t)(0xDC00u | (v & 0x3FFu)));
            }
            i += advance;
            continue;
        }

        // Keep malformed bytes representable instead of throwing.
        out.push_back((wchar_t)c0);
        ++i;
    }

    return out;
}

std::string wstring_to_utf8(const std::wstring& str) {
    // Hand-rolled, independent of std::wstring_convert / std::codecvt_utf8 which are
    // deprecated in C++17 and removed in C++26. Handles 16-bit wchar_t (Windows) by
    // reassembling UTF-16 surrogate pairs back into a single codepoint before UTF-8
    // encoding. Also the inverse round-trip partner of utf8_to_wstring above.
    std::string out;
    out.reserve(str.size() * 2);
    for (size_t i = 0; i < str.size();) {
        uint32_t cp = (uint32_t)str[i];
        ++i;
        if (sizeof(wchar_t) == 2 && cp >= 0xD800u && cp <= 0xDBFFu && i < str.size()) {
            const uint32_t low = (uint32_t)str[i];
            if (low >= 0xDC00u && low <= 0xDFFFu) {
                cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                ++i;
            }
        }
        if (cp < 0x80u) {
            out.push_back((char)cp);
        } else if (cp < 0x800u) {
            out.push_back((char)(0xC0u | (cp >> 6)));
            out.push_back((char)(0x80u | (cp & 0x3Fu)));
        } else if (cp < 0x10000u) {
            // Unpaired surrogates also fall here; encode them as-is for round-trip stability.
            out.push_back((char)(0xE0u | (cp >> 12)));
            out.push_back((char)(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back((char)(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0x10FFFFu) {
            out.push_back((char)(0xF0u | (cp >> 18)));
            out.push_back((char)(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back((char)(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back((char)(0x80u | (cp & 0x3Fu)));
        }
        // codepoints > 0x10FFFF are invalid; drop silently.
    }
    return out;
}

// Given a token as a UTF8 string, encode each byte into an wchar_t
void byte_encode_token(const std::string& token,
                       const std::unordered_map<uint8_t, wchar_t>& b2u,
                       std::wstring* result) {
    result->resize(0);
    for (char c : token) {
        wchar_t wc = b2u.at(uint8_t(c));
        result->push_back(wc);
    }
}

bool HuggingfaceTokenizer::load_vocab(std::ifstream& tok_file) {
    std::string line;
    // get nums
    int vocab_len, merge_len;
    std::getline(tok_file, line);
    std::istringstream line_str(line);
    line_str >> vocab_len >> merge_len;
    // Optional 3rd field: pre-tokenizer family id (added for HF-faithful encoding).
    // Absent in legacy assets -> stays 0 (legacy hand-rolled splitter).
    line_str >> pretok_family_;
    // load vocab
    decoder_.resize(vocab_len);
    for (int i = 0; i < vocab_len; i++) {
        std::getline(tok_file, line);
        const std::string token = maybe_decode_b64_token(line);
        encoder_.insert({token, i});
        decoder_[i] = token;
    }
    // load merge_rule
    for (int i = 0; i < merge_len; i++) {
        std::getline(tok_file, line);
        std::string left;
        std::string right;
        if (!parse_merge_line(line, &left, &right)) {
            std::cerr << "Error: invalid merge line: " << line << std::endl;
            return false;
        }
        bpe_ranks_.insert({{utf8_to_wstring(left), utf8_to_wstring(right)}, i});
    }
    // Heuristic mode detection: look at which whitespace marker dominates in the vocab.
    //   SentencePiece-style ("▁" = U+2581 lower-one-eighth block) → SpaceReplaceBPE
    //   GPT-2-style          ("Ġ" = U+0120)                        → GPT2ByteBPE
    // Picking the wrong path silently produces wrong tokenization.
    size_t space_piece_count = 0;
    size_t gpt_piece_count = 0;
    for (const auto& token : decoder_) {
        if (token.rfind(u8"▁", 0) == 0) ++space_piece_count;
        if (token.rfind("Ġ", 0) == 0) ++gpt_piece_count;
    }
    const size_t total_piece_count = space_piece_count + gpt_piece_count;
    if (space_piece_count > 1024 && space_piece_count > gpt_piece_count * 8) {
        mode_ = Mode::SpaceReplaceBPE;
        printf("huggingface tokenizer mode = space_replace_bpe\n");
    } else if (gpt_piece_count > 1024 && gpt_piece_count > space_piece_count * 8) {
        mode_ = Mode::GPT2ByteBPE;
        printf("huggingface tokenizer mode = gpt2_byte_bpe\n");
    } else {
        mode_ = Mode::GPT2ByteBPE;
        printf("huggingface tokenizer mode = gpt2_byte_bpe (fallback; "
               "ambiguous vocab: space=%zu gpt=%zu)\n",
               space_piece_count, gpt_piece_count);
        if (total_piece_count < 64) {
            std::cerr << "WARNING: HuggingfaceTokenizer vocab has very few whitespace-prefix "
                         "pieces; mode detection is unreliable. Verify tokenization on a known "
                         "sample before trusting inference output."
                      << std::endl;
        }
    }
    // bytes_to_unicode (GPT-2 / HF BPE style)
    // Build a full 256-entry mapping for fast encoding.
    for (int i = 0; i < 256; ++i) {
        b2u_[(size_t)i] = 0;
    }
    auto _fill_range = [&](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u_[(size_t)(uint8_t)c] = (wchar_t)c;
        }
    };

    _fill_range(L'!', L'~');
    _fill_range(L'¡', L'¬');
    _fill_range(L'®', L'ÿ');

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u_[(size_t)b] == 0) {
            b2u_[(size_t)b] = (wchar_t)(256 + n);
            n++;
        }
    }
    u2b_.clear();
    u2b_.reserve(512);
    for (int b = 0; b < 256; ++b) {
        u2b_.insert({b2u_[(size_t)b], (uint8_t)b});
    }
    return true;
}

std::string HuggingfaceTokenizer::byte_to_piece(unsigned char c) const {
    const int len = ::snprintf(nullptr, 0, "<0x%02X>", c);
    std::string s;
    s.resize(len);
    ::snprintf(&s[0], s.size() + 1, "<0x%02X>", c);
    return s;
}

void get_pairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>* pairs) {
    pairs->clear();

    if (word.size() < 2) return;

    wchar_t previous = word[0];
    for (int i = 1; i < word.size(); i++) {
        pairs->push_back({std::wstring(1, previous), std::wstring(1, word[i])});
        previous = word[i];
    }
}

void HuggingfaceTokenizer::bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result) {
    std::set<int> merged;  // records indices in pairs that were merged.
    auto _left = [](int i, std::set<int>& merged) {
        for (int j = i - 1; j >= -1; j--) {
            if (merged.find(j) == merged.end()) return j;
        }
        return -1;
    };
    auto _right = [](int i, int cap, std::set<int>& merged) {
        for (int j = i + 1; j < cap; j++) {
            if (merged.find(j) == merged.end()) return j;
        }
        return cap;
    };

    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(token, &pairs);

    while (true) {
        int min_score = INT_MAX;
        int to_merge = -1;  // indices into pairs.

        for (int i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {  // pair i is not merged.
                auto iter = bpe_ranks.find(pairs[i]);
                int score = iter != bpe_ranks.end() ? iter->second : INT_MAX;
                if (score < min_score) {
                    min_score = score;
                    to_merge = i;
                }
            }
        }

        if (to_merge == -1) break;

        merged.insert(to_merge);
        std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

        int l = _left(to_merge, merged);
        if (l >= 0) pairs[l].second = merge_into;
        int r = _right(to_merge, pairs.size(), merged);
        if (r < pairs.size()) pairs[r].first = merge_into;
    }  // end while (true)

    if (merged.size() == pairs.size()) {
        result->push_back(token);

    } else {
        for (int i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {
                if (_left(i, merged) < 0) result->push_back(pairs[i].first);
                result->push_back(pairs[i].second);
            }
        }
    }
}

// ---- Unicode-aware BPE pre-tokenizer ---------------------------------------
// Reproduces the HuggingFace `pre_tokenizer` regex of each model family exactly,
// operating on decoded Unicode codepoints (no std::regex, no ICU). Validated to
// match tokenizers' pre_tokenize_str() span-for-span. Each emitted span is then
// byte-level mapped + BPE-merged by emit_token(), as HF does after splitting.
namespace {
struct PreTokParams { int digit_max; bool cased; bool marks_in_letter; bool slash_trail; };

// family ids must stay in sync with convert_tokenizer.py and the .hpp comment.
static PreTokParams pretok_params(int family) {
    switch (family) {
        case 1: return {1, false, true,  false}; // Qwen:    \p{N}     | [\p{L}\p{M}]+
        case 2: return {1, false, false, false}; // MiniCPM: \p{N}     | \p{L}+
        case 3: return {3, false, false, false}; // GLM:     \p{N}{1,3}| \p{L}+
        case 4: return {3, true,  false, true};  // o200k:   cased letter runs, \p{N}{1,3}, [\r\n/]*
        default:return {1, false, false, false};
    }
}

// Decode UTF-8 into codepoints with byte offsets. offs has size cps+1, offs.back()==str.size().
// Malformed bytes are kept as their raw value (length 1), matching utf8_to_wstring's leniency.
static void decode_utf8_cps(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
    const unsigned char* p = (const unsigned char*)s.data();
    const size_t n = s.size();
    size_t i = 0;
    auto cont = [](unsigned char c){ return (c & 0xC0) == 0x80; };
    while (i < n) {
        const unsigned char c0 = p[i];
        uint32_t cp; size_t adv;
        if (c0 < 0x80) { cp = c0; adv = 1; }
        else if ((c0 & 0xE0) == 0xC0 && i + 1 < n && cont(p[i+1])) {
            cp = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(p[i+1] & 0x3F); adv = 2;
            if (cp < 0x80) { cp = c0; adv = 1; }
        }
        else if ((c0 & 0xF0) == 0xE0 && i + 2 < n && cont(p[i+1]) && cont(p[i+2])) {
            cp = ((uint32_t)(c0 & 0x0F) << 12) | ((uint32_t)(p[i+1] & 0x3F) << 6) | (uint32_t)(p[i+2] & 0x3F); adv = 3;
            if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) { cp = c0; adv = 1; }
        }
        else if ((c0 & 0xF8) == 0xF0 && i + 3 < n && cont(p[i+1]) && cont(p[i+2]) && cont(p[i+3])) {
            cp = ((uint32_t)(c0 & 0x07) << 18) | ((uint32_t)(p[i+1] & 0x3F) << 12) |
                 ((uint32_t)(p[i+2] & 0x3F) << 6) | (uint32_t)(p[i+3] & 0x3F); adv = 4;
            if (cp < 0x10000 || cp > 0x10FFFF) { cp = c0; adv = 1; }
        }
        else { cp = c0; adv = 1; }
        offs.push_back(i);
        cps.push_back(cp);
        i += adv;
    }
    offs.push_back(n);
}

static void pretokenize_unicode(const std::string& str, int family,
                                std::vector<std::pair<size_t, size_t>>& spans) {
    const PreTokParams P = pretok_params(family);
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    decode_utf8_cps(str, cp, off);
    const size_t m = cp.size();

    auto is_letter = [&](uint32_t c){ return uc_is_L(c) || (P.marks_in_letter && uc_is_M(c)); };
    auto is_other  = [&](uint32_t c){
        if (uc_is_space(c) || uc_is_L(c) || uc_is_N(c)) return false;
        if (P.marks_in_letter && uc_is_M(c)) return false;
        return true;
    };
    auto clsA = [&](uint32_t c){ return (uc_is_L(c) && !uc_is_Ll(c)) || uc_is_M(c); };          // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]
    auto clsB = [&](uint32_t c){ return (uc_is_L(c) && !uc_is_Lu(c) && !uc_is_Lt(c)) || uc_is_M(c); }; // [\p{Ll}\p{Lm}\p{Lo}\p{M}]
    auto lower = [](uint32_t c){ return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    auto match_contr = [&](size_t i) -> size_t {
        if (i + 1 >= m || cp[i] != '\'') return 0;
        const uint32_t c1 = lower(cp[i+1]);
        if (c1=='s' || c1=='t' || c1=='m' || c1=='d') return 2;
        if (i + 2 < m) {
            const uint32_t c2 = lower(cp[i+2]);
            if ((c1=='r'&&c2=='e') || (c1=='v'&&c2=='e') || (c1=='l'&&c2=='l')) return 3;
        }
        return 0;
    };
    auto emit = [&](size_t a, size_t b){ spans.emplace_back(off[a], off[b]); };

    size_t i = 0;
    while (i < m) {
        const uint32_t c = cp[i];
        // 1) contraction (simple families match it as a leading alternative)
        if (!P.cased) {
            const size_t mc = match_contr(i);
            if (mc) { emit(i, i + mc); i += mc; continue; }
        }
        // 2) letter run
        if (P.cased) {
            size_t j = i; bool lead = false;
            if (c != '\r' && c != '\n' && !uc_is_L(c) && !uc_is_N(c) &&
                i + 1 < m && (uc_is_L(cp[i+1]) || uc_is_M(cp[i+1]))) { j = i + 1; lead = true; }
            if (lead || uc_is_L(c) || uc_is_M(c)) {
                size_t end;
                size_t k1 = j; while (k1 < m && clsA(cp[k1])) ++k1;
                size_t b = k1; while (b < m && clsB(cp[b])) ++b;
                if (b > k1) { end = b; }
                else {
                    size_t k2 = j; while (k2 < m && clsA(cp[k2])) ++k2;
                    if (k2 > j) { size_t bb = k2; while (bb < m && clsB(cp[bb])) ++bb; end = bb; }
                    else end = j;
                }
                if (end > j) {
                    end += match_contr(end);
                    emit(i, end); i = end; continue;
                }
            }
        } else {
            size_t j = i; bool lead = false;
            if (c != '\r' && c != '\n' && !uc_is_L(c) && !uc_is_N(c) &&
                i + 1 < m && is_letter(cp[i+1])) { j = i + 1; lead = true; }
            if (lead || is_letter(c)) {
                size_t k = j; while (k < m && is_letter(cp[k])) ++k;
                if (k > j) { emit(i, k); i = k; continue; }
            }
        }
        // 3) number: \p{N}{1,digit_max}
        if (uc_is_N(c)) {
            size_t k = i; int cnt = 0;
            while (k < m && uc_is_N(cp[k]) && cnt < P.digit_max) { ++k; ++cnt; }
            emit(i, k); i = k; continue;
        }
        // 4) punctuation:  ?[^\s\p{L}(\p{M})\p{N}]+ [\r\n(/)]*
        {
            size_t j = i;
            if (c == ' ' && i + 1 < m && is_other(cp[i+1])) j = i + 1;
            if (j < m && is_other(cp[j])) {
                size_t k = j; while (k < m && is_other(cp[k])) ++k;
                while (k < m && (cp[k]=='\r' || cp[k]=='\n' || (P.slash_trail && cp[k]=='/'))) ++k;
                emit(i, k); i = k; continue;
            }
        }
        // 5/6/7) whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
        if (uc_is_space(c)) {
            size_t e = i; while (e < m && uc_is_space(cp[e])) ++e;
            long last_nl = -1;
            for (size_t k = i; k < e; ++k) if (cp[k]=='\n' || cp[k]=='\r') last_nl = (long)k;
            size_t seg;
            if (last_nl >= 0) seg = (size_t)last_nl + 1 - i;          // up to & incl last newline
            else if (e < m)  seg = (e - i > 1) ? (e - i - 1) : 1;     // followed by non-space: all but last
            else             seg = e - i;                            // EOF: whole run
            emit(i, i + seg); i += seg; continue;
        }
        // fallback: single codepoint
        emit(i, i + 1); ++i;
    }
}
} // anonymous namespace

void HuggingfaceTokenizer::encode(const std::string& str, std::vector<int>& ids) {
    if (mode_ == Mode::SpaceReplaceBPE) {
        std::string normalized;
        normalized.reserve(str.size() * 2);
        for (char c : str) {
            if (c == ' ') normalized += u8"▁";
            else normalized.push_back(c);
        }

        std::vector<std::wstring> bpe_tokens;
        bpe_tokens.reserve(normalized.size());
        bpe(utf8_to_wstring(normalized), bpe_ranks_, &bpe_tokens);
        for (const auto& ws : bpe_tokens) {
            const std::string token = wstring_to_utf8(ws);
            auto iter = encoder_.find(token);
            if (iter != encoder_.end()) {
                ids.push_back(iter->second);
                continue;
            }
            for (unsigned char b : token) {
                const std::string piece = byte_to_piece(b);
                auto byte_iter = encoder_.find(piece);
                if (byte_iter == encoder_.end()) {
                    std::cerr << "Error: byte fallback token not in vocab: " << piece << std::endl;
                    return;
                }
                ids.push_back(byte_iter->second);
            }
        }
        return;
    }

    // Regex-free GPT-2 byte-level BPE pre-tokenization.
    //
    // We intentionally avoid `std::regex` here:
    // - some libstdc++ builds have been observed to segfault on long inputs due to deep recursion/stack usage
    // - regex performance is unpredictable across platforms
    //
    // This implements the same simplified pattern previously used:
    //   ('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)
    // over raw bytes (UTF-8 is treated as bytes, consistent with byte-level BPE).
    const auto is_ascii_alpha = [](unsigned char c) -> bool {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    };
    const auto is_ascii_digit = [](unsigned char c) -> bool { return (c >= '0' && c <= '9'); };
    const auto is_ascii_space = [](unsigned char c) -> bool {
        switch (c) {
            case ' ': case '\t': case '\n': case '\r': case '\f': case '\v': return true;
            default: return false;
        }
    };
    const auto is_ascii_word = [&](unsigned char c) -> bool {
        return is_ascii_alpha(c) || is_ascii_digit(c) || c == '_';
    };

    // Heuristic reserve.
    ids.reserve(ids.size() + str.size() / 4);

    const auto emit_token = [&](const char *begin, const char *end) {
        if (!begin || !end || begin >= end) return;
        std::wstring wtoken;
        wtoken.reserve((size_t)(end - begin));
        for (const unsigned char *p = (const unsigned char *)begin; p < (const unsigned char *)end; ++p) {
            wtoken.push_back(b2u_[(size_t)*p]);
        }

        std::vector<std::wstring> bpe_tokens;
        bpe_tokens.reserve(wtoken.size());
        bpe(wtoken, bpe_ranks_, &bpe_tokens);
        for (const auto &ws : bpe_tokens) {
            const std::string s = wstring_to_utf8(ws);
            auto iter = encoder_.find(s);
            if (iter == encoder_.end()) {
                std::cerr << "Error: token not in vocab: " << s << std::endl;
                return;
            }
            ids.push_back(iter->second);
        }
    };

    // Unicode-aware pre-tokenization for known model families (matches HF exactly).
    // Falls back to the legacy ASCII splitter below when the asset carries no family.
    if (pretok_family_ != 0) {
        std::vector<std::pair<size_t, size_t>> spans;
        pretokenize_unicode(str, pretok_family_, spans);
        for (const auto& sp : spans) {
            emit_token(str.data() + sp.first, str.data() + sp.second);
        }
        return;
    }

    const char *data = str.data();
    const size_t n = str.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char c = (unsigned char)data[i];

        // Contractions: 's 't 're 've 'm 'll 'd
        if (c == '\'' && i + 1 < n) {
            const char *p = data + i;
            const size_t remain = n - i;
            auto match = [&](const char *lit) -> size_t {
                const size_t len = ::strlen(lit);
                if (remain >= len && ::memcmp(p, lit, len) == 0) return len;
                return 0;
            };
            size_t clen = 0;
            if ((clen = match("'s")) || (clen = match("'t")) || (clen = match("'re")) || (clen = match("'ve")) ||
                (clen = match("'m")) || (clen = match("'ll")) || (clen = match("'d"))) {
                emit_token(p, p + clen);
                i += clen;
                continue;
            }
        }

        // Optional leading space for alpha/digit/punct runs.
        if (c == ' ' && i + 1 < n) {
            const unsigned char next = (unsigned char)data[i + 1];
            if (is_ascii_alpha(next)) {
                size_t j = i + 2;
                while (j < n && is_ascii_alpha((unsigned char)data[j])) ++j;
                emit_token(data + i, data + j);
                i = j;
                continue;
            }
            if (is_ascii_digit(next)) {
                size_t j = i + 2;
                while (j < n && is_ascii_digit((unsigned char)data[j])) ++j;
                emit_token(data + i, data + j);
                i = j;
                continue;
            }
            if (!is_ascii_space(next) && !is_ascii_word(next)) {
                size_t j = i + 2;
                while (j < n) {
                    const unsigned char t = (unsigned char)data[j];
                    if (is_ascii_space(t) || is_ascii_word(t)) break;
                    ++j;
                }
                emit_token(data + i, data + j);
                i = j;
                continue;
            }
        }

        // Whitespace run.
        if (is_ascii_space(c)) {
            size_t j = i + 1;
            while (j < n && is_ascii_space((unsigned char)data[j])) ++j;
            emit_token(data + i, data + j);
            i = j;
            continue;
        }

        // Alpha run.
        if (is_ascii_alpha(c)) {
            size_t j = i + 1;
            while (j < n && is_ascii_alpha((unsigned char)data[j])) ++j;
            emit_token(data + i, data + j);
            i = j;
            continue;
        }

        // Digit run.
        if (is_ascii_digit(c)) {
            size_t j = i + 1;
            while (j < n && is_ascii_digit((unsigned char)data[j])) ++j;
            emit_token(data + i, data + j);
            i = j;
            continue;
        }

        // Punct / other non-space non-word run.
        {
            size_t j = i + 1;
            while (j < n) {
                const unsigned char t = (unsigned char)data[j];
                if (is_ascii_space(t) || is_ascii_word(t)) break;
                ++j;
            }
            emit_token(data + i, data + j);
            i = j;
            continue;
        }
    }
}

std::string HuggingfaceTokenizer::decode(int id) const {
    if (id >= decoder_.size()) {
        return "";
    }
    if (mode_ == Mode::SpaceReplaceBPE) {
        const std::string& token = decoder_.at(id);
        unsigned char byte_value = 0;
        if (parse_hex_byte_token(token, &byte_value)) {
            return std::string(1, (char)byte_value);
        }
        std::string out = token;
        replace_all(out, u8"▁", " ");
        return out;
    }
    std::wstring w = utf8_to_wstring(decoder_.at(id));
    std::string r;
    for (wchar_t c : w) {
        if (u2b_.find(c) != u2b_.end()) {
            r.push_back(char(u2b_.at(c)));
        }
    }
    return r;
}
}
}
