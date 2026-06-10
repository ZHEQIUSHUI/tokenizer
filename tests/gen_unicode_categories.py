#!/usr/bin/env python3
"""Generate src/tokenizer/unicode_categories.hpp from Python's unicodedata.

The BPE pre-tokenizer needs Unicode general-category predicates (\\p{L} \\p{M}
\\p{N} and the letter sub-categories Ll/Lu/Lt) to reproduce HuggingFace's
pre-tokenization regex exactly, without pulling in std::regex or ICU. We emit a
compact, binary-searchable range table.

Run from the repo root:  python3 tests/gen_unicode_categories.py
"""
import os
import unicodedata as ud

FL_L, FL_M, FL_N, FL_Ll, FL_Lu, FL_Lt = 1, 2, 4, 8, 16, 32

def fl(cp):
    c = ud.category(chr(cp))
    f = 0
    if c[0] == 'L': f |= FL_L
    if c[0] == 'M': f |= FL_M
    if c[0] == 'N': f |= FL_N
    if c == 'Ll': f |= FL_Ll
    elif c == 'Lu': f |= FL_Lu
    elif c == 'Lt': f |= FL_Lt
    return f

def build_ranges():
    ranges = []
    prev = 0
    start = 0
    for cp in range(0, 0x110000):
        f = fl(cp)
        if f != prev:
            if prev != 0:
                ranges.append((start, cp - 1, prev))
            start = cp
            prev = f
    if prev != 0:
        ranges.append((start, 0x10FFFF, prev))
    return ranges

def main():
    ranges = build_ranges()
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, '..', 'src', 'tokenizer', 'unicode_categories.hpp')
    out = []
    out.append("#pragma once")
    out.append("// AUTO-GENERATED from Python unicodedata (Unicode %s). Do not edit by hand." % ud.unidata_version)
    out.append("// Regenerate with tests/gen_unicode_categories.py")
    out.append("// Compact category ranges for the BPE pre-tokenizer (\\p{L}\\p{M}\\p{N} + Ll/Lu/Lt).")
    out.append("#include <cstdint>")
    out.append("#include <cstddef>")
    out.append("namespace MNN { namespace Transformer {")
    out.append("enum UCFlag { UC_L=1, UC_M=2, UC_N=4, UC_Ll=8, UC_Lu=16, UC_Lt=32 };")
    out.append("struct UCRange { uint32_t lo; uint32_t hi; uint8_t flags; };")
    out.append("static const UCRange kUCRanges[] = {")
    line = "  "
    for lo, hi, f in ranges:
        tok = "{0x%X,0x%X,%d}," % (lo, hi, f)
        if len(line) + len(tok) > 110:
            out.append(line)
            line = "  "
        line += tok
    out.append(line)
    out.append("};")
    out.append("static const size_t kUCRangeCount = sizeof(kUCRanges)/sizeof(kUCRanges[0]);")
    out.append("""
inline uint8_t uc_flags(uint32_t cp) {
    size_t lo = 0, hi = kUCRangeCount;
    while (lo < hi) {
        size_t mid = (lo + hi) >> 1;
        if (cp < kUCRanges[mid].lo) hi = mid;
        else if (cp > kUCRanges[mid].hi) lo = mid + 1;
        else return kUCRanges[mid].flags;
    }
    return 0;
}
inline bool uc_is_L(uint32_t cp){ return uc_flags(cp) & UC_L; }
inline bool uc_is_M(uint32_t cp){ return uc_flags(cp) & UC_M; }
inline bool uc_is_N(uint32_t cp){ return uc_flags(cp) & UC_N; }
inline bool uc_is_Ll(uint32_t cp){ return uc_flags(cp) & UC_Ll; }
inline bool uc_is_Lu(uint32_t cp){ return uc_flags(cp) & UC_Lu; }
inline bool uc_is_Lt(uint32_t cp){ return uc_flags(cp) & UC_Lt; }
// \\s : onig unicode whitespace
inline bool uc_is_space(uint32_t cp){
    switch(cp){
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x1C: case 0x1D: case 0x1E: case 0x1F: case 0x85: case 0xA0:
        case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F:
        case 0x3000: case 0xFEFF: return true;
        default: return (cp>=0x2000 && cp<=0x200A);
    }
}
} }
""")
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("Unicode %s, %d ranges -> %s" % (ud.unidata_version, len(ranges), os.path.normpath(out_path)))

if __name__ == '__main__':
    main()
