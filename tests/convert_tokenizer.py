import os
import json
import glob
import base64
import argparse

from transformers import AutoTokenizer


class LlmExporter():
    '''
    Base class for all llm model export. Inherits from [`torch.nn.Module`].
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stop_ids = []
        self.is_fastvlm = 'fastvlm' in self.args.tokenizer_path.lower()
        self.load_pretrained()

    def _encode_no_special_tokens(self, text):
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except:
            return self.tokenizer.encode(text)

    def _resolve_local_file(self, filename):
        candidates = []

        # explicit local path
        candidates.append(os.path.join(self.args.tokenizer_path, filename))

        # tokenizer runtime path
        if hasattr(self.tokenizer, 'name_or_path') and isinstance(self.tokenizer.name_or_path, str):
            candidates.append(os.path.join(self.tokenizer.name_or_path, filename))

        # tokenizer init kwargs may keep resolved local cache paths
        init_kwargs = getattr(self.tokenizer, 'init_kwargs', None)
        if isinstance(init_kwargs, dict):
            name_or_path = init_kwargs.get('name_or_path', None)
            if isinstance(name_or_path, str):
                candidates.append(os.path.join(name_or_path, filename))

            vocab_file = init_kwargs.get('vocab_file', None)
            if isinstance(vocab_file, str):
                candidates.append(os.path.join(os.path.dirname(vocab_file), filename))

            merges_file = init_kwargs.get('merges_file', None)
            if isinstance(merges_file, str) and filename == 'merges.txt':
                candidates.append(merges_file)

        for path in candidates:
            if isinstance(path, str) and os.path.exists(path):
                return path
        return None

    def _load_tokenizer_json_bpe(self):
        tokenizer_json = self._resolve_local_file('tokenizer.json')
        if tokenizer_json is None:
            return None

        with open(tokenizer_json, 'r', encoding='utf-8') as f:
            config = json.load(f)

        model = config.get('model', {})
        if model.get('type', '').upper() != 'BPE':
            return None

        vocab = model.get('vocab', None)
        merges = model.get('merges', None)
        if not isinstance(vocab, dict) or not merges:
            return None

        max_id = -1
        for idx in vocab.values():
            max_id = max(max_id, int(idx))
        for item in config.get('added_tokens', []):
            if isinstance(item, dict) and 'id' in item:
                max_id = max(max_id, int(item['id']))

        if max_id < 0:
            raise RuntimeError(
                f'{tokenizer_json}: model.vocab is empty, cannot export HF BPE tokenizer')

        vocab_list = ['<unk>' for _ in range(max_id + 1)]
        filled = [False] * (max_id + 1)
        for token, idx in vocab.items():
            vocab_list[int(idx)] = token
            filled[int(idx)] = True
        for item in config.get('added_tokens', []):
            if not isinstance(item, dict) or 'id' not in item or 'content' not in item:
                continue
            idx = int(item['id'])
            if idx >= len(vocab_list):
                vocab_list.extend(['<unk>' for _ in range(idx + 1 - len(vocab_list))])
                filled.extend([False] * (idx + 1 - len(filled)))
            vocab_list[idx] = item['content']
            filled[idx] = True

        gaps = sum(1 for f in filled if not f)
        if gaps > 0:
            # An unfilled id means some vocab entry is missing. Decoder on the device
            # will return the literal string "<unk>" for these ids, which looks like
            # correct output but is actually a placeholder bug. Surface it loudly.
            print(f'[convert_tokenizer] WARNING: {gaps} / {len(vocab_list)} vocab ids '
                  f'have no entry in tokenizer.json and will decode as "<unk>".')

        merge_list = []
        for merge in merges:
            if isinstance(merge, (list, tuple)) and len(merge) == 2:
                left, right = str(merge[0]), str(merge[1])
            elif isinstance(merge, str):
                parts = merge.split(' ', 1)
                if len(parts) != 2:
                    raise RuntimeError(f'invalid BPE merge rule in {tokenizer_json}: {merge!r}')
                left, right = parts[0], parts[1]
            else:
                raise RuntimeError(f'invalid BPE merge rule in {tokenizer_json}: {merge!r}')
            if not left or not right:
                raise RuntimeError(f'empty side in BPE merge rule in {tokenizer_json}: {merge!r}')
            merge_list.append((left, right))

        if not merge_list:
            raise RuntimeError(
                f'{tokenizer_json}: BPE model has vocab but no merges; refusing to export')

        return vocab_list, merge_list

    def _pretok_family(self):
        """Detect the BPE pre-tokenizer family so the device engine can reproduce the
        model's HuggingFace pre_tokenizer regex exactly (see tokenizer_optimized.cpp).
          0 = legacy/unknown (engine falls back to its ASCII splitter)
          1 = Qwen    ( [\\p{L}\\p{M}]+ , single \\p{N} )
          2 = MiniCPM ( \\p{L}+ , single \\p{N} )
          3 = GLM     ( \\p{L}+ , \\p{N}{1,3} )
          4 = o200k   ( cased letter runs, \\p{N}{1,3}, [\\r\\n/]* )  -- gpt-oss / MiniMax
        """
        tokenizer_json = self._resolve_local_file('tokenizer.json')
        if tokenizer_json is None:
            return 0
        try:
            with open(tokenizer_json, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            return 0
        regexes = []
        def walk(p):
            if not isinstance(p, dict):
                return
            if p.get('type') == 'Split' and isinstance(p.get('pattern'), dict):
                r = p['pattern'].get('Regex')
                if r:
                    regexes.append(r)
            for sub in p.get('pretokenizers', []) or []:
                walk(sub)
        walk(cfg.get('pre_tokenizer') or {})
        if not regexes:
            return 0
        rx = '\n'.join(regexes)
        cased = ('\\p{Lu}' in rx) or ('\\p{Lt}' in rx)
        marks = '\\p{L}\\p{M}' in rx
        digit3 = '\\p{N}{1,3}' in rx
        if cased:
            family = 4
        elif marks:
            family = 1
        elif digit3:
            family = 3
        else:
            family = 2
        print(f'[convert_tokenizer] pre_tokenizer family = {family}')
        return family

    def _ensure_fastvlm_image_single_token(self):
        # FastVLM vision injection expects "<image>" to map to a single token id.
        # Keep this logic isolated to FastVLM to avoid changing behavior of other models.
        if not self.is_fastvlm:
            return

        image_token = "<image>"
        before = self._encode_no_special_tokens(image_token)
        if len(before) == 1:
            return

        vocab_before = len(self.tokenizer.get_vocab())
        added = self.tokenizer.add_tokens([image_token], special_tokens=True)
        after = self._encode_no_special_tokens(image_token)

        if len(after) != 1:
            raise RuntimeError(f"FastVLM patch failed: {image_token} is not single token, ids={after}")

        vocab_after = len(self.tokenizer.get_vocab())
        print(f"[FastVLM] patched {image_token}: ids {before} -> {after}, add_tokens={added}, vocab {vocab_before}->{vocab_after}")

    def load_pretrained(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, trust_remote_code=True, use_fast=False)
        except:
            self.tokenizer = None
        if None == self.tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, trust_remote_code=True)
            except:
                self.tokenizer = None
        if None == self.tokenizer:
            # Workaround for some newer tokenizer configs (e.g. google/gemma-4-31B-it) where
            # `extra_special_tokens` may not match Transformers' expected schema.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path, trust_remote_code=True, extra_special_tokens={})
            except:
                self.tokenizer = None
        if None == self.tokenizer:
            print("Default load tokenizer failed for ", self.args.tokenizer_path)
            return

        self._ensure_fastvlm_image_single_token()
        
        generation_config_file = self._resolve_local_file('generation_config.json')
        if generation_config_file is not None:
            self.generation_config = json.load(open(generation_config_file, 'r'))
        else:
            self.generation_config = None
        
        
        self.stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'im_end_id'):
            self.stop_ids.append(self.tokenizer.im_end_id)
        try:
            eot_id = self.tokenizer.encode('<|eot_id|>')
            if len(eot_id) == 1:
                self.stop_ids.append(eot_id[0])
            # gemma/gemma-2
            eot_id = self.tokenizer.encode('<end_of_turn>')
            if len(eot_id) == 2 and eot_id[0] == 2:
                self.stop_ids.append(eot_id[1])
        except:
            pass
        # if hasattr(self.model, 'generation_config') and self.model.generation_config is not None:
        if self.generation_config is not None:
            eos_token_id = self.generation_config.get('eos_token_id', None)
            from collections.abc import Iterable
            if isinstance(eos_token_id, int):
                self.stop_ids.append(eos_token_id)
            elif isinstance(eos_token_id, Iterable):
                for id in eos_token_id:
                    self.stop_ids.append(id)
                
        self.stop_ids = [stop_id for stop_id in self.stop_ids if stop_id is not None]
        self.stop_ids = list(set(self.stop_ids))
        

    def export_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError(f"tokenizer is not loaded: {self.args.tokenizer_path}")

        # load tokenizer file
        tokenizer_model = self._resolve_local_file('tokenizer.model')
        ice_text_model = self._resolve_local_file('ice_text.model')
        try:
            import sentencepiece as spm
            if tokenizer_model is not None:
                self.sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif ice_text_model is not None:
                self.sp_model = spm.SentencePieceProcessor(ice_text_model)
            else:
                self.sp_model = None
        except:
            self.sp_model = None
        self.merge_txt = self._resolve_local_file('merges.txt')
        tokenizer_json_bpe = self._load_tokenizer_json_bpe()
        # Qwen2/Qwen3 are GPT-2 byte-level BPE. Whether the loaded tokenizer exposes
        # `mergeable_ranks` (tiktoken-style) varies across transformers versions: older
        # ones do -> the tiktoken longest-match path below produces token ids that don't
        # match HF (e.g. "Determine" -> [92648,68] not [35,24308]); newer ones don't.
        # To be version-independent, whenever this is a Qwen2/Qwen3 tokenizer that has an
        # HF BPE in tokenizer.json, always export via tokenizer.json (HUGGINGFACE type 3,
        # clean official merges) — skip both the tiktoken and the merges.txt paths.
        _tok_cls = type(self.tokenizer).__name__ if self.tokenizer is not None else ''
        is_qwen_bpe = ('Qwen2Tokenizer' in _tok_cls) or ('Qwen3Tokenizer' in _tok_cls)
        qwen_prefer_tokenizer_json = is_qwen_bpe and (tokenizer_json_bpe is not None)
        force_huggingface_export = self.is_fastvlm and (self.merge_txt is not None)
        # TOKENIZER MAGIC NUMBER
        MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        SENTENCEPIECE = 0; TIKTOIKEN = 1; BERT = 2; HUGGINGFACE = 3
        def write_line(fp, *args):
            for arg in args:
                for token in arg:
                    fp.write(str(token) + ' ')
            fp.write('\n')
        def write_header(fp, type, speicals, prefix = []):
            fp.write(f'{MAGIC_NUMBER} {type}\n')
            fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
            write_line(fp, speicals, self.stop_ids, prefix)

        # if not os.path.exists(self.args.dst_path):
        #     os.makedirs(self.args.dst_path)
        
        file_path = self.args.dst_path
        special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'all_special_ids'): #gemma3
            special_list.extend(self.tokenizer.all_special_ids)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)
            
        if self.generation_config is not None:
            generation_config = self.generation_config
            if hasattr(generation_config, 'user_token_id'):
                special_list.append(generation_config.user_token_id)
            if hasattr(generation_config, 'assistant_token_id'):
                special_list.append(generation_config.assistant_token_id)
                
        vocab_list = []
        prefix_list = []
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()
        if len(prefix_list) == 0:
            try:
                test_txt = 'A'
                ids = self.tokenizer.encode(test_txt)
                get_txt = self.tokenizer.decode(ids[-1])
                if len(ids) > 1 and get_txt == test_txt:
                    prefix_list += ids[:-1]
            except:
                pass

        if self.sp_model is not None:
            # senetencepiece
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6
            for i in range(self.sp_model.GetPieceSize()):
                token = self.sp_model.IdToPiece(i)
                score = self.sp_model.GetScore(i)
                token_type = NORMAL
                if self.sp_model.IsUnknown(i):
                    token_type = UNKNOWN
                elif self.sp_model.IsControl(i):
                    token_type = CONTROL
                elif self.sp_model.IsUnused(i):
                    token_type = UNUSED
                elif self.sp_model.IsByte(i):
                    token_type = BYTE
                if getattr(self.args, 'path', '') == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')
                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {token_type}\n')
            # add special tokens to vocab_list
            for index in special_list:
                if index >= len(vocab_list):
                    try:
                        token = self.tokenizer.decode(index)
                        token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                        vocab_list.append(f'{token_encode} {0} {NORMAL}\n')
                    except:
                        pass
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                if getattr(self, 'model_type', '') == "gemma3" or getattr(self, 'model_type', '') == "gemma3-text":
                    fp.write(f'{len(vocab_list) + 1}\n') # len(vocab_list)==262144, self.tokenizer([262144])=='image_soft_token' is a special token
                else:
                    fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif hasattr(self.tokenizer, 'mergeable_ranks') and not force_huggingface_export and not qwen_prefer_tokenizer_json:
            # tikton
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)
        elif self.merge_txt is not None and (tokenizer_json_bpe is None or force_huggingface_export):
            # huggingface tokenizer via merges.txt. Only used when tokenizer.json carries no
            # usable BPE (or FastVLM forces this path): a model's merges.txt can be a stale or
            # partial subset of tokenizer.json's merges (observed on MiniCPM, where the missing
            # CJK merges broke Chinese tokenization), so tokenizer.json is preferred below.
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]
            # load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k
            # load merge
            with open(self.merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)
            # write to tokenizer.txt
            family = self._pretok_family()
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)} {family}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        elif tokenizer_json_bpe is not None:
            # HuggingFace BPE tokenizer.json can carry merges without a merges.txt file
            # (for example Gemma4). Export it as type 3 to preserve BPE merge ranks.
            vocab_list, merge_list = tokenizer_json_bpe
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            family = self._pretok_family()

            # GPT-2 byte-level pieces never contain whitespace (space->Ġ, newline->Ċ), so they
            # can be written verbatim. Only b64-encode the rare token that would break the
            # line / space-separated-merge format (contains whitespace, or literally starts
            # with the "b64:" sentinel). This keeps the asset roughly half the size of
            # encoding everything while staying lossless. The device parser accepts both forms.
            def b64(tok):
                return 'b64:' + base64.b64encode(tok.encode('utf-8')).decode('utf8')
            def vocab_bad(tok):
                return ('\n' in tok) or ('\r' in tok) or tok.startswith('b64:') or tok == ''
            def merge_bad(tok):
                return vocab_bad(tok) or (' ' in tok) or ('\t' in tok)

            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)} {family}\n')
                for token in vocab_list:
                    fp.write((b64(token) if vocab_bad(token) else token) + '\n')
                for left, right in merge_list:
                    # The device merge parser requires both sides in the same form, so if
                    # either side needs b64 we encode both.
                    if merge_bad(left) or merge_bad(right):
                        fp.write(b64(left) + ' ' + b64(right) + '\n')
                    else:
                        fp.write(left + ' ' + right + '\n')
        else:
            # tiktoken or bert
            if 'bert' in type(self.tokenizer).__name__.lower():
                tokenizer_type = BERT
            else:
                tokenizer_type = TIKTOIKEN
            # bert tokenizer
            def unicode_to_byte(u: int):
                if u >= 256 and u <= 288:
                    return u - 256
                if u >= 289 and u <= 322:
                    return u - 162
                if u == 323:
                    return 173
                # need not convert using jinja
                # if u == 65372: # |
                #     return 124
                # if u == 9601:  # _
                #     return 95
                return u
            vocab = self.tokenizer.get_vocab()
            vocab_list = ['<unk>' for i in range(len(vocab))]
            for k, v in vocab.items():
                try:
                    vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                except:
                    vocab_list[int(v)] = k.encode('utf-8')

            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, tokenizer_type, special_list)
                fp.write(f'{len(vocab_list)}\n')
                for v in vocab_list:
                    line = base64.b64encode(v).decode("utf8") + "\n"
                    fp.write(line)
        return file_path


def main():
    parser = argparse.ArgumentParser(description='llm_exporter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--tokenizer_path', type=str, help='tokenizer path(hf path or local dir, for example: "Qwen/Qwen3-VL-2B-Instruct" "Qwen/Qwen3-1.7B")')
    parser.add_argument('--dst_path', type=str, default='./tokenizer', help='export tokenizer path, defaut is `./qwen3_tokenizer.txt`.')
    args = parser.parse_args()


    llm_exporter = LlmExporter(args)
    llm_exporter.export_tokenizer()

if __name__ == '__main__':
    main()
