"""Minimal CLIP BPE tokenizer (portable: same algorithm will be ported to Kotlin). Matches HF CLIPTokenizer
(without ftfy) for SD 1.x: lower-case, whitespace collapse, byte-level BPE, BOS 49406, EOS/PAD 49407, max 77."""
import json, regex, html
PAT = regex.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex.IGNORECASE)
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
class ClipTokenizer:
    BOS, EOS, MAXLEN = 49406, 49407, 77
    def __init__(self, vocab_path, merges_path):
        self.enc = json.load(open(vocab_path, encoding="utf-8"))
        lines = open(merges_path, encoding="utf-8").read().split("\n")[1:49152-256-2+1]
        self.ranks = {tuple(l.split()): i for i, l in enumerate(lines)}
        self.b2u = bytes_to_unicode(); self.cache = {}
    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks: break
            a, b = best; out = []; i = 0
            while i < len(word):
                if i < len(word)-1 and word[i] == a and word[i+1] == b: out.append(a+b); i += 2
                else: out.append(word[i]); i += 1
            word = out
        self.cache[token] = word; return word
    def encode(self, text):
        text = html.unescape(html.unescape(text))
        text = regex.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in PAT.findall(text):
            u = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            ids += [self.enc[p] for p in self.bpe(u)]
        ids = [self.BOS] + ids[:self.MAXLEN-2] + [self.EOS]
        return ids + [self.EOS] * (self.MAXLEN - len(ids))
