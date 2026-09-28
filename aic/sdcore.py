"""Portable maths of the AI Image Create pipeline — a line-for-line port of the Python reference
(reference/sd_ref.py, canny.py, face_mask.py, clip_tokenizer.py) that the Android Kotlin app was parity-tested
against. Only change vs the reference: no Linux-only `resource` import, and the 2x-upscale resize uses a BLAS
tensordot instead of a naive einsum (same maths, ~100x faster at 4x resolution)."""
from __future__ import annotations

import html
import json
import math
import re

import numpy as np

from .resources_path import res

# ---------------- sizing / resize (portable) ----------------
def target_size(w, h, long_side=512, mult=64):
    """WxH (multiples of 64, area <= long_side^2, sides 256..1.5*long_side) with the closest aspect ratio
    (errors < 3 % count as equal, then the largest area). 3:2 -> 576x384, 4:3 -> 512x384, 16:9 -> 576x320, 1:1 -> 512x512."""
    budget = long_side * long_side; best = None; r = math.log(w / h)
    for W in range(256, int(long_side * 1.5) + 1, mult):
        for H in range(256, int(long_side * 1.5) + 1, mult):
            if W * H > budget: continue
            e = abs(math.log(W / H) - r); key = (int(e / 0.03), -W * H, e)
            if best is None or key < best[0]: best = (key, W, H)
    return best[1], best[2]


def _weights(n_out, n_in, x0, span):
    scale = span / n_out; support = max(1.0, scale)
    Wm = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        c = x0 + (i + 0.5) * scale
        lo = int(math.floor(c - support)); hi = int(math.ceil(c + support))
        tot = 0.0; ws = []
        for k in range(lo, hi + 1):
            wv = 1.0 - abs(k + 0.5 - c) / support
            if wv > 0: ws.append((min(max(k, 0), n_in - 1), wv)); tot += wv
        for k, wv in ws: Wm[i, k] += wv / tot
    return Wm


def resize_cover(rgb_u8, W, H):
    """Scale to cover WxH keeping aspect ratio, centre-crop (no stretching)."""
    h, w, _ = rgb_u8.shape; sc = max(W / w, H / h); cw = W / sc; ch = H / sc
    Rx = _weights(W, w, (w - cw) / 2.0, cw); Ry = _weights(H, h, (h - ch) / 2.0, ch)
    f = rgb_u8.astype(np.float64)
    out = np.einsum("yh,hwc->ywc", Ry, f); out = np.einsum("xw,ywc->yxc", Rx, out)
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def resize_stretch(rgb_u8, W, H):
    h, w, _ = rgb_u8.shape
    Rx = _weights(W, w, 0.0, float(w)); Ry = _weights(H, h, 0.0, float(h))
    out = np.einsum("yh,hwc->ywc", Ry, rgb_u8.astype(np.float64)); out = np.einsum("xw,ywc->yxc", Rx, out)
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def resize_stretch_fast(rgb_u8, W, H):
    """resize_stretch with BLAS contractions (used for the big x4 -> x2 step of the upscaler)."""
    h, w, _ = rgb_u8.shape
    Rx = _weights(W, w, 0.0, float(w)); Ry = _weights(H, h, 0.0, float(h))
    f = rgb_u8.astype(np.float64)
    out = np.tensordot(Ry, f, axes=([1], [0]))                    # (H, w, 3)
    out = np.tensordot(Rx, out, axes=([1], [1])).transpose(1, 0, 2)  # (W, H, 3) -> (H, W, 3)
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


# ---------------- RNG (portable splitmix64 + Box-Muller) ----------------
M64 = (1 << 64) - 1


def gaussian(seed, stream, n):
    base = np.uint64((seed * 1000003 + stream) & M64)
    k = np.arange(1, 2 * ((n + 1) // 2) + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = base + k * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 1.0) * (1.0 / 9007199254740992.0)
    u1 = u[0::2]; u2 = u[1::2]; r = np.sqrt(-2.0 * np.log(u1)); th = 2.0 * math.pi * u2
    out = np.empty(u1.size * 2); out[0::2] = r * np.cos(th); out[1::2] = r * np.sin(th)
    return out[:n].astype(np.float32)


# ---------------- LCM scheduler ----------------
BETAS = (np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2)
ACP = np.cumprod(1.0 - BETAS)


def lcm_timesteps(strength, steps):
    orig = np.arange(1, int(50 * strength) + 1) * 20 - 1
    idx = np.floor(np.linspace(0, len(orig), num=steps, endpoint=False)).astype(np.int64)
    ts = orig[::-1][idx]
    return [int(t) for t in dict.fromkeys(ts.tolist())]


def w_embedding(w, dim=256):
    hd = dim // 2; e = np.float32(math.log(10000.0)) / np.float32(hd - 1)
    f = np.exp(np.arange(hd, dtype=np.float32) * -e); v = np.float32((w - 1.0) * 1000.0) * f
    return np.concatenate([np.sin(v), np.cos(v)])[None].astype(np.float32)


# ---------------- colour keeping ----------------
def _blur_axis(a, sig, axis):
    rad = int(math.ceil(3 * sig)); k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sig) ** 2); k /= k.sum()
    n = a.shape[axis]; idx = np.clip(np.arange(-rad, n + rad), 0, n - 1)
    p = np.take(a, idx, axis=axis); out = np.zeros_like(a)
    for j, kv in enumerate(k):
        sl = [slice(None)] * a.ndim; sl[axis] = slice(j, j + n); out += kv * p[tuple(sl)]
    return out


def gblur(a, sig): return _blur_axis(_blur_axis(a, sig, 0), sig, 1)


def rgb2ycc(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return np.stack([0.299 * r + 0.587 * g + 0.114 * b, -0.168736 * r - 0.331264 * g + 0.5 * b, 0.5 * r - 0.418688 * g - 0.081312 * b], -1)


def ycc2rgb(y):
    Y, Cb, Cr = y[..., 0], y[..., 1], y[..., 2]
    return np.stack([Y + 1.402 * Cr, Y - 0.344136 * Cb - 0.714136 * Cr, Y + 1.772 * Cb], -1)


def keep_colors(out_u8, ref_u8, luma=0.6, chroma=1.0):
    if luma <= 0 and chroma <= 0: return out_u8
    sig = 0.02 * max(out_u8.shape[:2])
    o = rgb2ycc(out_u8.astype(np.float64)); r = rgb2ycc(ref_u8.astype(np.float64))
    d = gblur(r, sig) - gblur(o, sig)
    o[..., 0] += luma * d[..., 0]; o[..., 1:] += chroma * d[..., 1:]
    return np.clip(np.floor(ycc2rgb(o) + 0.5), 0, 255).astype(np.uint8)


def init_transform(rgb_u8, kind, amount=0.55):
    if not kind: return rgb_u8
    if kind == "sketch":
        g = gray_u8(rgb_u8).astype(np.float64)
        g = 255.0 - (255.0 - g) * amount
        return np.repeat(np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    raise ValueError(kind)


# ---------------- Canny (== cv2.Canny(100, 200)) ----------------
def gray_u8(rgb_u8):
    r = rgb_u8[..., 0].astype(np.int32); g = rgb_u8[..., 1].astype(np.int32); b = rgb_u8[..., 2].astype(np.int32)
    return ((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14).astype(np.uint8)


def canny(gray, low=100, high=200):
    H, W = gray.shape
    p = np.pad(gray.astype(np.int32), 1, mode="edge")
    dx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    dy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.abs(dx) + np.abs(dy)
    M = np.zeros((H + 2, W + 2), np.int64); M[1:-1, 1:-1] = mag
    ax = np.abs(dx).astype(np.int64); ay = np.abs(dy).astype(np.int64) << 15
    tg22x = ax * 13573; tg67x = tg22x + (ax << 16)
    c = M[1:-1, 1:-1]; L = M[1:-1, :-2]; R = M[1:-1, 2:]; U = M[:-2, 1:-1]; D = M[2:, 1:-1]
    s = np.where((dx ^ dy) < 0, -1, 1)
    UL = M[:-2, :-2]; UR = M[:-2, 2:]; DL = M[2:, :-2]; DR = M[2:, 2:]
    Up_s = np.where(s < 0, UR, UL); Dn_s = np.where(s < 0, DL, DR)
    horiz = ay < tg22x; vert = (~horiz) & (ay > tg67x); diag = (~horiz) & (~vert)
    keep = np.zeros((H, W), bool)
    keep |= horiz & (c > L) & (c >= R); keep |= vert & (c > U) & (c >= D); keep |= diag & (c > Up_s) & (c > Dn_s)
    keep &= c > low
    strong = keep & (c > high); weak = keep & ~strong
    out = np.zeros((H, W), np.uint8); out[strong] = 1
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for yy in (y - 1, y, y + 1):
            if yy < 0 or yy >= H: continue
            for xx in (x - 1, x, x + 1):
                if 0 <= xx < W and weak[yy, xx] and not out[yy, xx]:
                    out[yy, xx] = 1; stack.append((yy, xx))
    return out


# ---------------- face "likeness" mask ----------------
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def fill_polygon(H, W, pts):
    m = np.zeros((H, W), np.float64); n = len(pts)
    for y in range(H):
        yc = y + 0.5; xs = []
        for i in range(n):
            x0, y0 = pts[i]; x1, y1 = pts[(i + 1) % n]
            if (y0 <= yc < y1) or (y1 <= yc < y0): xs.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a = max(0, math.ceil(xs[k] - 0.5)); b = min(W, math.ceil(xs[k + 1] - 0.5))
            if b > a: m[y, a:b] = 1.0
    return m


def mask_from_ovals(H, W, ovals, grow=0.12, feather=0.08):
    m = np.zeros((H, W), np.float64)
    for pts in ovals:
        pts = np.asarray(pts, np.float64); c = pts.mean(0); size = float(np.max(pts.max(0) - pts.min(0)))
        p = c + (pts - c) * (1 + grow)
        top = p[:, 1] < c[1]; p[top, 1] -= 0.10 * size
        mm = fill_polygon(H, W, p)
        k = max(3, int(feather * size) | 1); sig = k / 2.0
        mm = _blur_axis(_blur_axis(mm, sig, 0), sig, 1)
        m = np.maximum(m, mm)
    return np.clip(m, 0, 1).astype(np.float32)


# ---------------- 2x upscaler (portable parts) ----------------
def _lanczos_taps(n_in):
    out = []
    for i in range(n_in * 2):
        c = (i + 0.5) / 2.0 - 0.5; f0 = math.floor(c); idx = []; ws = []
        for j in range(6):
            k = f0 - 2 + j; d = c - k
            wv = (1.0 if d == 0 else math.sin(math.pi * d) / (math.pi * d)) * (1.0 if d == 0 else math.sin(math.pi * (d / 3.0)) / (math.pi * (d / 3.0))) if abs(d) < 3.0 else 0.0
            idx.append(min(max(k, 0), n_in - 1)); ws.append(wv)
        t = sum(ws); out.append((idx, [w / t for w in ws]))
    return out


def lanczos2x(rgb_u8):
    H, W, _ = rgb_u8.shape; f = rgb_u8.astype(np.float64)
    ry = _lanczos_taps(H); rx = _lanczos_taps(W)
    tmp = np.zeros((2 * H, W, 3))
    for j in range(6):
        idx = np.array([r[0][j] for r in ry]); w = np.array([r[1][j] for r in ry])
        tmp += w[:, None, None] * f[idx]
    out = np.zeros((2 * H, 2 * W, 3))
    for j in range(6):
        idx = np.array([r[0][j] for r in rx]); w = np.array([r[1][j] for r in rx])
        out += w[None, :, None] * tmp[:, idx]
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


UPSCALE_BLEND = 0.7

# ---------------- prompts / presets / safety constants ----------------
PRESETS = json.loads(res("presets.json").read_text(encoding="utf-8"))
DEFAULTS = PRESETS.pop("_defaults")
NSFW_NEGATIVE = "nude, naked, nsfw, nipples, genitals, sexual"   # always appended, never shown, not removable
NSFW_THRESHOLDS = {"standard": 0.5, "relaxed": 0.85}              # block if P(nsfw) > threshold


def nsfw_threshold(strictness: str) -> float:
    """Unknown / tampered values fall back to the STRICTER level — there is no 'off'."""
    return NSFW_THRESHOLDS.get(str(strictness).lower(), NSFW_THRESHOLDS["standard"])


def full_negative(preset, user_negative=None):
    p = PRESETS.get(preset, {}) if preset else {}
    parts = [DEFAULTS["negative"] if user_negative is None else user_negative.strip(), p.get("negative_extra", ""), NSFW_NEGATIVE]
    return ", ".join(x for x in parts if x)


def face_strength_for(strength, face_delta):
    return max(DEFAULTS["face_min"], min(strength, strength - face_delta))


def preset_kwargs(preset, quality=True):
    p = PRESETS[preset]
    kw = dict(strength=p["strength"], steps=DEFAULTS["steps"], w=p["w"], adapter_scale=p["adapter"], keep_luma=p["keep_luma"],
              keep_chroma=p["keep_chroma"], init=p.get("init"), init_amount=p.get("init_amount", 0.55), encoder=DEFAULTS["encoder"], decoder=DEFAULTS["decoder"])
    kw.update(cfg=p.get("cfg", DEFAULTS["cfg"]), negative=full_negative(preset), cfg_steps=99 if quality else 1)
    kw["face_delta"] = p.get("face_delta", 0.25)
    return kw


_INSTR = re.compile(r"^(please\s+)?(make|turn|change|convert|transform|put)\s+(it|this|the photo|the picture|the image|me|him|her|them|us)\s*(into|to|look like|like|in|at|on)?\s*", re.I)


def clean_user_prompt(t):
    t = re.sub(r"\s+", " ", (t or "")).strip().rstrip(".!")
    return _INSTR.sub("", t).strip()


def build_prompt(user, preset):
    p = PRESETS[preset] if preset else None
    user = clean_user_prompt(user)
    if p is None: return user
    return p["prompt"].replace("{}", user) if user else p["prompt_empty"]


# ---------------- CLIP BPE tokenizer ----------------
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class ClipTokenizer:
    BOS, EOS, MAXLEN = 49406, 49407, 77

    def __init__(self, vocab_path=None, merges_path=None):
        import regex
        self._regex = regex
        self.PAT = regex.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex.IGNORECASE)
        vocab_path = vocab_path or res("tokenizer", "vocab.json"); merges_path = merges_path or res("tokenizer", "merges.txt")
        self.enc = json.load(open(vocab_path, encoding="utf-8"))
        lines = open(merges_path, encoding="utf-8").read().split("\n")[1:49152 - 256 - 2 + 1]
        self.ranks = {tuple(l.split()): i for i, l in enumerate(lines)}
        self.b2u = _bytes_to_unicode(); self.cache = {}

    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 30))
            if best not in self.ranks: break
            a, b = best; out = []; i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b: out.append(a + b); i += 2
                else: out.append(word[i]); i += 1
            word = out
        self.cache[token] = word; return word

    def encode(self, text):
        text = html.unescape(html.unescape(text))
        text = self._regex.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in self.PAT.findall(text):
            u = "".join(self.b2u[b] for b in tok.encode("utf-8"))
            ids += [self.enc[p] for p in self.bpe(u)]
        ids = [self.BOS] + ids[:self.MAXLEN - 2] + [self.EOS]
        return ids + [self.EOS] * (self.MAXLEN - len(ids))
