"""The image-to-image pipeline — port of reference/sd_ref.py Pipeline.generate and the Android SdPipeline.kt:
CLIP text encoder (+ negative for CFG) -> TAESD encoder -> Canny + T2I-Adapter -> 4-step LCM with differential
"keep face" strength -> SD VAE decoder -> colour keeping -> NSFW classifier (ALWAYS; there is no way to skip it)
-> optional 2x upscale (only for images that passed the filter)."""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image, ImageOps

from . import sdcore as C
from .engine import Engine

LONG_SIDES = {"standard": 512, "large": 768}
# ETA model: stage costs in "UNet evaluations" (same as the Android app)
COST = dict(TEXT=0.25, ENC=0.1, ADAPTER=0.15, DECODE=2.4, SAFETY=0.15, UPSCALE=0.7)


class Cancelled(Exception):
    pass


@dataclass
class GenParams:
    prompt: str = ""
    strength: float = 0.55
    quality: str = "best"         # "best" (CFG every step, 8 UNet evals) | "fast" (CFG first step only, 5 evals)
    seed: int = 1234
    keep_face: bool = True
    keep_colors: bool = True
    upscale: bool = True
    strictness: str = "relaxed"   # "relaxed" (block if > 0.85, default) | "standard" (block if > 0.5)
    size: str = "standard"        # "standard" (area <= 512^2, e.g. 576x384) | "large" (area <= 768^2, e.g. 896x576)
    preset: str = "none"          # the Windows UI always uses "none" (prompt only); other presets kept for parity tests


@dataclass
class GenResult:
    image: np.ndarray | None      # None when the safety filter blocked the picture (nothing may be shown or saved)
    base: np.ndarray | None       # model-resolution output (before the 2x upscale); None when blocked
    input: np.ndarray             # the prepared (cropped, resized) input the model saw
    blocked: bool
    nsfw: float
    threshold: float
    prompt: str
    negative: str
    seed: int
    seconds: float
    timesteps: list
    size: tuple
    faces: int
    timings: dict = field(default_factory=dict)
    provider: str = "CPU"
    safety_checked: bool = False

    def summary(self):
        d = {k: v for k, v in asdict(self).items() if k not in ("image", "base", "input")}
        d["negative"] = "(hidden)"  # the NSFW terms are never shown in the UI
        return d


def load_photo(path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(im)


def prepare(decoded_rgb: np.ndarray, size: str = "standard") -> np.ndarray:
    W, H = C.target_size(decoded_rgb.shape[1], decoded_rgb.shape[0], LONG_SIDES.get(size, 512))
    return C.resize_cover(decoded_rgb, W, H)


def new_seed() -> int:
    return random.SystemRandom().randrange(1, 2 ** 31)


class Pipeline:
    def __init__(self, engine: Engine):
        self.e = engine
        self.tok = C.ClipTokenizer()

    # -- helpers
    def _text(self, text):
        return self.e.run("text_encoder", {"input_ids": np.array([self.tok.encode(text)], np.int64)})[0]

    def nsfw_score(self, img_u8) -> float:
        sm = C.resize_stretch(img_u8, 224, 224).astype(np.float32) / 255.0
        px = ((sm - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
        return float(self.e.run("safety", {"pixel_values": px})[0][0, 1])

    def upscale2x(self, rgb_u8):
        x = (rgb_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        y = self.e.run("upscaler", {"image": x})[0][0]
        x4 = np.clip(np.floor(y.transpose(1, 2, 0) * 255.0 + 0.5), 0, 255).astype(np.uint8)
        H, W, _ = rgb_u8.shape
        esr = C.resize_stretch_fast(x4, W * 2, H * 2).astype(np.float64); lz = C.lanczos2x(rgb_u8).astype(np.float64)
        return np.clip(np.floor(C.UPSCALE_BLEND * esr + (1 - C.UPSCALE_BLEND) * lz + 0.5), 0, 255).astype(np.uint8)

    # -- main entry used by the app
    def generate(self, decoded_rgb: np.ndarray, p: GenParams, progress=None, cancel: threading.Event | None = None,
                 sec_per_eval_hint: float | None = None) -> GenResult:
        progress = progress or (lambda label, frac, eta: None); cancel = cancel or threading.Event()
        rgb = prepare(decoded_rgb, p.size)
        mask, ovals = (None, [])
        if p.keep_face:
            from .face import face_mask
            mask, ovals = face_mask(rgb)
        kw = C.preset_kwargs(p.preset, p.quality != "fast")
        kw["strength"] = float(p.strength)
        if not p.keep_colors: kw["keep_luma"] = 0.0; kw["keep_chroma"] = 0.0
        return self.run_core(rgb, C.build_prompt(p.prompt, p.preset), seed=int(p.seed), face_mask=mask, faces=len(ovals),
                             strictness=p.strictness, upscale=p.upscale, progress=progress, cancel=cancel,
                             sec_per_eval_hint=sec_per_eval_hint, **kw)

    def run_core(self, rgb, prompt, strength=0.5, steps=4, w=8.0, adapter_scale=0.6, seed=1234, decoder="vae_decoder",
                 encoder="taesd_encoder", negative=None, cfg=1.0, keep_luma=0.0, keep_chroma=0.0, cfg_steps=99, init=None,
                 init_amount=0.55, face_mask=None, face_strength=None, face_delta=None, faces=0, strictness="relaxed",
                 upscale=False, progress=None, cancel=None, sec_per_eval_hint=None) -> GenResult:
        """Line-for-line port of the reference generate(); the safety classifier is not optional."""
        progress = progress or (lambda label, frac, eta: None); cancel = cancel or threading.Event()

        def check():
            if cancel.is_set(): raise Cancelled()
        t0 = time.time(); H, W, _ = rgb.shape; self.e.timing = {}
        if negative is None: negative = C.full_negative(None)
        if C.NSFW_NEGATIVE not in negative:                      # belt and braces: the NSFW terms are always there
            negative = ", ".join(x for x in (negative, C.NSFW_NEGATIVE) if x)
        ts = C.lcm_timesteps(strength, steps)
        evals = len(ts) + (min(cfg_steps, len(ts)) if cfg > 1.0 else 0)
        total = COST["TEXT"] * 2 + COST["ENC"] + COST["ADAPTER"] + evals + COST["DECODE"] + COST["SAFETY"] + (COST["UPSCALE"] if upscale else 0)
        st = {"units": 0.0, "spu": sec_per_eval_hint}

        def step(label, add):
            st["units"] += add
            progress(label, min(1.0, st["units"] / total), (total - st["units"]) * st["spu"] if st["spu"] else None)

        progress("Reading your prompt…", 0.0, total * sec_per_eval_hint if sec_per_eval_hint else None)
        cond = self._text(prompt); check()
        unc = self._text(negative) if cfg > 1.0 else None
        step("Encoding the photo…", COST["TEXT"] * 2); check()
        x = (C.init_transform(rgb, init, init_amount).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        lat = self.e.run(encoder, {"image": x})[0]
        h, wl = lat.shape[2], lat.shape[3]
        step("Finding edges…", COST["ENC"]); check()
        if adapter_scale > 0:
            ed = C.canny(C.gray_u8(rgb)).astype(np.float32)[None, None]
            res = [r * np.float32(adapter_scale) for r in self.e.run("adapter_canny", {"edges": ed})]
        else:
            res = [np.zeros((1, c, h // f, wl // f), np.float32) for c, f in [(320, 1), (640, 2), (1280, 4), (1280, 8)]]
        step("Creating…", COST["ADAPTER"]); check()
        wemb = C.w_embedding(w)
        n = lat.size; a0 = C.ACP[ts[0]]; eps0 = C.gaussian(seed, 0, n).reshape(lat.shape)
        z = (np.float32(math.sqrt(a0)) * lat + np.float32(math.sqrt(1 - a0)) * eps0).astype(np.float32)
        if face_strength is None and face_delta is not None and face_mask is not None:
            face_strength = C.face_strength_for(strength, face_delta)
        smap = None
        if face_mask is not None and face_strength is not None and face_strength < strength:
            mk = face_mask.reshape(h, 8, wl, 8).mean(axis=(1, 3)).astype(np.float32)
            smap = (np.float32(strength) * (1 - mk) + np.float32(face_strength) * mk)[None, None]

        if "unet" not in self.e.sessions:            # load the big model before timing the steps (keeps the ETA honest)
            progress("Loading the image generator…", min(1.0, st["units"] / total), None); self.e.session("unet"); check()

        def unet(feed):
            te = time.time(); o = self.e.run("unet", feed)[0]; sec = time.time() - te
            st["spu"] = sec if st["spu"] is None else 0.5 * st["spu"] + 0.5 * sec
            return o
        for i, t in enumerate(ts):
            check()
            if smap is not None:
                keep = (smap * 1000.0 < t)
                if keep.any():
                    at = C.ACP[t]; orig = (np.float32(math.sqrt(at)) * lat + np.float32(math.sqrt(1 - at)) * eps0).astype(np.float32)
                    z = np.where(keep, orig, z).astype(np.float32)
            feed = {"sample": z, "timestep": np.array([t], np.int64), "encoder_hidden_states": cond, "timestep_cond": wemb}
            for k in range(4): feed[f"adapter_res{k}"] = res[k]
            eps = unet(feed)
            step(f"Creating… step {i + 1} of {len(ts)}", 1.0); check()
            if unc is not None and i < cfg_steps:
                feed["encoder_hidden_states"] = unc; eu = unet(feed); eps = eu + np.float32(cfg) * (eps - eu)
                step(f"Creating… step {i + 1} of {len(ts)}", 1.0); check()
            a = C.ACP[t]; pt = ts[i + 1] if i + 1 < len(ts) else -1; ap = C.ACP[pt] if pt >= 0 else 1.0
            stt = t * 10.0; cskip = 0.25 / (stt * stt + 0.25); cout = stt / math.sqrt(stt * stt + 0.25)
            x0 = (z - np.float32(math.sqrt(1 - a)) * eps) / np.float32(math.sqrt(a))
            den = (np.float32(cout) * x0 + np.float32(cskip) * z).astype(np.float32)
            if pt >= 0: z = (np.float32(math.sqrt(ap)) * den + np.float32(math.sqrt(1 - ap)) * C.gaussian(seed, i + 1, n).reshape(z.shape)).astype(np.float32)
            else: z = den
        progress("Finishing the picture…", min(1.0, st["units"] / total), (total - st["units"]) * st["spu"] if st["spu"] else None)
        img = self.e.run(decoder, {"latent": z})[0][0]
        out = np.clip(np.floor((np.clip(img, -1, 1).transpose(1, 2, 0) + 1.0) * 127.5 + 0.5), 0, 255).astype(np.uint8)
        out = C.keep_colors(out, rgb, keep_luma, keep_chroma)
        step("Safety check…", COST["DECODE"])
        # ---- NSFW classifier: always runs, cannot be switched off ----
        nsfw = self.nsfw_score(out)
        thr = C.nsfw_threshold(strictness)
        blocked = not (nsfw <= thr)                 # NaN or anything above the threshold -> blocked
        final = None; base = None
        if not blocked:
            base = out
            if upscale:
                step("Sharpening (2× upscale)…", COST["SAFETY"]); check()
                final = self.upscale2x(out)
            else:
                final = out
        else:
            out = None                               # drop the pixels right away
        progress("Done", 1.0, 0.0)
        per = self.e.info.per_model
        prov = "DirectML" if per.get("unet") == "DirectML" else "CPU"
        timings = dict(self.e.timing); timings["sec_per_unet_eval"] = st["spu"]
        return GenResult(image=final, base=base, input=rgb, blocked=blocked, nsfw=nsfw, threshold=thr, prompt=prompt,
                         negative=negative, seed=seed, seconds=time.time() - t0, timesteps=ts, size=(W, H), faces=faces,
                         timings=timings, provider=prov, safety_checked=True)
