"""Command-line modes of AIImageCreate(.exe) / AIImageCreate_cli.exe:

  (no args)                      start the app
  --selftest --photo P           headless end-to-end test on CPU: models -> one picture -> NSFW check -> PNG + JSON report
  --dml-smoke                    (with --selftest) also open the GPU models on DirectML and report the providers used
  --generate IN OUT --prompt T   make one picture (same defaults as the app)
  --parity-cases TESTDATA OUT    run testdata/parity_cases.json through the app pipeline (CPU) for the parity test
  --samples PHOTO OUT            sample sheet: original + several prompt-only results
  --bench                        UNet speed on the selected device (standard + large)
  --derive-fp16                  build (and SHA-check) the half-precision UNet used on DirectML GPUs
  --download / --import DIR      install the models (from the internet / from a folder with the release files)
  --screenshots DIR              render the UI states to PNGs
  --gui-smoke SECONDS            start the GUI, close it after SECONDS and write a marker file

The NSFW safety filter is part of every mode that produces a picture; no option, file or environment variable skips it.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__

log = logging.getLogger("aic")


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / "AIImageCreate"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _setup_logging(verbose=True):
    handlers = [logging.FileHandler(app_data_dir() / "aiimagecreate.log", encoding="utf-8")]
    if verbose and sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)


def versions():
    import onnxruntime as ort
    from PIL import __version__ as pilv
    try:
        import mediapipe as mp; mpv = mp.__version__
    except Exception as e:  # noqa: BLE001
        mpv = f"error: {e}"
    return dict(app=__version__, python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), cpu_count=os.cpu_count(), onnxruntime=ort.__version__,
                providers=ort.get_available_providers(), mediapipe=mpv, numpy=np.__version__, pillow=pilv,
                frozen=bool(getattr(sys, "frozen", False)))


def _store(args):
    from .models import ModelStore
    return ModelStore(Path(args.models) if args.models else (Path(os.environ["AIC_MODELS"]) if os.environ.get("AIC_MODELS") else None))


def _install(store, import_dir=None):
    from . import models as M
    last = [0.0]

    def cb(p):
        now = time.time()
        if now - last[0] > 3 or p.stage == "done":
            last[0] = now
            log.info("models: %s %s %.1f/%.1f MB %.1f MB/s", p.stage, p.file, p.done / 1e6, p.total / 1e6, p.bps / 1e6)
    if import_dir:
        got = store.import_from(Path(import_dir), cb); log.info("imported %s", got)
    if not store.all_installed():
        store.ensure(cb)
    assert store.all_installed(), f"models missing: {[s.name for s in store.missing()]}"
    return M


def _pipeline(store, device="cpu", gpu_half=True):
    from .engine import Engine
    from .pipeline import Pipeline
    return Pipeline(Engine(store, device, gpu_half=gpu_half))


def _printer():
    last = [0.0]

    def cb(label, frac, eta):
        now = time.time()
        if now - last[0] > 1.5 or frac >= 1.0:
            last[0] = now
            log.info("  %3.0f%%  %s%s", frac * 100, label, f"  (about {eta:.0f}s left)" if eta else "")
    return cb


def _save_png(img, path):
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


# ------------------------------------------------------------------ modes
def selftest(args) -> int:
    from .pipeline import GenParams, load_photo
    from . import sdcore as C
    out = Path(args.out or (app_data_dir() / "selftest")); out.mkdir(parents=True, exist_ok=True)
    rep = {"versions": versions(), "ok": False, "steps": {}}
    t0 = time.time()
    try:
        store = _store(args); rep["models_dir"] = str(store.dir)
        _install(store, args.import_dir); rep["steps"]["models"] = "ok"
        pipe = _pipeline(store, "cpu")
        photo = load_photo(args.photo)
        p = GenParams(prompt=args.prompt or "watercolor painting", seed=args.seed)
        r = pipe.generate(photo, p, progress=_printer())
        rep["result"] = r.summary()
        rep["nsfw_filter"] = {"ran": bool(r.safety_checked and "safety" in r.timings), "score": r.nsfw,
                              "threshold": r.threshold, "strictness": p.strictness, "blocked": r.blocked,
                              "nsfw_terms_in_negative": C.NSFW_NEGATIVE in r.negative,
                              "safety_model_seconds": r.timings.get("safety")}
        assert rep["nsfw_filter"]["ran"], "NSFW classifier did not run"
        assert rep["nsfw_filter"]["nsfw_terms_in_negative"], "NSFW terms missing from the negative prompt"
        if r.blocked:
            raise AssertionError(f"selftest picture was blocked by the filter (score {r.nsfw:.3f})")
        W, H = r.size
        assert r.base.shape == (H, W, 3) and r.image.shape == (2 * H, 2 * W, 3), "unexpected output size"
        assert float(r.image.std()) > 10, "output looks empty"
        of = out / "selftest_result.png"; _save_png(r.image, of)
        _save_png(r.input, out / "selftest_input.png")
        rep["output"] = {"file": str(of), "bytes": of.stat().st_size, "size": [2 * W, 2 * H], "model_size": [W, H]}
        rep["steps"]["generate_cpu"] = "ok"; rep["provider"] = r.provider
        rep["seconds_generate"] = round(r.seconds, 1)
        # second picture in the same session (Regenerate) — shows warm speed
        r2 = pipe.generate(photo, GenParams(prompt=p.prompt, seed=args.seed + 1, upscale=False))
        rep["seconds_regenerate_no_upscale"] = round(r2.seconds, 1); rep["regenerate_nsfw"] = r2.nsfw
        rep["sec_per_unet_eval_cpu"] = r2.timings.get("sec_per_unet_eval")
        if args.dml_smoke:
            rep["dml_smoke"] = dml_smoke(store)
        rep["ok"] = True
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"; rep["traceback"] = traceback.format_exc()
        log.error("selftest failed: %s", rep["traceback"])
    rep["seconds_total"] = round(time.time() - t0, 1)
    (out / "selftest_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    log.info("SELFTEST %s -> %s", "PASS" if rep["ok"] else "FAIL", out / "selftest_report.json")
    return 0 if rep["ok"] else 1


def dml_smoke(store) -> dict:
    """Open the DirectML sessions for real and report which provider ONNX Runtime actually used."""
    import onnxruntime as ort
    from .engine import Engine, gpu_adapter_name
    from . import models as M
    d = {"available_providers": ort.get_available_providers(), "adapters": gpu_adapter_name()}
    if "DmlExecutionProvider" not in d["available_providers"]:
        d["status"] = "DirectML provider not in this onnxruntime build"; return d
    for name in ("upscaler", "vae_decoder"):
        path = str(store.onnx(name))
        try:
            so = ort.SessionOptions(); so.enable_mem_pattern = False; so.log_severity_level = 3
            s = ort.InferenceSession(path, so, providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
            d[f"{name}_providers"] = s.get_providers()
            if name == "upscaler":
                t = time.time(); y = s.run(None, {"image": np.random.rand(1, 3, 64, 96).astype(np.float32)})[0]
                d["upscaler_run"] = {"out_shape": list(y.shape), "seconds": round(time.time() - t, 3), "finite": bool(np.isfinite(y).all())}
        except Exception as e:  # noqa: BLE001
            d[f"{name}_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    e = Engine(store, "auto")
    try:
        e.session("upscaler")
    except Exception as ex:  # noqa: BLE001
        d["engine_error"] = str(ex)
    d["engine_active"] = e.info.active; d["engine_label"] = e.info.label(); d["engine_fallback_reason"] = e.info.fallback_reason
    d["fp16_unet_installed"] = M.fp16_installed(store)
    used = d.get("upscaler_providers") or []
    d["status"] = "DirectML used" if used[:1] == ["DmlExecutionProvider"] else f"DirectML not used (got {used})"
    return d


def parity_cases(args) -> int:
    from .pipeline import GenParams, load_photo
    import onnxruntime as ort
    td, out = Path(args.parity_cases[0]), Path(args.parity_cases[1]); out.mkdir(parents=True, exist_ok=True)
    store = _store(args); _install(store, args.import_dir)
    pipe = _pipeline(store, "cpu")
    res = {"onnxruntime": ort.__version__, "versions": versions(), "cases": []}
    for c in json.loads((td / "parity_cases.json").read_text()):
        p = GenParams(prompt=c["prompt"], strength=c["strength"], quality=c["quality"], seed=c["seed"], keep_face=c["keep_face"],
                      keep_colors=c["keep_colors"], upscale=c["upscale"], size=c["size"])
        r = pipe.generate(load_photo(td / c["photo"]), p)
        assert r.safety_checked and not r.blocked
        _save_png(r.base, out / f"{c['key']}_base.png")
        if c["upscale"]: _save_png(r.image, out / f"{c['key']}_final.png")
        row = dict(key=c["key"], size=list(r.size), faces=r.faces, nsfw=r.nsfw, timesteps=r.timesteps, seconds=round(r.seconds, 1),
                   sec_per_unet_eval=r.timings.get("sec_per_unet_eval"))
        log.info("%s", json.dumps(row)); res["cases"].append(row)
    (out / "app_cases.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


def generate(args) -> int:
    from .pipeline import GenParams, load_photo
    store = _store(args); _install(store, args.import_dir)
    pipe = _pipeline(store, args.device, not args.no_half)
    p = GenParams(prompt=args.prompt or "", strength=args.strength, quality=args.quality, seed=args.seed,
                  keep_face=not args.no_face, keep_colors=not args.no_colors, upscale=not args.no_upscale,
                  strictness=args.strictness if args.strictness in ("standard", "relaxed") else "relaxed", size=args.size)
    r = pipe.generate(load_photo(args.generate[0]), p, progress=_printer())
    info = r.summary(); info["device"] = pipe.e.info.label(); info["unet_variant"] = pipe.e.unet_variant
    if r.blocked:
        log.warning("The safety filter blocked this picture (score %.3f > %.2f). Nothing was saved.", r.nsfw, r.threshold)
    else:
        _save_png(r.image, args.generate[1])
    Path(str(args.generate[1]) + ".json").write_text(json.dumps(info, indent=1, default=str))
    log.info("%s", json.dumps(info, default=str))
    return 2 if r.blocked else 0


SAMPLE_PROMPTS = ["watercolor painting", "anime style", "make it a beach at sunset", "oil painting", "cyberpunk city at night"]


def samples(args) -> int:
    from PIL import Image, ImageDraw, ImageFont
    from .pipeline import GenParams, load_photo, prepare
    store = _store(args); _install(store, args.import_dir)
    pipe = _pipeline(store, args.device)
    photo = load_photo(args.samples[0]); out = Path(args.samples[1])
    tiles = [("Original", prepare(photo))]; meta = []
    for pr in (args.prompts or SAMPLE_PROMPTS):
        r = pipe.generate(photo, GenParams(prompt=pr, seed=args.seed), progress=_printer())
        meta.append(dict(prompt=pr, nsfw=r.nsfw, blocked=r.blocked, seconds=round(r.seconds, 1), provider=r.provider))
        if r.blocked:
            ph = np.full_like(tiles[0][1], 40); tiles.append((f'"{pr}" — blocked by the safety filter', ph))
        else:
            tiles.append((f'"{pr}"', r.image))
    tw = 600; th = int(round(tiles[0][1].shape[0] * tw / tiles[0][1].shape[1])); cols = 3; rows = (len(tiles) + cols - 1) // cols
    pad, lab = 12, 40
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th + lab) + (rows + 1) * pad + 50), (24, 26, 32))
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("segoeui.ttf", 24); tfont = ImageFont.truetype("segoeuib.ttf", 26)
    except OSError:
        font = tfont = ImageFont.load_default()
    d.text((pad, 12), f"AI Image Create for Windows {__version__} — prompt only, default settings (seed {args.seed}, "
                      f"change 0.55, Best, keep face + colours, 2x) — {pipe.e.info.label()}", fill=(235, 235, 240), font=tfont)
    for i, (label, img) in enumerate(tiles):
        x = pad + (i % cols) * (tw + pad); y = 50 + pad + (i // cols) * (th + lab + pad)
        sheet.paste(Image.fromarray(img).resize((tw, th), Image.LANCZOS), (x, y + lab))
        d.text((x + 4, y + 6), label, fill=(235, 235, 240), font=font)
    out.parent.mkdir(parents=True, exist_ok=True); sheet.save(out, optimize=True)
    for i, (label, img) in enumerate(tiles[1:]):
        Image.fromarray(img).save(out.parent / f"sample_{i + 1}.jpg", quality=92)
    (out.parent / "samples.json").write_text(json.dumps(meta, indent=1))
    log.info("samples -> %s (%d bytes)", out, out.stat().st_size)
    return 0


def bench(args) -> int:
    from .engine import Engine
    store = _store(args); _install(store, args.import_dir)
    e = Engine(store, args.device, gpu_half=not args.no_half)
    rep = {"versions": versions(), "device": args.device}
    for size, (W, H) in (("standard", (576, 384)), ("large", (896, 576))):
        h, w = H // 8, W // 8
        feed = {"sample": np.random.randn(1, 4, h, w).astype(np.float32), "timestep": np.array([500], np.int64),
                "encoder_hidden_states": np.random.randn(1, 77, 768).astype(np.float32),
                "timestep_cond": np.random.randn(1, 256).astype(np.float32)}
        for k, (c, f) in enumerate([(320, 1), (640, 2), (1280, 4), (1280, 8)]):
            feed[f"adapter_res{k}"] = np.zeros((1, c, h // f, w // f), np.float32)
        e.run("unet", feed); ts = []
        for _ in range(args.bench_iters):
            t = time.time(); e.run("unet", feed); ts.append(time.time() - t)
        rep[size] = {"size": [W, H], "sec_per_unet_eval": round(float(np.median(ts)), 3)}
        log.info("unet %s %dx%d: %.3f s/eval", size, W, H, np.median(ts))
    rep["label"] = e.info.label(); rep["unet_variant"] = e.unet_variant; rep["per_model"] = e.info.per_model
    print(json.dumps(rep, indent=1, default=str))
    if args.out: Path(args.out).write_text(json.dumps(rep, indent=1, default=str))
    return 0


def derive_fp16(args) -> int:
    from . import models as M
    store = _store(args); _install(store, args.import_dir)
    t = time.time(); last = [0.0]

    def cb(f):
        if time.time() - last[0] > 3: last[0] = time.time(); log.info("fp16: %.0f%%", f * 100)
    M.derive_unet_fp16(store, cb)
    ok = M.fp16_installed(store)
    log.info("FP16 %s (%.0f s)", "OK (SHA-256 verified)" if ok else "FAILED", time.time() - t)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="AIImageCreate", description="AI Image Create for Windows " + __version__)
    ap.add_argument("--models", help="model folder (default %%LOCALAPPDATA%%\\AIImageCreate\\models)")
    ap.add_argument("--import", dest="import_dir", help="install models from a folder with the release files")
    ap.add_argument("--download", action="store_true", help="download + install the models, then exit")
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--dml-smoke", action="store_true")
    ap.add_argument("--photo"); ap.add_argument("--out")
    ap.add_argument("--generate", nargs=2, metavar=("IN", "OUT")); ap.add_argument("--prompt")
    ap.add_argument("--strength", type=float, default=0.55); ap.add_argument("--quality", choices=["best", "fast"], default="best")
    ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--size", choices=["standard", "large"], default="standard")
    ap.add_argument("--no-face", action="store_true"); ap.add_argument("--no-colors", action="store_true")
    ap.add_argument("--no-upscale", action="store_true")
    ap.add_argument("--strictness", choices=["relaxed", "standard"], default="relaxed")
    ap.add_argument("--device", choices=["auto", "dml", "cpu"], default="auto"); ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--parity-cases", nargs=2, metavar=("TESTDATA", "OUT"))
    ap.add_argument("--samples", nargs=2, metavar=("PHOTO", "OUT")); ap.add_argument("--prompts", nargs="*")
    ap.add_argument("--bench", action="store_true"); ap.add_argument("--bench-iters", type=int, default=3)
    ap.add_argument("--derive-fp16", action="store_true")
    ap.add_argument("--screenshots", metavar="DIR"); ap.add_argument("--gui-smoke", type=float, metavar="SECONDS")
    args = ap.parse_args(argv)
    gui = not any([args.selftest, args.generate, args.parity_cases, args.samples, args.bench, args.derive_fp16,
                   args.download, args.screenshots])
    _setup_logging(verbose=not gui)
    try:
        if gui:
            from .gui.app import run
            return run(args)
        log.info("AI Image Create %s %s", __version__, json.dumps(versions()))
        if args.screenshots:
            from .gui.app import screenshots
            return screenshots(args)
        if args.selftest:
            if not args.photo: ap.error("--selftest needs --photo (a picture of a private person)")
            return selftest(args)
        if args.generate: return generate(args)
        if args.parity_cases: return parity_cases(args)
        if args.samples: return samples(args)
        if args.bench: return bench(args)
        if args.derive_fp16: return derive_fp16(args)
        if args.download:
            _install(_store(args), args.import_dir); log.info("models installed"); return 0
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        log.error("fatal: %s", traceback.format_exc())
        return 1
    return 0


def entry():
    code = main()
    try:
        sys.stdout and sys.stdout.flush(); sys.stderr and sys.stderr.flush()
    finally:
        os._exit(code or 0)
