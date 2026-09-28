"""Run testdata/parity_cases.json through the UNMODIFIED Python reference pipeline (reference/sd_ref.py — the same code
the Android app was verified against). Runs in a separate venv (onnxruntime 1.22.1 CPU, no app code imported).
usage: python ref_cases.py MODELS_DIR TESTDATA_DIR OUT_DIR FACE_LANDMARKER_TASK"""
import sys, os, json, time, types
HERE = os.path.dirname(os.path.abspath(__file__)); REF = os.path.join(os.path.dirname(HERE), "reference")
sys.modules.setdefault("resource", types.ModuleType("resource"))   # sd_ref imports the Unix-only 'resource' module (CLI only)
sys.path.insert(0, REF)
import numpy as np
from PIL import Image, ImageOps
import sd_ref as R
import face_mask as FM

models, testdata, out, task = sys.argv[1:5]
FM.MODEL = task
os.makedirs(out, exist_ok=True)
pipe = R.Pipeline([models], threads=os.cpu_count() or 4, keep_open=True)
res = {"onnxruntime": R.ort.__version__, "cases": []}
for c in json.load(open(os.path.join(testdata, "parity_cases.json"))):
    t0 = time.time()
    im = ImageOps.exif_transpose(Image.open(os.path.join(testdata, c["photo"]))).convert("RGB"); a = np.asarray(im)
    W, H = R.target_size(a.shape[1], a.shape[0], {"standard": 512, "large": 768}[c["size"]]); rgb = R.resize_cover(a, W, H)
    kw = R.preset_kwargs("none", c["quality"] == "best"); kw["strength"] = c["strength"]
    if not c["keep_colors"]: kw["keep_luma"] = 0.0; kw["keep_chroma"] = 0.0
    m = None; nf = 0
    if c["keep_face"]:
        ov = FM.detect_ovals(rgb); nf = len(ov)
        if ov: m = FM.mask_from_ovals(H, W, ov)
    img, info = pipe.generate(rgb, R.build_prompt(c["prompt"], "none"), seed=c["seed"], face_mask=m, **kw)
    Image.fromarray(img).save(os.path.join(out, c["key"] + "_base.png"))
    if c["upscale"]: Image.fromarray(R.upscale2x(pipe.m, img)).save(os.path.join(out, c["key"] + "_final.png"))
    row = dict(key=c["key"], size=[W, H], faces=nf, nsfw=info["nsfw"], timesteps=info["timesteps"], seconds=round(time.time() - t0, 1))
    print(json.dumps(row), flush=True); res["cases"].append(row)
json.dump(res, open(os.path.join(out, "reference.json"), "w"), indent=1)
