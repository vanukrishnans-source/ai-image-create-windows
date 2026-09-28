"""Re-run the Android parity cases (reference/parity_dump.py) through the Windows app pipeline and compare with the
reference outputs in ../parity (or a folder given with --ref). usage: parity_local.py MODELS_DIR WORK_DIR [--ref DIR]"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from aic import models as M, sdcore as C
from aic.engine import Engine
from aic.pipeline import Pipeline, load_photo
from aic.face import detect_ovals

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)); return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)

models, work = sys.argv[1], sys.argv[2]
ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else "/workspace/ai-image-app/parity"
refj = json.load(open(os.path.join(ref, "reference.json")))
eng = Engine(M.ModelStore(models), "cpu"); pipe = Pipeline(eng)
tok_ok = all(pipe.tok.encode(t) == ids for t, ids in refj["tokens"])
out = {"tokens_identical": tok_ok, "cases": []}
for case in refj["cases"]:
    key = case["key"]
    dec = np.asarray(Image.open(os.path.join(ref, f"{key}_decoded.png")).convert("RGB"))
    W, H = C.target_size(dec.shape[1], dec.shape[0]); rgb = C.resize_cover(dec, W, H)
    rin = np.asarray(Image.open(os.path.join(ref, f"{key}_input.png")).convert("RGB"))
    ov = detect_ovals(rgb) if case["keepFace"] else []
    rov = json.load(open(os.path.join(ref, f"{key}_ovals.json")))
    ovd = max([float(np.abs(np.array(o)[:, 0] - np.array(r[0])).max()) for o, r in zip(ov, rov)] + [0.0]) if len(ov) == len(rov) else None
    m = C.mask_from_ovals(H, W, ov) if ov else None
    kw = C.preset_kwargs(case["preset"], case["quality"] == "BEST")
    t0 = time.time()
    r = pipe.run_core(rgb, C.build_prompt(case["user"], case["preset"]), seed=case["seed"], face_mask=m, faces=len(ov),
                      upscale=case["upscale"], **kw)
    base_ref = np.asarray(Image.open(os.path.join(ref, f"{key}_base.png")).convert("RGB"))
    row = dict(key=key, preset=case["preset"], quality=case["quality"], seed=case["seed"], size=[W, H],
               input_identical=bool((rgb == rin).all()), faces=len(ov), faces_ref=len(rov), oval_max_dx_px=ovd,
               nsfw=r.nsfw, nsfw_ref=case["nsfw"], blocked=r.blocked, seconds=round(time.time() - t0, 1))
    if r.base is not None:
        row["psnr_base_db"] = round(psnr(r.base, base_ref), 2); row["maxdiff_base"] = int(np.abs(r.base.astype(int) - base_ref.astype(int)).max())
        Image.fromarray(r.base).save(os.path.join(work, f"{key}_base_app.png"))
        if case["upscale"]:
            fin_ref = np.asarray(Image.open(os.path.join(ref, f"{key}_final.png")).convert("RGB"))
            row["psnr_final_db"] = round(psnr(r.image, fin_ref), 2)
    print(json.dumps(row), flush=True); out["cases"].append(row)
json.dump(out, open(os.path.join(work, "parity_local.json"), "w"), indent=1)
