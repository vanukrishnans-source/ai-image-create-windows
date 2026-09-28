"""Compare the packaged Windows app outputs with the reference outputs. usage: compare_parity.py APP_DIR REF_DIR OUT_JSON"""
import sys, os, json
import numpy as np
from PIL import Image

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)

app, ref, outp = sys.argv[1:4]
ra = json.load(open(os.path.join(app, "app_cases.json"))); rr = json.load(open(os.path.join(ref, "reference.json")))
rows = []; ok = True
for ca, cr in zip(ra["cases"], rr["cases"]):
    assert ca["key"] == cr["key"]
    row = dict(key=ca["key"], size=ca["size"], faces_app=ca["faces"], faces_ref=cr["faces"], nsfw_app=ca["nsfw"], nsfw_ref=cr["nsfw"])
    for kind in ("base", "final"):
        pa = os.path.join(app, f"{ca['key']}_{kind}.png"); pr = os.path.join(ref, f"{cr['key']}_{kind}.png")
        if os.path.exists(pa) and os.path.exists(pr):
            a = np.asarray(Image.open(pa).convert("RGB")); b = np.asarray(Image.open(pr).convert("RGB"))
            p = psnr(a, b); row[f"psnr_{kind}_db"] = "inf" if p == float("inf") else round(p, 2)
            row[f"maxdiff_{kind}"] = int(np.abs(a.astype(int) - b.astype(int)).max())
            if p <= 60: ok = False
    rows.append(row); print(json.dumps(row))
res = {"app_onnxruntime": ra.get("onnxruntime"), "ref_onnxruntime": rr.get("onnxruntime"), "pass_psnr_gt_60db": ok, "cases": rows}
json.dump(res, open(outp, "w"), indent=1)
print("PARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
