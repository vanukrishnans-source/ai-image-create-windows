"""CI helper: download the models-v1 release assets (7 .wts + graphs-v1.zip) into a cache folder, SHA-256 checked,
using the same URLs / checksums as the app. usage: prefetch_release.py DIR"""
import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
from aic import models as M

d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
st = M.ModelStore(d / "_tmpstore")
items = [(M.GRAPHS_ASSET, M.GRAPHS_BYTES, M.GRAPHS_SHA)] + [(s.asset, s.wts_bytes, s.sha256) for s in M.ALL]
for asset, size, sha in items:
    p = d / asset
    if p.is_file() and p.stat().st_size == size and M.sha256_file(p) == sha:
        print("cached", asset); continue
    print("download", asset, size, flush=True)
    st._fetch_file(asset, size, sha, p, 0, M.DOWNLOAD_BYTES, threading.Event(), lambda pr: None)
print("all release assets present + verified")
