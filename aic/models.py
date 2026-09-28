"""Model store — Python port of the Android WeightStore: the same GitHub release assets (models-v1: 7 packed .wts
blobs + graphs-v1.zip), resumable HTTP Range downloads with a mirror list, SHA-256 verification, then the same
fp16/int8 -> fp32 rebuild (reference/sd_ref.py rebuild_fp32) into <name>.fp32.bin next to <name>.onnx, a '<name>.ok'
marker holding the verified SHA-256, and the blob deleted afterwards."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIRRORS = [
    "https://github.com/vanukrishnans-source/ai-image-create-models/releases/download/models-v1/",
]
REPO_URL = "https://github.com/vanukrishnans-source/ai-image-create-models"
USER_AGENT = "AIImageCreate/1.0 (Windows)"


@dataclass(frozen=True)
class WeightSpec:
    name: str
    label: str
    wts_bytes: int
    sha256: str
    fp32_bytes: int

    @property
    def asset(self): return f"{self.name}.wts"


TAESD = WeightSpec("taesd_encoder", "Image encoder (TAESD)", 2_445_056, "b66728d9892b977253d89c29e7e21a812d15d72f811578a62d33cdea50edf70d", 5_399_552)
UPSCALER = WeightSpec("upscaler", "2× upscaler (Real-ESRGAN)", 2_426_496, "8f096cbb62605a1b834604805c742897b8d8d72e468145f8ff0c646d62806ff8", 5_914_880)
SAFETY = WeightSpec("safety", "Safety filter (NSFW classifier)", 86_259_460, "836b8faf51ac81d1815f4e4c740a7d3912eafbd2671ca3f6f55e39678fa619cf", 344_752_128)
VAE = WeightSpec("vae_decoder", "Image decoder (SD VAE)", 98_980_352, "a9c4c45c903ba936b9a5d8ba01d8afa055b66a3e314f318fb29de48544c56ce1", 199_410_176)
ADAPTER = WeightSpec("adapter_canny", "Structure guide (T2I-Adapter)", 154_001_280, "6ee10a57bb6dbd430ebdaf85e2f988d5fc09790145cdbfd4f0882401d21bf92c", 308_237_312)
TEXT = WeightSpec("text_encoder", "Text encoder (CLIP)", 246_120_960, "bc3213d7add3434c7ea793f722c67a1d6ce787277725d7ee29631611496352ce", 493_764_608)
UNET = WeightSpec("unet", "Image generator (LCM Dreamshaper v7)", 861_521_280, "1ca5403f209817566e8ddb5e77c9506fcc3b5e01778d48c0e818dfe633545369", 3_444_035_328)
ALL = [TAESD, UPSCALER, SAFETY, VAE, ADAPTER, TEXT, UNET]

GRAPHS_ASSET = "graphs-v1.zip"
GRAPHS_BYTES = 114_963
GRAPHS_SHA = "801b240ce1d166ffc0c2424c651cf11c8fa69d333918de64444b4084381a0b3b"

DOWNLOAD_BYTES = sum(s.wts_bytes for s in ALL) + GRAPHS_BYTES      # 1,451,754,884
INSTALLED_BYTES = sum(s.fp32_bytes for s in ALL)                    # ~4.80 GB


class ChecksumError(IOError): pass
class NotEnoughStorage(IOError): pass
class Cancelled(Exception): pass


def default_dir() -> Path:
    env = os.environ.get("AIC_MODELS")
    if env: return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "AIImageCreate" / "models"


def sha256_file(p: Path, cancel=None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            if cancel is not None and cancel.is_set(): raise Cancelled()
            b = f.read(4 << 20)
            if not b: break
            h.update(b)
    return h.hexdigest()


def _ssl_ctx():
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except Exception:  # noqa: BLE001
        pass
    return ctx


def rebuild_fp32(manifest: dict, blob_path: Path, out_path: Path, cancel=None, progress=None):
    """Same maths as reference/sd_ref.py rebuild_fp32 (f16 -> fp32, q8 per-channel -> fp32), 16 KiB-aligned offsets."""
    tmp = Path(str(out_path) + ".part")
    blob = np.memmap(blob_path, dtype=np.uint8, mode="r")
    with open(tmp, "wb") as f:
        pos = 0
        for t in manifest["tensors"]:
            if cancel is not None and cancel.is_set(): raise Cancelled()
            pad = t["fp32_offset"] - pos
            if pad < 0: raise IOError("tensor table out of order")
            if pad: f.write(b"\0" * pad)
            pos = t["fp32_offset"]; n = t["count"]; o = t["w_offset"]
            if t["kind"] == "f16":
                a = np.frombuffer(blob[o:o + 2 * n].tobytes(), np.float16).astype(np.float32)
            elif t["kind"] == "q8":
                ns = t["nscale"]; sc = np.frombuffer(blob[o:o + 4 * ns].tobytes(), np.float32)
                q = np.frombuffer(blob[o + 4 * ns:o + 4 * ns + n].tobytes(), np.int8).astype(np.float32)
                shp = t["shape"]
                if t["axis"] == 1: a = (q.reshape(shp[0], -1) * sc[None, :]).reshape(-1)
                else: a = (q.reshape(shp[0], -1) * sc[:, None]).reshape(-1)
            else:
                raise IOError(f"unknown tensor kind {t['kind']}")
            f.write(a.astype(np.float32).tobytes()); pos += 4 * n
            if progress: progress(pos / manifest["fp32_bytes"])
        if manifest["fp32_bytes"] > pos: f.write(b"\0" * (manifest["fp32_bytes"] - pos))
    del blob
    if tmp.stat().st_size != manifest["fp32_bytes"]:
        raise IOError(f"{manifest['name']}: rebuilt {tmp.stat().st_size} != {manifest['fp32_bytes']}")
    if out_path.exists(): out_path.unlink()
    os.replace(tmp, out_path)


@dataclass
class Progress:
    stage: str          # "download" | "verify" | "unpack" | "done"
    done: int           # download bytes over all files
    total: int
    file: str
    bps: float = 0.0
    fraction: float = 0.0


class ModelStore:
    def __init__(self, d: Path | None = None, mirrors=None):
        self.dir = Path(d) if d else default_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mirrors = list(mirrors or MIRRORS)

    # ---- state ----
    def onnx(self, name): return self.dir / f"{name}.onnx"
    def fp32(self, s: WeightSpec): return self.dir / f"{s.name}.fp32.bin"
    def manifest_path(self, name): return self.dir / f"{name}.json"
    def _blob(self, s): return self.dir / s.asset
    def _part(self, s): return self.dir / (s.asset + ".part")
    def _marker(self, s): return self.dir / f"{s.name}.ok"

    def graphs_ok(self):
        return all(self.onnx(s.name).is_file() and self.manifest_path(s.name).is_file() for s in ALL)

    def is_installed(self, s: WeightSpec) -> bool:
        f = self.fp32(s); m = self._marker(s)
        try:
            return f.is_file() and f.stat().st_size == s.fp32_bytes and self.onnx(s.name).is_file() and \
                m.is_file() and m.read_text().strip() == s.sha256
        except OSError:
            return False

    def all_installed(self): return all(self.is_installed(s) for s in ALL)

    def missing(self): return [s for s in ALL if not self.is_installed(s)]

    def bytes_present(self):
        tot = GRAPHS_BYTES if self.graphs_ok() else 0
        for s in ALL:
            if self.is_installed(s): tot += s.wts_bytes
            elif self._blob(s).is_file(): tot += min(self._blob(s).stat().st_size, s.wts_bytes)
            elif self._part(s).is_file(): tot += min(self._part(s).stat().st_size, s.wts_bytes)
        return tot

    def free_space_needed(self):
        miss = self.missing()
        if not miss: return 0
        have = sum(s.wts_bytes if self._blob(s).is_file() else (self._part(s).stat().st_size if self._part(s).is_file() else 0) for s in miss)
        return sum(s.fp32_bytes for s in miss) + max(s.wts_bytes for s in miss) + 100_000_000 - have

    def free_space(self):
        try: return shutil.disk_usage(self.dir).free
        except OSError: return -1

    def clear(self):
        for s in ALL:
            for p in (self.fp32(s), self._blob(s), self._part(s), self._marker(s), self.onnx(s.name), self.manifest_path(s.name),
                      Path(str(self.fp32(s)) + ".part")):
                try: p.unlink()
                except OSError: pass
        for n in ("unet.fp16.bin", "unet.fp16.bin.part", "unet_fp16.onnx", "unet_fp16.ok"):
            try: (self.dir / n).unlink()
            except OSError: pass

    # ---- install ----
    def _install_graphs(self, zip_path: Path):
        if sha256_file(zip_path) != GRAPHS_SHA: raise ChecksumError(f"{GRAPHS_ASSET} failed the checksum check")
        with zipfile.ZipFile(zip_path) as z:
            for s in ALL:
                for ext in (".onnx", ".json"):
                    (self.dir / f"{s.name}{ext}").write_bytes(z.read(f"{s.name}{ext}"))

    def _unpack(self, s: WeightSpec, blob: Path, cancel, cb):
        man = json.loads(self.manifest_path(s.name).read_text(encoding="utf-8"))
        if man["wts_bytes"] != s.wts_bytes or man["fp32_bytes"] != s.fp32_bytes or man["wts_sha256"] != s.sha256:
            raise IOError(f"{s.name}: tensor table mismatch")
        rebuild_fp32(man, blob, self.fp32(s), cancel, cb)
        self._marker(s).write_text(s.sha256)

    def ensure(self, progress=None, cancel: threading.Event | None = None):
        """Blocking install of everything missing. Raises Cancelled (partial data kept), ChecksumError, NotEnoughStorage."""
        cancel = cancel or threading.Event(); progress = progress or (lambda p: None)
        miss = self.missing()
        if not miss and self.graphs_ok(): return
        need = self.free_space_needed(); free = self.free_space()
        if 0 < free < need:
            raise NotEnoughStorage(f"Not enough free disk space: setup needs about {need / 1e9:.1f} GB free on "
                                   f"{self.dir.anchor or self.dir}, only {free / 1e9:.1f} GB is available.")
        total = DOWNLOAD_BYTES
        if not self.graphs_ok():
            zp = self.dir / GRAPHS_ASSET
            self._fetch_file(GRAPHS_ASSET, GRAPHS_BYTES, GRAPHS_SHA, zp, 0, total, cancel, progress)
            self._install_graphs(zp); zp.unlink()
        base = GRAPHS_BYTES + sum(s.wts_bytes for s in ALL if self.is_installed(s))
        for s in miss:
            if not (self._blob(s).is_file() and self._blob(s).stat().st_size == s.wts_bytes):
                self._fetch_file(s.asset, s.wts_bytes, s.sha256, self._blob(s), base, total, cancel, progress)
            progress(Progress("unpack", base + s.wts_bytes, total, s.name))
            self._unpack(s, self._blob(s), cancel, lambda fr, s=s, b=base: progress(Progress("unpack", b + s.wts_bytes, total, s.name, fraction=fr)))
            try: self._blob(s).unlink()
            except OSError: pass
            base += s.wts_bytes
        progress(Progress("done", total, total, ""))

    def _fetch_file(self, asset, size, sha, dest: Path, base, total, cancel, progress):
        part = Path(str(dest) + ".part")
        last_err = None
        for attempt in range(3):
            for m in self.mirrors:
                try:
                    self._download(m + asset, size, part, base, total, cancel, progress, asset)
                    last_err = None; break
                except Cancelled: raise
                except (IOError, urllib.error.URLError, TimeoutError) as e: last_err = e
            if last_err is None: break
            if cancel.is_set(): raise Cancelled()
            time.sleep(2 * (attempt + 1))
        if last_err is not None: raise IOError(f"download of {asset} failed: {last_err}")
        if part.stat().st_size != size: raise IOError(f"{asset}: size {part.stat().st_size} != {size}")
        progress(Progress("verify", base + size, total, asset))
        got = sha256_file(part, cancel)
        if got != sha:
            part.unlink()
            raise ChecksumError(f"{asset} failed the checksum check (got {got[:12]}…). It was deleted — press Retry to download it again.")
        if dest.exists(): dest.unlink()
        os.replace(part, dest)

    def _download(self, url, size, part: Path, base, total, cancel, progress, label):
        have = part.stat().st_size if part.is_file() else 0
        if have > size: part.unlink(); have = 0
        if have == size: return
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
        if have: req.add_header("Range", f"bytes={have}-")
        try:
            r = urllib.request.urlopen(req, timeout=30, context=_ssl_ctx())
        except urllib.error.HTTPError as e:
            if e.code == 416: part.unlink(missing_ok=True)
            raise IOError(f"HTTP {e.code} for {label}") from e
        with r:
            code = r.status
            append = code == 206 and have > 0
            if code == 200: have = 0
            elif not append: raise IOError(f"HTTP {code} for {label}")
            with open(part, "ab" if append else "wb") as out:
                got = have; t0 = time.time(); start = have; last = 0.0
                while True:
                    if cancel.is_set(): raise Cancelled()
                    b = r.read(1 << 20)
                    if not b: break
                    out.write(b); got += len(b)
                    if got > size: raise IOError(f"{label}: server sent more data than expected")
                    now = time.time()
                    if now - last > 0.25:
                        last = now; dt = now - t0
                        progress(Progress("download", base + got, total, label, (got - start) / dt if dt > 0 else 0.0))

    def import_from(self, folder: Path, progress=None, cancel=None):
        """Install from a folder holding the release files (e.g. copied from another PC): SHA-checked, then unpacked."""
        folder = Path(folder); cancel = cancel or threading.Event(); progress = progress or (lambda p: None)
        found = []
        zp = folder / GRAPHS_ASSET
        if not self.graphs_ok():
            if not zp.is_file(): raise IOError(f"{GRAPHS_ASSET} not found in {folder}")
            self._install_graphs(zp)
        base = GRAPHS_BYTES
        for s in ALL:
            if self.is_installed(s): base += s.wts_bytes; continue
            p = folder / s.asset
            if not p.is_file() or p.stat().st_size != s.wts_bytes: continue
            progress(Progress("verify", base, DOWNLOAD_BYTES, s.asset))
            if sha256_file(p, cancel) != s.sha256: raise ChecksumError(f"{p.name} failed the checksum check")
            self._unpack(s, p, cancel, lambda fr, s=s, b=base: progress(Progress("unpack", b + s.wts_bytes, DOWNLOAD_BYTES, s.name, fraction=fr)))
            base += s.wts_bytes; found.append(s.name)
        return found


# ---------------------------------------------------------------- DirectML fp16 UNet (derived locally, no download)
FP16_BYTES = 1_719_289_600


def _to_f16(a):
    """Exactly onnxruntime.transformers.float16.convert_np_to_float16 (clamp to the fp16 range, keep sign/finiteness)."""
    a = np.where(np.logical_and(0 < a, a < 5.96e-08), 5.96e-08, a)
    a = np.where(np.logical_and(-5.96e-08 < a, a < 0), -5.96e-08, a)
    a = np.where(np.logical_and(65504.0 < a, a < float("inf")), 65504.0, a)
    a = np.where(np.logical_and(float("-inf") < a, a < -65504.0), -65504.0, a)
    return np.float16(a)


def fp16_installed(store: "ModelStore") -> bool:
    from .resources_path import res
    m = store.dir / "unet_fp16.ok"; b = store.dir / "unet.fp16.bin"
    try:
        man = json.loads(res("dml", "unet_fp16.json").read_text())
        return b.is_file() and b.stat().st_size == man["bytes"] and m.is_file() and m.read_text().strip() == man["sha256"] \
            and (store.dir / "unet_fp16.onnx").is_file()
    except (OSError, ValueError):
        return False


def derive_unet_fp16(store: "ModelStore", progress=None, cancel=None):
    """Build unet.fp16.bin for the DirectML fast path by casting the installed fp32 UNet weights to fp16 (the same cast
    the fp16 graph was made with), then check it against the SHA-256 recorded when the graph was built."""
    from .resources_path import res
    if not store.is_installed(UNET): raise IOError("UNet not installed")
    man = json.loads(res("dml", "unet_fp16.json").read_text())
    src = {t["name"]: t for t in json.loads(store.manifest_path("unet").read_text())["tensors"]}
    need = man["bytes"] + 50_000_000; free = store.free_space()
    if 0 < free < need: raise NotEnoughStorage(f"The GPU fast path needs {need / 1e9:.1f} GB free disk space.")
    out = store.dir / "unet.fp16.bin"; tmp = Path(str(out) + ".part")
    f32 = np.memmap(store.fp32(UNET), dtype=np.float32, mode="r")
    h = hashlib.sha256()
    with open(tmp, "wb") as f:
        pos = 0
        for i, t in enumerate(man["tensors"]):
            if cancel is not None and cancel.is_set(): raise Cancelled()
            if t["offset"] > pos:
                pad = b"\0" * (t["offset"] - pos); f.write(pad); h.update(pad); pos = t["offset"]
            s = src[t["name"]]; o = s["fp32_offset"] // 4
            b = _to_f16(np.asarray(f32[o:o + s["count"]])).tobytes()
            if len(b) != t["length"]: raise IOError(f"fp16 length mismatch for {t['name']}")
            f.write(b); h.update(b); pos += len(b)
            if progress: progress(pos / man["bytes"])
        if man["bytes"] > pos:
            pad = b"\0" * (man["bytes"] - pos); f.write(pad); h.update(pad)
    del f32
    if h.hexdigest() != man["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ChecksumError("GPU fast-path weights did not match the expected checksum")
    os.replace(tmp, out)
    shutil.copyfile(res("dml", "unet_fp16.onnx"), store.dir / "unet_fp16.onnx")
    (store.dir / "unet_fp16.ok").write_text(man["sha256"])
