"""ONNX Runtime sessions: DirectML (Radeon 780M) for the heavy models, CPU for the tiny ones, automatic CPU fallback.

* GPU_MODELS (UNet, VAE decoder, upscaler) use the DirectML EP when available; if a DirectML session can't be created
  or a run fails (e.g. out of GPU memory) that model silently falls back to the CPU EP and the badge says why.
* The text encoder, TAESD encoder, T2I-Adapter and the NSFW classifier always run on CPU: they take < 0.1 s each, and
  keeping the safety classifier on CPU gives exactly the same scores as the Android app.
* Sessions stay open between pictures (24 GB RAM), so Regenerate is fast.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import models as M

log = logging.getLogger("aic")
GPU_MODELS = ("unet", "vae_decoder", "upscaler")
SPECS = {s.name: s for s in M.ALL}


@dataclass
class DeviceInfo:
    requested: str = "auto"
    active: str = "CPU"          # "DirectML" or "CPU" (for the GPU models)
    adapter: str = ""
    fallback_reason: str = ""
    per_model: dict = field(default_factory=dict)

    def label(self):
        if self.active == "DirectML":
            return "GPU · DirectML" + (f" · {self.adapter}" if self.adapter else "")
        return "CPU" + (" (GPU unavailable)" if self.fallback_reason else "")


def gpu_adapter_name() -> str:
    if os.name != "nt": return ""
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join '; '"],
                             capture_output=True, text=True, timeout=15, creationflags=0x08000000).stdout.strip()
        return out
    except Exception:  # noqa: BLE001
        return ""


class Engine:
    def __init__(self, store: M.ModelStore, device: str = "auto", threads: int = 0, gpu_half: bool = True,
                 on_half_failed=None):
        import onnxruntime as ort
        self.ort = ort; self.store = store; self.device = device
        self.gpu_half = gpu_half; self.on_half_failed = on_half_failed
        self.unet_variant = "fp32"   # "fp16" when the DirectML half-precision UNet is in use
        self.threads = threads or int(os.environ.get("ORT_THREADS", "0") or 0)
        self.lock = threading.RLock(); self.sessions = {}; self.timing = {}
        self.info = DeviceInfo(requested=device)
        self.dml_available = "DmlExecutionProvider" in ort.get_available_providers()
        if device in ("auto", "dml") and not self.dml_available:
            self.info.fallback_reason = "DirectML is not available in this onnxruntime build"
        self.info.active = "DirectML" if device in ("auto", "dml") and self.dml_available else "CPU"
        if self.info.active == "DirectML":
            self.info.adapter = gpu_adapter_name().split(";")[0].strip()

    def _options(self, dml: bool):
        so = self.ort.SessionOptions(); so.log_severity_level = 3
        so.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if dml:
            so.enable_mem_pattern = False
            so.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            so.enable_cpu_mem_arena = False; so.inter_op_num_threads = 1
            if self.threads: so.intra_op_num_threads = self.threads
        return so

    def _fallback(self, name, e):
        if self.device == "dml": raise e
        log.warning("DirectML failed for %s: %s -> CPU", name, e)
        self.info.fallback_reason = f"{type(e).__name__}: {str(e)[:200]}"
        if all(self.info.per_model.get(m) != "DirectML" for m in GPU_MODELS if m != name):
            self.info.active = "CPU"

    def _open(self, name, force_cpu=False):
        s = SPECS[name]
        if not self.store.is_installed(s): raise FileNotFoundError(f"model not installed: {name}")
        path = str(self.store.onnx(name)); t0 = time.time()
        if not force_cpu and name in GPU_MODELS and self.info.active == "DirectML":
            if name == "unet" and self.gpu_half and M.fp16_installed(self.store):
                try:
                    sess = self.ort.InferenceSession(str(self.store.dir / "unet_fp16.onnx"), self._options(True),
                                                     providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
                    used = sess.get_providers()
                    if not used or used[0] != "DmlExecutionProvider":
                        raise RuntimeError(f"DirectML provider not used (got {used})")
                    self.info.per_model[name] = "DirectML"; self.unet_variant = "fp16"
                    self.timing["load_" + name] = time.time() - t0
                    return sess
                except Exception as e:  # noqa: BLE001 — try the fp32 graph on DirectML next
                    log.warning("fp16 UNet on DirectML failed: %s -> fp32", e)
                    self._half_failed(str(e))
            try:
                sess = self.ort.InferenceSession(path, self._options(True),
                                                 providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
                used = sess.get_providers()
                if not used or used[0] != "DmlExecutionProvider":
                    raise RuntimeError(f"DirectML provider not used (got {used})")
                self.info.per_model[name] = "DirectML"
                if name == "unet": self.unet_variant = "fp32"
                self.timing["load_" + name] = time.time() - t0
                return sess
            except Exception as e:  # noqa: BLE001
                self._fallback(name, e)
        if name == "unet": self.unet_variant = "fp32"
        sess = self.ort.InferenceSession(path, self._options(False), providers=["CPUExecutionProvider"])
        self.info.per_model[name] = "CPU"; self.timing["load_" + name] = time.time() - t0
        return sess

    def _half_failed(self, why):
        self.gpu_half = False
        if self.on_half_failed:
            try: self.on_half_failed(why)
            except Exception: pass  # noqa: BLE001

    def session(self, name):
        with self.lock:
            if name not in self.sessions: self.sessions[name] = self._open(name)
            return self.sessions[name]

    def run(self, name, feeds):
        sess = self.session(name)
        with self.lock:
            t0 = time.time()
            try:
                out = sess.run(None, feeds)
            except Exception as e:  # noqa: BLE001 — a DirectML run failure (e.g. out of GPU memory) -> CPU, retry once
                if self.info.per_model.get(name) != "DirectML": raise
                self._fallback(name, e)
                self.sessions[name] = sess = self._open(name, force_cpu=True)
                out = sess.run(None, feeds)
            if name == "unet" and self.unet_variant == "fp16" and not np.isfinite(out[0]).all():
                # half precision overflowed on this GPU/driver: switch to the fp32 UNet for good and redo this step
                log.warning("fp16 UNet produced NaN/Inf -> fp32")
                self._half_failed("NaN/Inf in fp16 UNet output")
                self.sessions[name] = sess = self._open(name)
                out = sess.run(None, feeds)
            self.timing[name] = self.timing.get(name, 0.0) + time.time() - t0
            return out

    def warm_up(self, names=("text_encoder", "taesd_encoder", "adapter_canny", "unet", "vae_decoder", "safety", "upscaler"), cb=None):
        for i, n in enumerate(names):
            if cb: cb(n, i, len(names))
            self.session(n)

    def close(self):
        with self.lock: self.sessions.clear()
