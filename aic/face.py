"""'Keep face likeness': MediaPipe FaceLandmarker face ovals (same model file + options as reference/face_mask.py and
the Android app). One landmarker per thread; the model is read into memory so non-ASCII Windows paths work."""
from __future__ import annotations

import threading

import numpy as np

from .resources_path import res
from .sdcore import OVAL, mask_from_ovals

_tls = threading.local()


def _landmarker():
    lm = getattr(_tls, "lm", None)
    if lm is None:
        from . import _install_matplotlib_stub
        _install_matplotlib_stub()
        from mediapipe.tasks.python import vision, BaseOptions
        buf = res("face_landmarker.task").read_bytes()
        lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=buf), running_mode=vision.RunningMode.IMAGE, num_faces=6,
            min_face_detection_confidence=0.4, min_face_presence_confidence=0.4))
        _tls.lm = lm
    return lm


def detect_ovals(rgb_u8):
    """List of 36x2 oval point arrays (pixels) for every face MediaPipe finds."""
    import mediapipe as mp
    H, W, _ = rgb_u8.shape
    r = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_u8)))
    return [np.array([[f[i].x * W, f[i].y * H] for i in OVAL], np.float64) for f in r.face_landmarks]


def face_mask(rgb_u8):
    ov = detect_ovals(rgb_u8); H, W, _ = rgb_u8.shape
    return (mask_from_ovals(H, W, ov) if ov else None), ov
