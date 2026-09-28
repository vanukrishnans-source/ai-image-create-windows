"""AI Image Create — Qt (PySide6) GUI for Windows, touch-first for the ROG Ally X (7" 1080p, 150 % scaling).

"Prompt alone": attach a photo, type what you want, pick how much to change, Create. Everything else lives in a
collapsed Options panel. Pages: Setup (one-time model download) · Main · Progress · Result · Blocked.
The NSFW safety filter is always on — there is no option for it except Relaxed/Standard strictness.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSettings, QStandardPaths, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QKeyEvent, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__
from .. import models as M
from ..pipeline import Cancelled as GenCancelled, GenParams, GenResult, load_photo, new_seed, prepare
from .theme import QSS

log = logging.getLogger("aic")
PAGE_SETUP, PAGE_MAIN, PAGE_PROGRESS, PAGE_RESULT, PAGE_BLOCKED = range(5)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}
STRICT = {"relaxed": "Relaxed (default)", "standard": "Standard"}


def qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()


def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    a = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
    return a.copy()


def pix(rgb, w, h):
    """RGB ndarray -> QPixmap fitted into w x h logical px (rendered at the device pixel ratio)."""
    dpr = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    p = QPixmap.fromImage(qimage(rgb)).scaled(int(w * dpr), int(h * dpr), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
    p.setDevicePixelRatio(dpr)
    return p


def pictures_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation) or str(Path.home() / "Pictures")
    return Path(base) / "AIImageCreate"


class Worker(QThread):
    progressed = Signal(object)
    done = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn, parent=None):
        super().__init__(parent); self.fn = fn; self.cancel = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self.cancel, self.progressed.emit))
        except (M.Cancelled, GenCancelled):
            self.failed.emit("cancelled", "")
        except Exception as e:  # noqa: BLE001
            if self.cancel.is_set(): self.failed.emit("cancelled", "")
            else: self.failed.emit(f"{e}", traceback.format_exc())


def button(text, kind=None, min_w=0, tip=None):
    b = QPushButton(text)
    if kind: b.setObjectName(kind)
    if min_w: b.setMinimumWidth(min_w)
    if tip: b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def label(text="", kind=None, wrap=False):
    l = QLabel(text)
    if kind: l.setObjectName(kind)
    l.setWordWrap(wrap)
    return l


def card():
    f = QFrame(); f.setObjectName("card"); return f


class Segmented(QWidget):
    changed = Signal(object)

    def __init__(self, items):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(8)
        self.group = QButtonGroup(self); self.group.setExclusive(True); self.buttons = {}
        for text, value in items:
            b = button(text, "seg"); b.setCheckable(True); b.setMinimumHeight(50)
            self.group.addButton(b); lay.addWidget(b, 1); self.buttons[value] = b
            b.clicked.connect(lambda _=False, v=value: self.changed.emit(v))

    def set(self, value):
        if value in self.buttons: self.buttons[value].setChecked(True)

    def value(self):
        for v, b in self.buttons.items():
            if b.isChecked(): return v


class PhotoSlot(QFrame):
    """Tap to open, drop a file, or paste. Shows the photo exactly as the AI will see it (cropped to the output shape)."""
    clicked = Signal()
    dropped = Signal(object)

    def __init__(self):
        super().__init__(); self.setObjectName("card"); self.setAcceptDrops(True)
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(8)
        top = QHBoxLayout(); top.addWidget(label("Your photo", "section")); top.addStretch(1)
        self.paste_btn = button("Paste", tip="Paste a picture from the clipboard (Ctrl+V)"); top.addWidget(self.paste_btn)
        self.open_btn = button("Open…", tip="Choose a photo (Ctrl+O)"); top.addWidget(self.open_btn)
        lay.addLayout(top)
        self.image = QLabel("Tap here to choose a photo\n\nor drag & drop it here\nor paste it (Ctrl+V)")
        self.image.setObjectName("slot"); self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(300, 300); self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.image, 1)
        self.info = label("", "hint", True); lay.addWidget(self.info)
        self.rgb = None
        self.open_btn.clicked.connect(self.clicked.emit)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.clicked.emit()

    def dragEnterEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() or md.hasImage(): e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and md.urls(): self.dropped.emit(md.urls()[0].toLocalFile())
        elif md.hasImage(): self.dropped.emit(QImage(md.imageData()))

    def set_image(self, rgb):
        self.rgb = rgb; self._render()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()

    def _render(self):
        if self.rgb is not None:
            self.image.setPixmap(pix(self.rgb, max(100, self.image.width() - 8), max(100, self.image.height() - 8)))


class BeforeAfter(QWidget):
    """Drag (or tap) anywhere to move the divider: left of it = original, right of it = result."""

    def __init__(self):
        super().__init__(); self.before = self.after = None; self.pos_frac = 0.5
        self.setMinimumSize(400, 300); self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.SizeHorCursor); self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_images(self, before_rgb, after_rgb):
        self.after = QImage(qimage(after_rgb))
        self.before = qimage(before_rgb).scaled(self.after.width(), self.after.height(), Qt.AspectRatioMode.IgnoreAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
        self.pos_frac = 0.5; self.update()

    def _target(self):
        if self.after is None: return QRectF()
        W, H = self.width(), self.height(); iw, ih = self.after.width(), self.after.height()
        s = min(W / iw, H / ih); w, h = iw * s, ih * s
        return QRectF((W - w) / 2, (H - h) / 2, w, h)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.after is None: return
        r = self._target(); x = r.left() + r.width() * self.pos_frac
        p.drawImage(r, self.after)
        sx = self.before.width() * self.pos_frac
        p.drawImage(QRectF(r.left(), r.top(), x - r.left(), r.height()), self.before, QRectF(0, 0, sx, self.before.height()))
        p.setPen(QPen(QColor("#ffffff"), 3)); p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.setBrush(QColor("#8b7bff")); p.setPen(QPen(QColor("#ffffff"), 3)); p.drawEllipse(QPointF(x, r.center().y()), 22, 22)
        p.setPen(QPen(QColor("#0c0a1f"), 3))
        cy = r.center().y()
        for d in (-1, 1):
            path = QPainterPath(); path.moveTo(x + d * 6, cy - 8); path.lineTo(x + d * 14, cy); path.lineTo(x + d * 6, cy + 8); p.drawPath(path)
        f = QFont(self.font()); f.setPointSizeF(max(9.0, f.pointSizeF())); f.setBold(True); p.setFont(f)
        for text, left in (("Before", True), ("After", False)):
            fm = p.fontMetrics(); tw = fm.horizontalAdvance(text) + 20; th = fm.height() + 10
            bx = r.left() + 10 if left else r.right() - tw - 10
            p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(0, 0, 0, 150)); p.drawRoundedRect(QRectF(bx, r.top() + 10, tw, th), 8, 8)
            p.setPen(QColor("#ffffff")); p.drawText(QRectF(bx, r.top() + 10, tw, th), Qt.AlignmentFlag.AlignCenter, text)

    def _move(self, x):
        r = self._target()
        if r.width() > 0: self.pos_frac = float(min(1.0, max(0.0, (x - r.left()) / r.width()))); self.update()

    def mousePressEvent(self, e): self._move(e.position().x())
    def mouseMoveEvent(self, e): self._move(e.position().x())

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.pos_frac = min(1.0, max(0.0, self.pos_frac + (0.05 if e.key() == Qt.Key.Key_Right else -0.05))); self.update()
        else: super().keyPressEvent(e)


class MainWindow(QMainWindow):
    def __init__(self, store: M.ModelStore, device="auto", settings: QSettings | None = None):
        super().__init__()
        self.store = store; self.qs = settings or QSettings("vanu", "AIImageCreate")
        self.device = device if device in ("cpu", "dml") else self._get("device", "auto", ("auto", "cpu"))
        self.pipe = None; self.engine = None; self.ready = threading.Event(); self.warm = None
        self.worker = None; self.photo = None; self.photo_name = ""; self.result: GenResult | None = None
        self.last_params: GenParams | None = None; self.saved_path = None; self.sec_per_eval = None
        self.setWindowTitle(f"AI Image Create {__version__}")
        central = QWidget(); v = QVBoxLayout(central); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        v.addWidget(self._topbar())
        self.stack = QStackedWidget(); v.addWidget(self.stack, 1)
        for page in (self._page_setup(), self._page_main(), self._page_progress(), self._page_result(), self._page_blocked()):
            self.stack.addWidget(page)
        self.setCentralWidget(central); self.setAcceptDrops(True)
        QShortcut(QKeySequence.StandardKey.Paste, self, activated=self._paste)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._pick_photo)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        self._load_options()
        if store.all_installed():
            self.go(PAGE_MAIN); self._start_warmup()
        else:
            self.go(PAGE_SETUP)
        self._set_chip()

    # ------------------------------------------------------------------ settings helpers
    def _get(self, key, default, allowed=None):
        v = self.qs.value(key, default)
        if isinstance(default, bool): v = str(v).lower() in ("1", "true", "yes")
        elif isinstance(default, int): v = int(v) if str(v).lstrip("-").isdigit() else default
        elif isinstance(default, float):
            try: v = float(v)
            except (TypeError, ValueError): v = default
        if allowed is not None and v not in allowed: v = default
        return v

    # ------------------------------------------------------------------ top bar / navigation
    def _topbar(self):
        bar = QFrame(); bar.setObjectName("topbar"); lay = QHBoxLayout(bar); lay.setContentsMargins(16, 6, 12, 6)
        lay.addWidget(label("AI Image Create", "apptitle")); lay.addSpacing(12)
        self.chip = label("…", "chip"); lay.addWidget(self.chip); lay.addStretch(1)
        about = button("About"); about.clicked.connect(self._about); lay.addWidget(about)
        return bar

    def _set_chip(self, text=None):
        if text:
            self.chip.setText(text); self.chip.setProperty("state", "")
        elif self.engine is None:
            dml = False
            try:
                import onnxruntime as ort; dml = "DmlExecutionProvider" in ort.get_available_providers()
            except Exception:  # noqa: BLE001
                pass
            self.chip.setText("Processor: GPU (DirectML) when ready" if dml and self.device != "cpu" else "Processor: CPU")
            self.chip.setProperty("state", "" if dml and self.device != "cpu" else "cpu")
        else:
            i = self.engine.info
            if i.active == "DirectML":
                half = " · fp16" if self.engine.unet_variant == "fp16" else ""
                self.chip.setText(f"GPU · DirectML{half}" + (f" · {i.adapter}" if i.adapter else "")); self.chip.setProperty("state", "")
            else:
                self.chip.setText("CPU" + (" (GPU unavailable)" if i.fallback_reason and i.requested != "cpu" else ""))
                self.chip.setProperty("state", "cpu")
            self.chip.setToolTip(i.fallback_reason or "")
        self.chip.style().unpolish(self.chip); self.chip.style().polish(self.chip)

    def go(self, page):
        self.stack.setCurrentIndex(page)
        if page == PAGE_MAIN: self._refresh_main(); self.prompt.setFocus()
        focus = {PAGE_SETUP: self.btn_dl, PAGE_PROGRESS: self.btn_cancel, PAGE_RESULT: self.btn_save, PAGE_BLOCKED: self.btn_blk_edit}.get(page)
        if focus is not None and focus.isEnabled(): focus.setFocus()

    def _about(self):
        QMessageBox.about(self, "About AI Image Create",
            f"<b>AI Image Create {__version__}</b> for Windows (x64) — desktop port of the Android app by vanu krishnan.<br><br>"
            "Turns your photo into a new picture from a text prompt, entirely on this PC (nothing is uploaded).<br><br>"
            "<b>Models</b> (downloaded once from github.com/vanukrishnans-source/ai-image-create-models): LCM-Dreamshaper-v7 "
            "(CreativeML OpenRAIL-M — use restrictions apply), SD VAE + TAESD (MIT), T2I-Adapter canny (Apache-2.0), CLIP text "
            "encoder (MIT), Real-ESRGAN general x4v3 (BSD-3-Clause), Stable Diffusion safety checker (CreativeML OpenRAIL-M), "
            "MediaPipe Face Landmarker (Apache-2.0).<br><br>"
            "The safety filter is always on. Uses ONNX Runtime + DirectML (MIT), Qt 6 / PySide6 (LGPLv3), Pillow, NumPy. "
            "See THIRD_PARTY.md next to the app.")

    # ------------------------------------------------------------------ setup page
    def _page_setup(self):
        w = QWidget(); outer = QVBoxLayout(w); outer.setContentsMargins(40, 22, 40, 22); outer.setSpacing(12)
        outer.addWidget(label(f"One-time setup: download the AI models ({M.DOWNLOAD_BYTES / 1e9:.2f} GB)", "title"))
        outer.addWidget(label("The models come from the public GitHub release “ai-image-create-models / models-v1”, are checked "
                              "with SHA-256 and unpacked into your user folder (about "
                              f"{M.INSTALLED_BYTES / 1e9:.1f} GB on disk; PCs with a DirectML GPU also prepare a 1.7 GB "
                              "half-precision copy for speed). You can pause and resume, or import the files from a folder.",
                              "subtitle", True))
        c = card(); g = QGridLayout(c); g.setContentsMargins(18, 14, 18, 14); g.setHorizontalSpacing(18); g.setVerticalSpacing(6)
        self.setup_rows = {}
        for i, s in enumerate(M.ALL):
            name = label(s.label); size = label(f"{s.wts_bytes / 1e6:,.0f} MB", "hint"); st = label("", "hint")
            g.addWidget(name, i, 0); g.addWidget(size, i, 1, Qt.AlignmentFlag.AlignRight); g.addWidget(st, i, 2)
            self.setup_rows[s.name] = st
        g.setColumnStretch(0, 1); outer.addWidget(c)
        self.setup_bar = QProgressBar(); self.setup_bar.setRange(0, 1000); outer.addWidget(self.setup_bar)
        self.setup_status = label("", "subtitle", True); outer.addWidget(self.setup_status)
        outer.addWidget(label("By downloading you accept the model licences, including the CreativeML OpenRAIL-M use "
                              "restrictions (no illegal, harmful or non-consensual content). The safety filter is always on.", "hint", True))
        outer.addStretch(1)
        row = QHBoxLayout()
        self.btn_import = button("Import from folder…", tip="Use the release files copied from another PC")
        self.btn_import.clicked.connect(self._import_models); row.addWidget(self.btn_import)
        row.addStretch(1)
        self.btn_pause = button("Pause"); self.btn_pause.clicked.connect(self._cancel_worker); self.btn_pause.setEnabled(False); row.addWidget(self.btn_pause)
        self.btn_dl = button("Download", "primary", 260); self.btn_dl.clicked.connect(self._download); row.addWidget(self.btn_dl)
        outer.addLayout(row)
        self._refresh_setup()
        return w

    def _refresh_setup(self):
        for s in M.ALL:
            self.setup_rows[s.name].setText("✓ installed" if self.store.is_installed(s) else "")
        have = self.store.bytes_present(); self.setup_bar.setValue(int(1000 * have / M.DOWNLOAD_BYTES))
        if have and not self.store.all_installed():
            self.btn_dl.setText("Resume"); self.setup_status.setText(f"{have / 1e6:,.0f} of {M.DOWNLOAD_BYTES / 1e6:,.0f} MB already here.")

    def _download(self, import_dir=None):
        if self.worker and self.worker.isRunning(): return
        self.btn_dl.setEnabled(False); self.btn_import.setEnabled(False); self.btn_pause.setEnabled(True)
        self.setup_status.setText("Connecting…")
        store = self.store; want_fp16 = self._want_fp16()

        def work(cancel, emit):
            cb = lambda p: emit(("dl", p))
            if import_dir: store.import_from(Path(import_dir), cb, cancel)
            store.ensure(cb, cancel)
            if want_fp16 and not M.fp16_installed(store):
                emit(("gpu", 0.0)); M.derive_unet_fp16(store, lambda f: emit(("gpu", f)), cancel)
            return True
        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._setup_progress); self.worker.done.connect(self._setup_done)
        self.worker.failed.connect(self._setup_failed); self.worker.start()

    def _setup_progress(self, d):
        kind, p = d
        if kind == "gpu":
            self.setup_bar.setValue(int(1000 * p)); self.setup_status.setText(f"Preparing the GPU fast path (one time)… {p * 100:.0f}%"); return
        self.setup_bar.setValue(int(1000 * p.done / max(1, p.total)))
        if p.stage == "download":
            eta = (p.total - p.done) / p.bps if p.bps > 0 else None
            self.setup_status.setText(f"Downloading {p.file} · {p.done / 1e6:,.0f} of {p.total / 1e6:,.0f} MB"
                                      + (f" · {p.bps / 1e6:.1f} MB/s · about {int(eta // 60)} min {int(eta % 60)} s left" if eta else ""))
        elif p.stage == "verify": self.setup_status.setText(f"Checking {p.file} (SHA-256)…")
        elif p.stage == "unpack": self.setup_status.setText(f"Unpacking {p.file}… {p.fraction * 100:.0f}%")
        self._refresh_rows_only()

    def _refresh_rows_only(self):
        for s in M.ALL:
            self.setup_rows[s.name].setText("✓ installed" if self.store.is_installed(s) else "")

    def _setup_done(self, _):
        self.btn_pause.setEnabled(False); self.btn_dl.setEnabled(True); self.btn_import.setEnabled(True)
        self._refresh_setup(); self.setup_status.setText("All set.")
        self.go(PAGE_MAIN); self._start_warmup()

    def _setup_failed(self, msg, tb):
        if tb: log.error("setup failed: %s", tb)
        self.btn_dl.setEnabled(True); self.btn_import.setEnabled(True); self.btn_pause.setEnabled(False); self._refresh_setup()
        if msg == "cancelled":
            self.setup_status.setText("Paused — tap Resume to continue where it stopped."); self.btn_dl.setText("Resume")
        else:
            self.setup_status.setText(f"Setup stopped: {msg}\nCheck the internet connection / free disk space and tap Resume.")

    def _import_models(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with the model release files (.wts + graphs-v1.zip)")
        if d: self._download(import_dir=d)

    def _want_fp16(self):
        if self.device == "cpu" or not self._get("gpu_half", True) or self._get("gpu_half_bad", False): return False
        try:
            import onnxruntime as ort; return "DmlExecutionProvider" in ort.get_available_providers()
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------ engine warm-up (background)
    def _start_warmup(self):
        if self.warm is not None: return
        self.ready.clear(); want_fp16 = self._want_fp16(); store = self.store

        def work(cancel, emit):
            from ..engine import Engine
            from ..pipeline import Pipeline
            if want_fp16 and not M.fp16_installed(store):
                emit(("gpu", 0.0)); M.derive_unet_fp16(store, lambda f: emit(("gpu", f)), cancel)
            eng = Engine(store, self.device, gpu_half=self._get("gpu_half", True) and not self._get("gpu_half_bad", False),
                         on_half_failed=lambda why: self.qs.setValue("gpu_half_bad", True))
            pipe = Pipeline(eng); self.engine, self.pipe = eng, pipe
            emit(("chip", None))
            try:
                eng.warm_up(cb=lambda n, i, t: emit(("warm", (n, i, t))))
            finally:
                self.ready.set()
            return True
        self.warm = Worker(work, self)
        self.warm.progressed.connect(self._warm_progress)
        self.warm.done.connect(lambda _: (self._set_chip(), self._refresh_main()))
        self.warm.failed.connect(lambda m, tb: (log.error("warm-up failed: %s %s", m, tb), self.ready.set(), self._set_chip()))
        self.warm.start()

    def _warm_progress(self, d):
        kind, v = d
        if kind == "gpu": self._set_chip(f"Preparing GPU fast path… {v * 100:.0f}%")
        elif kind == "chip": self._set_chip()
        elif kind == "warm": self.status_hint.setText(f"Loading AI models… ({v[1] + 1}/{v[2]})")

    # ------------------------------------------------------------------ main page
    def _page_main(self):
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(14)
        self.slot = PhotoSlot(); self.slot.clicked.connect(self._pick_photo); self.slot.dropped.connect(self._open_any)
        self.slot.paste_btn.clicked.connect(self._paste)
        lay.addWidget(self.slot, 5)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(10)
        pc = card(); self.prompt_card = pc; pl = QVBoxLayout(pc); pl.setContentsMargins(14, 12, 14, 12); pl.setSpacing(6)
        pl.addWidget(label("What should the new picture be?", "section"))
        self.prompt = QPlainTextEdit(); self.prompt.setPlaceholderText("e.g. watercolor painting · anime style · make it a beach at sunset · "
                                                                        "oil painting · cyberpunk city at night")
        self.prompt.setMinimumHeight(110); self.prompt.textChanged.connect(self._refresh_main)
        pl.addWidget(self.prompt, 1)
        rl.addWidget(pc, 3)
        sc = card(); sl = QVBoxLayout(sc); sl.setContentsMargins(14, 8, 14, 8); sl.setSpacing(0)
        top = QHBoxLayout(); top.addWidget(label("How much to change", "section")); top.addStretch(1)
        self.s_val = label("", "section"); top.addWidget(self.s_val); sl.addLayout(top)
        self.s_strength = QSlider(Qt.Orientation.Horizontal); self.s_strength.setRange(30, 80); self.s_strength.setSingleStep(1); self.s_strength.setPageStep(5)
        self.s_strength.valueChanged.connect(self._strength_changed); sl.addWidget(self.s_strength)
        ends = QHBoxLayout(); ends.addWidget(label("a little (keeps the photo)", "hint")); ends.addStretch(1); ends.addWidget(label("a lot (follows the prompt)", "hint"))
        sl.addLayout(ends); rl.addWidget(sc)
        self.btn_opts = button("▸  Options", "toggle"); self.btn_opts.setCheckable(True); self.btn_opts.toggled.connect(self._toggle_options)
        rl.addWidget(self.btn_opts)
        self.opts_scroll = QScrollArea(); self.opts_scroll.setWidgetResizable(True); self.opts_scroll.setWidget(self._options_panel())
        self.opts_scroll.setVisible(False); rl.addWidget(self.opts_scroll, 4)
        self.status_hint = label("", "hint", True); rl.addWidget(self.status_hint)
        self.btn_create = button("Create", "primary"); self.btn_create.setMinimumHeight(64); self.btn_create.clicked.connect(self._create)
        rl.addWidget(self.btn_create)
        lay.addWidget(right, 6)
        return w

    def _options_panel(self):
        c = card(); grid = QHBoxLayout(c); grid.setContentsMargins(14, 10, 14, 12); grid.setSpacing(18)
        cols = [QVBoxLayout(), QVBoxLayout()]
        for col in cols: col.setSpacing(4); grid.addLayout(col, 1)

        def row(ci, title, widget, hint=None):
            cols[ci].addWidget(label(title, "section")); cols[ci].addWidget(widget)
            if hint: cols[ci].addWidget(label(hint, "hint", True))
        self.o_quality = Segmented([("Best", "best"), ("Fast", "fast")])
        row(0, "Quality", self.o_quality, "Best: 8 AI steps. Fast: 5 steps, ~35% quicker.")
        self.o_size = Segmented([("Standard", "standard"), ("Large", "large")])
        row(0, "Output size", self.o_size, "Standard ≈ 512² (e.g. 576×384, 1152×768 with 2×). Large ≈ 768² (e.g. 768×512, "
                                           "1536×1024 with 2×) — more detail, ~2.5× slower, faces may drift more. Never stretched.")
        self.o_upscale = QCheckBox("Sharpen + enlarge 2×"); cols[0].addWidget(self.o_upscale)
        self.o_face = QCheckBox("Keep face likeness"); cols[0].addWidget(self.o_face)
        self.o_colors = QCheckBox("Keep the photo's colours"); cols[0].addWidget(self.o_colors)
        cols[1].addWidget(label("Seed", "section"))
        self.o_random = QCheckBox("New random seed each time"); cols[1].addWidget(self.o_random)
        self.o_seed = QSpinBox(); self.o_seed.setRange(1, 2 ** 31 - 1); self.o_seed.setMinimumWidth(170); cols[1].addWidget(self.o_seed)
        self.o_strict = Segmented([("Relaxed", "relaxed"), ("Standard", "standard")])
        row(1, "Safety filter strictness", self.o_strict, "The safety filter is always on. Relaxed (default) blocks clear "
                                                          "nudity; Standard also blocks borderline pictures.")
        self.o_device = Segmented([("Auto (GPU)", "auto"), ("CPU only", "cpu")])
        row(1, "Processor", self.o_device, "Applies after restarting the app.")
        self.o_half = QCheckBox("GPU half precision (faster)"); cols[1].addWidget(self.o_half)
        for col in cols: col.addStretch(1)
        for sgn in (self.o_quality.changed, self.o_size.changed, self.o_strict.changed, self.o_device.changed):
            sgn.connect(lambda _v: self._save_options())
        self.o_size.changed.connect(lambda _v: self.photo is not None and self.slot.set_image(prepare(self.photo, _v)))
        for cb in (self.o_upscale, self.o_face, self.o_colors, self.o_random, self.o_half):
            cb.toggled.connect(lambda _v: self._save_options())
        self.o_seed.valueChanged.connect(lambda _v: self._save_options())
        return c

    def _load_options(self):
        self._loading = True
        try:
            self._load_options_inner()
        finally:
            self._loading = False
        self.o_seed.setEnabled(not self.o_random.isChecked())

    def _load_options_inner(self):
        self.s_strength.setValue(int(round(100 * min(0.8, max(0.3, self._get("strength", 0.55))))))
        self.o_quality.set(self._get("quality", "best", ("best", "fast")))
        self.o_size.set(self._get("size", "standard", ("standard", "large")))
        self.o_upscale.setChecked(self._get("upscale", True)); self.o_face.setChecked(self._get("keep_face", True))
        self.o_colors.setChecked(self._get("keep_colors", True)); self.o_random.setChecked(self._get("random_seed", True))
        self.o_seed.setValue(min(2 ** 31 - 1, max(1, self._get("seed", 1234))))
        self.o_strict.set(self._get("strictness", "relaxed", ("relaxed", "standard")))   # anything else -> Relaxed default
        self.o_device.set(self._get("device", "auto", ("auto", "cpu"))); self.o_half.setChecked(self._get("gpu_half", True))
        self._strength_changed(self.s_strength.value())

    def _save_options(self):
        if getattr(self, "_loading", False): return
        q = self.qs
        q.setValue("strength", self.s_strength.value() / 100); q.setValue("quality", self.o_quality.value() or "best")
        q.setValue("size", self.o_size.value() or "standard"); q.setValue("upscale", self.o_upscale.isChecked())
        q.setValue("keep_face", self.o_face.isChecked()); q.setValue("keep_colors", self.o_colors.isChecked())
        q.setValue("random_seed", self.o_random.isChecked()); q.setValue("seed", self.o_seed.value())
        q.setValue("strictness", self.o_strict.value() or "relaxed"); q.setValue("device", self.o_device.value() or "auto")
        if q.value("gpu_half") is not None and str(q.value("gpu_half")).lower() != str(self.o_half.isChecked()).lower():
            q.setValue("gpu_half_bad", False)
        q.setValue("gpu_half", self.o_half.isChecked())
        self.o_seed.setEnabled(not self.o_random.isChecked())

    def _strength_changed(self, v):
        words = "subtle" if v < 45 else ("balanced" if v < 62 else "strong")
        self.s_val.setText(f"{v / 100:.2f} · {words}")
        if hasattr(self, "o_quality"): self._save_options()

    def _toggle_options(self, on):
        self.opts_scroll.setVisible(on); self.btn_opts.setText(("▾" if on else "▸") + "  Options")
        self.prompt.setMinimumHeight(60 if on else 110)
        self.prompt_card.setMaximumHeight(130 if on else 16777215)

    def params(self, seed=None) -> GenParams:
        if seed is None:
            seed = new_seed() if self.o_random.isChecked() else self.o_seed.value()
        return GenParams(prompt=self.prompt.toPlainText().strip(), strength=self.s_strength.value() / 100,
                         quality=self.o_quality.value() or "best", seed=int(seed), keep_face=self.o_face.isChecked(),
                         keep_colors=self.o_colors.isChecked(), upscale=self.o_upscale.isChecked(),
                         strictness=self.o_strict.value() if self.o_strict.value() in ("relaxed", "standard") else "relaxed",
                         size=self.o_size.value() or "standard")

    def _refresh_main(self):
        has_p = self.photo is not None; has_t = bool(self.prompt.toPlainText().strip())
        self.btn_create.setEnabled(has_p and has_t and self.store.all_installed())
        if not has_p: self.status_hint.setText("Attach a photo to start.")
        elif not has_t: self.status_hint.setText("Type what the new picture should be.")
        elif self.warm is not None and self.warm.isRunning(): pass
        else:
            p = self.params(seed=1); est = self._estimate(p)
            self.status_hint.setText(f"Ready. " + (f"About {est:.0f} s on this PC." if est else ""))

    def _estimate(self, p: GenParams):
        if not self.sec_per_eval: return None
        evals = 8 if p.quality == "best" else 5
        f = 2.5 if p.size == "large" else 1.0
        return self.sec_per_eval * f * (evals + 3.6 + (0.7 if p.upscale else 0))

    # photo input
    def _pick_photo(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose a photo", str(Path.home() / "Pictures"),
                                           "Pictures (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;All files (*)")
        if f: self.set_photo(f)

    def _open_any(self, x):
        if isinstance(x, QImage): self.set_photo_rgb(qimage_to_rgb(x), "pasted picture")
        elif x: self.set_photo(x)

    def _paste(self):
        md = QApplication.clipboard().mimeData()
        if md is None: return
        if md.hasImage():
            img = QApplication.clipboard().image()
            if not img.isNull(): self.set_photo_rgb(qimage_to_rgb(img), "pasted picture"); return
        if md.hasUrls() and md.urls() and md.urls()[0].isLocalFile():
            self.set_photo(md.urls()[0].toLocalFile()); return
        if self.stack.currentIndex() == PAGE_MAIN and self.prompt.hasFocus() and md.hasText():
            self.prompt.insertPlainText(md.text())

    def set_photo(self, path):
        try:
            rgb = load_photo(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Can't open this file", f"That file couldn't be opened as a picture.\n\n{e}"); return
        self.set_photo_rgb(rgb, Path(path).name)

    def set_photo_rgb(self, rgb, name):
        if rgb.shape[0] < 64 or rgb.shape[1] < 64:
            QMessageBox.warning(self, "Photo too small", "Please use a photo at least 64×64 pixels."); return
        self.photo = rgb; self.photo_name = name
        self.slot.set_image(prepare(rgb, self.o_size.value() or "standard"))
        self.slot.info.setText(f"{name} · {rgb.shape[1]}×{rgb.shape[0]}")
        if self.stack.currentIndex() != PAGE_SETUP: self.go(PAGE_MAIN)
        self._refresh_main()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage(): e.acceptProposedAction()

    def dropEvent(self, e):
        md = e.mimeData()
        if md.hasUrls() and md.urls(): self._open_any(md.urls()[0].toLocalFile())
        elif md.hasImage(): self._open_any(QImage(md.imageData()))

    # ------------------------------------------------------------------ create / progress
    def _page_progress(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(60, 30, 60, 30); lay.setSpacing(14)
        lay.addStretch(1)
        row = QHBoxLayout(); self.prog_thumb = QLabel(); self.prog_thumb.setFixedSize(360, 260); self.prog_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addStretch(1); row.addWidget(self.prog_thumb)
        col = QVBoxLayout(); col.setSpacing(8)
        self.prog_title = label("Creating your picture…", "big"); col.addWidget(self.prog_title)
        self.prog_prompt = label("", "subtitle", True); self.prog_prompt.setMaximumWidth(560); col.addWidget(self.prog_prompt)
        col.addStretch(1); row.addSpacing(24); row.addLayout(col); row.addStretch(1); lay.addLayout(row)
        self.prog_bar = QProgressBar(); self.prog_bar.setRange(0, 1000); lay.addWidget(self.prog_bar)
        self.prog_stage = label("", "section"); lay.addWidget(self.prog_stage)
        self.prog_eta = label("", "subtitle"); lay.addWidget(self.prog_eta)
        lay.addStretch(1)
        r2 = QHBoxLayout(); r2.addStretch(1); self.btn_cancel = button("Cancel", min_w=220); self.btn_cancel.clicked.connect(self._cancel_worker)
        r2.addWidget(self.btn_cancel); r2.addStretch(1); lay.addLayout(r2)
        return w

    def _create(self, seed=None, photo=None):
        if self.worker and self.worker.isRunning(): return
        if photo is not None: pass
        if self.photo is None or not self.prompt.toPlainText().strip(): return
        p = self.params(seed); self.last_params = p
        if not self.o_random.isChecked() and seed is not None: self.o_seed.setValue(p.seed)
        rgb = self.photo
        self.prog_thumb.setPixmap(pix(prepare(rgb, p.size), 360, 260))
        self.prog_prompt.setText(f"“{p.prompt}” · change {p.strength:.2f} · {'Best' if p.quality == 'best' else 'Fast'} · seed {p.seed}")
        self.prog_bar.setValue(0); self.prog_stage.setText("Getting ready…"); self.prog_eta.setText(""); self.btn_cancel.setEnabled(True)
        self.go(PAGE_PROGRESS)
        if self.warm is None: self._start_warmup()
        hint = self.sec_per_eval * (2.5 if p.size == "large" else 1.0) if self.sec_per_eval else None

        def work(cancel, emit):
            while not self.ready.wait(0.2):
                if cancel.is_set(): raise GenCancelled()
            if self.pipe is None: raise RuntimeError("the AI models could not be loaded (see the log file)")
            return self.pipe.generate(rgb, p, progress=lambda l, f, e: emit((l, f, e)), cancel=cancel, sec_per_eval_hint=hint)
        self.worker = Worker(work, self)
        self.worker.progressed.connect(self._progress); self.worker.done.connect(self._finished); self.worker.failed.connect(self._failed)
        self.worker.start()

    def _progress(self, d):
        l, f, eta = d
        self.prog_bar.setValue(int(1000 * f)); self.prog_stage.setText(l)
        self.prog_eta.setText(f"About {eta:.0f} s left" if eta and eta > 0.5 else "")

    def _cancel_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel.set(); self.btn_cancel.setEnabled(False); self.btn_pause.setEnabled(False)
            self.prog_stage.setText("Stopping…")

    def _failed(self, msg, tb):
        if tb: log.error("generate failed: %s", tb)
        self._set_chip()
        self.go(PAGE_MAIN)
        if msg != "cancelled":
            QMessageBox.warning(self, "Something went wrong", f"The picture couldn't be created.\n\n{msg}")

    def _finished(self, r: GenResult):
        self._set_chip()
        spe = r.timings.get("sec_per_unet_eval")
        if spe: self.sec_per_eval = spe / (2.5 if (self.last_params and self.last_params.size == "large") else 1.0)
        if r.blocked:
            self.result = None; self.saved_path = None       # nothing is kept, shown or saved
            self.blk_info.setText(f"Safety filter: {STRICT.get(self.last_params.strictness if self.last_params else 'relaxed')} · seed {r.seed}")
            self.go(PAGE_BLOCKED); return
        self.result = r; self.saved_path = None
        self.ba.set_images(r.input, r.image)
        dev = "GPU (DirectML)" if r.provider == "DirectML" else "CPU"
        self.res_info.setText(f"{r.image.shape[1]}×{r.image.shape[0]} · seed {r.seed} · {r.seconds:.0f} s on {dev}")
        self.btn_save.setText("Save"); self.btn_save.setEnabled(True)
        self.go(PAGE_RESULT)

    # ------------------------------------------------------------------ result page
    def _page_result(self):
        w = QWidget(); lay = QHBoxLayout(w); lay.setContentsMargins(12, 10, 12, 10); lay.setSpacing(12)
        self.ba = BeforeAfter(); lay.addWidget(self.ba, 1)
        side = QVBoxLayout(); side.setSpacing(8)
        side.addWidget(label("Your new picture", "big"))
        self.res_info = label("", "hint", True); side.addWidget(self.res_info)
        side.addWidget(label("Drag the slider on the picture to compare.", "hint", True))
        self.btn_save = button("Save", "primary"); self.btn_save.setMinimumHeight(58); self.btn_save.clicked.connect(self._save)
        side.addWidget(self.btn_save)
        b_open = button("Open folder"); b_open.clicked.connect(self._open_folder); side.addWidget(b_open)
        b_copy = button("Copy image"); b_copy.clicked.connect(self._copy); side.addWidget(b_copy)
        b_regen = button("Regenerate (new seed)"); b_regen.clicked.connect(lambda: self._create(seed=new_seed())); side.addWidget(b_regen)
        b_use = button("Use result as new input"); b_use.clicked.connect(self._use_result); side.addWidget(b_use)
        side.addWidget(label(f"Saved pictures go to {pictures_dir()}", "hint", True))
        side.addStretch(1)
        b_back = button("← Edit prompt"); b_back.clicked.connect(lambda: self.go(PAGE_MAIN)); side.addWidget(b_back)
        sw = QWidget(); sw.setLayout(side); sw.setFixedWidth(300); lay.addWidget(sw)
        return w

    def _save(self):
        if self.stack.currentIndex() != PAGE_RESULT or self.result is None or self.result.image is None or self.result.blocked: return
        from PIL import Image, PngImagePlugin
        d = pictures_dir(); d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = d / f"AIImage_{stamp}_{self.result.seed}.png"
        meta = PngImagePlugin.PngInfo(); meta.add_text("Software", f"AI Image Create {__version__} (Windows)")
        meta.add_text("Prompt", self.last_params.prompt if self.last_params else "")
        Image.fromarray(self.result.image).save(path, pnginfo=meta)
        self.saved_path = path; self.btn_save.setText("Saved ✓"); self.btn_save.setEnabled(False)
        self.res_info.setText(self.res_info.text().split("\nSaved")[0] + f"\nSaved to {path}")

    def _open_folder(self):
        d = pictures_dir(); d.mkdir(parents=True, exist_ok=True)
        if os.name == "nt" and self.saved_path is not None:
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(str(self.saved_path))]); return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))

    def _copy(self):
        if self.result is not None and self.result.image is not None:
            QApplication.clipboard().setImage(qimage(self.result.image)); self.res_info.setText(self.res_info.text().split("\nCopied")[0] + "\nCopied to the clipboard.")

    def _use_result(self):
        if self.result is not None and self.result.image is not None:
            self.set_photo_rgb(self.result.image, "previous result"); self.go(PAGE_MAIN)

    # ------------------------------------------------------------------ blocked page
    def _page_blocked(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(80, 40, 80, 40); lay.setSpacing(14)
        lay.addStretch(1)
        icon = label("🛡", "title"); icon.setStyleSheet("font-size:64px;"); icon.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(icon)
        t = label("This picture was held back by the safety filter", "title"); t.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(t)
        m = label("The result looked like it might contain nudity, so it wasn't shown or saved. "
                  "Try a different prompt, a lower “How much to change”, or simply try again with a new seed.", "subtitle", True)
        m.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(m)
        self.blk_info = label("", "hint"); self.blk_info.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(self.blk_info)
        lay.addStretch(1)
        row = QHBoxLayout(); row.addStretch(1)
        self.btn_blk_edit = button("Edit prompt", min_w=240); self.btn_blk_edit.clicked.connect(lambda: self.go(PAGE_MAIN)); row.addWidget(self.btn_blk_edit)
        b2 = button("Try again (new seed)", "primary", 280); b2.clicked.connect(lambda: self._create(seed=new_seed())); row.addWidget(b2)
        row.addStretch(1); lay.addLayout(row)
        return w

    # ------------------------------------------------------------------ keys / close
    def keyPressEvent(self, e: QKeyEvent):
        page = self.stack.currentIndex(); k = e.key()
        if k == Qt.Key.Key_Escape:
            if page == PAGE_PROGRESS: self._cancel_worker()
            elif page in (PAGE_RESULT, PAGE_BLOCKED): self.go(PAGE_MAIN)
            return
        if k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and page == PAGE_MAIN and (e.modifiers() & Qt.KeyboardModifier.ControlModifier or not self.prompt.hasFocus()):
            self._create(); return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        for wk in (self.worker, self.warm):
            if wk is not None and wk.isRunning():
                wk.cancel.set(); wk.wait(20000)
        e.accept()


def make_app(argv=None):
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("AI Image Create"); app.setOrganizationName("vanu")
    from ..resources_path import res
    check = res("check.png").as_posix()
    app.setStyle("Fusion"); app.setStyleSheet(QSS + f'QCheckBox::indicator:checked {{ image: url("{check}"); }}\n')
    ico = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2] / "packaging")) / "icon.ico"
    if ico.is_file(): app.setWindowIcon(QIcon(str(ico)))
    return app


def run(args=None) -> int:
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("vanu.AIImageCreate")
        except Exception:  # noqa: BLE001
            pass
    app = make_app()
    store = M.ModelStore(Path(args.models) if args is not None and args.models else
                         (Path(os.environ["AIC_MODELS"]) if os.environ.get("AIC_MODELS") else None))
    dev = getattr(args, "device", "auto") if args is not None else "auto"
    win = MainWindow(store, dev)
    from .gamepad import Gamepad
    win.gamepad = Gamepad(win)
    win.resize(1280, 720)
    if QApplication.primaryScreen() and QApplication.primaryScreen().availableGeometry().width() <= 1400:
        win.showMaximized()          # Ally X: 1920x1080 @150% = 1280x720 logical
    else:
        win.show()
    if args is not None and getattr(args, "photo", None): win.set_photo(args.photo)
    if args is not None and getattr(args, "prompt", None): win.prompt.setPlainText(args.prompt)
    if args is not None and getattr(args, "gui_smoke", None):
        def smoke():
            from ..cli import app_data_dir
            info = {"page": win.stack.currentIndex(), "chip": win.chip.text(), "title": win.windowTitle(),
                    "models_installed": store.all_installed(), "time": time.time()}
            (app_data_dir() / "gui_smoke.json").write_text(json.dumps(info))
            win.close(); app.quit()
        QTimer.singleShot(int(args.gui_smoke * 1000), smoke)
    return app.exec()


def screenshots(args) -> int:
    from .screens import take_screenshots
    return take_screenshots(args)
