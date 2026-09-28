"""Render the real UI states to PNGs (README / release notes / CI evidence).

Real: a real partial model download into a temporary folder (then cancelled), the real photo + prompt, the Options
panel, a real generation (progress captured mid-run, then the Result page with the before/after slider).
The "blocked" page is rendered by handing the UI a blocked result object (image withheld) — no explicit picture is
ever generated for it. Run with QT_QPA_PLATFORM=offscreen; 1280x720 logical at QT_SCALE_FACTOR=1.5 = 1920x1080 PNGs,
i.e. what the Ally X shows at 150 % scaling.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QSettings

from .. import models as M
from ..pipeline import GenResult
from .app import PAGE_MAIN, PAGE_PROGRESS, PAGE_SETUP, MainWindow, make_app


def take_screenshots(args) -> int:
    os.environ.setdefault("QT_SCALE_FACTOR", "1.5")
    if os.name == "nt" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        # the offscreen platform uses Qt's FreeType font database, which only finds fonts via QT_QPA_FONTDIR
        os.environ.setdefault("QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"))
    out = Path(args.screenshots); out.mkdir(parents=True, exist_ok=True)
    app = make_app(["AIImageCreate"])
    tmpq = Path(tempfile.mkdtemp(prefix="aic_qs_"))
    qs = QSettings(str(tmpq / "settings.ini"), QSettings.Format.IniFormat)    # fresh defaults, not the user's settings

    def spin(sec=0.05, until=None, timeout=600):
        t0 = time.time()
        while True:
            app.processEvents(); time.sleep(0.02)
            if until is None and time.time() - t0 >= sec: return True
            if until is not None and until(): return True
            if time.time() - t0 > timeout: return False

    def shot(win, name):
        spin(0.3)
        p = out / f"{name}.png"; win.grab().save(str(p)); print("wrote", p, flush=True)

    # 1-2) Setup page with an empty model folder, then a few seconds of a real download (paused -> .part kept)
    tmp = Path(tempfile.mkdtemp(prefix="aic_setup_"))
    w0 = MainWindow(M.ModelStore(tmp), "cpu", qs); w0.resize(1280, 720); w0.show()
    w0.go(PAGE_SETUP); shot(w0, "01_setup")
    w0._download()
    spin(until=lambda: w0.setup_bar.value() > 12 or not w0.worker.isRunning(), timeout=90)
    shot(w0, "02_setup_downloading")
    w0._cancel_worker(); spin(until=lambda: not w0.worker.isRunning(), timeout=60); spin(0.3)
    shot(w0, "02b_setup_paused")
    w0.close(); shutil.rmtree(tmp, ignore_errors=True)

    store = M.ModelStore(Path(args.models) if args.models else (Path(os.environ["AIC_MODELS"]) if os.environ.get("AIC_MODELS") else None))
    win = MainWindow(store, getattr(args, "device", "auto"), qs); win.resize(1280, 720); win.show()
    spin(until=lambda: win.ready.is_set(), timeout=900)
    win.go(PAGE_MAIN); shot(win, "03_main_empty")
    if not args.photo:
        return 0
    win.set_photo(args.photo); win.prompt.setPlainText(args.prompt or "watercolor painting")
    win.o_random.setChecked(False); win.o_seed.setValue(args.seed or 1234)
    win.go(PAGE_MAIN); shot(win, "04_main_photo_prompt")
    win.btn_opts.setChecked(True); spin(0.2); shot(win, "05_options")
    sb = win.opts_scroll.verticalScrollBar(); sb.setValue(sb.maximum()); spin(0.2)
    if sb.maximum() > 0: shot(win, "05b_options_more")
    sb.setValue(0)
    win.btn_opts.setChecked(False)
    win._create()
    spin(until=lambda: (not win.worker.isRunning()) or (win.stack.currentIndex() == PAGE_PROGRESS and win.prog_bar.value() > 350), timeout=900)
    shot(win, "06_progress")
    spin(until=lambda: not win.worker.isRunning(), timeout=1800); spin(0.5)
    win.ba.pos_frac = 0.5; win.ba.update()
    shot(win, "07_result_before_after")
    r = win.result
    print("result:", None if r is None else r.summary(), flush=True)
    # blocked page: a blocked result as the pipeline returns it (no pixels) — nothing explicit is generated
    fake = GenResult(image=None, base=None, input=r.input if r is not None else None, blocked=True, nsfw=0.97, threshold=0.85,
                     prompt="(example)", negative="", seed=4242, seconds=0.0, timesteps=[], size=(576, 384), faces=0,
                     safety_checked=True)
    win._finished(fake); shot(win, "08_blocked")
    win.close(); shutil.rmtree(tmpq, ignore_errors=True)
    return 0
