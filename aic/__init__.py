"""AI Image Create for Windows — desktop port of the Android AI Image Create 1.0.1 app (com.vanu.aiimagecreate)."""
__version__ = "1.0.0"


def _install_matplotlib_stub():
    """mediapipe 1.0.x imports matplotlib.pyplot (drawing helpers only) while importing its package. The packaged
    app doesn't ship matplotlib, so register an empty stand-in BEFORE anything imports mediapipe."""
    import sys
    if "matplotlib.pyplot" in sys.modules:
        _stub_cv2(); return
    try:
        import importlib.util
        if importlib.util.find_spec("matplotlib") is not None:
            _stub_cv2(); return
    except Exception:  # noqa: BLE001
        pass
    import types
    mpl = types.ModuleType("matplotlib"); plt = types.ModuleType("matplotlib.pyplot")
    mpl.pyplot = plt; sys.modules["matplotlib"] = mpl; sys.modules["matplotlib.pyplot"] = plt
    _stub_cv2()


def _stub_cv2():
    """mediapipe.tasks.python.vision imports cv2 for its drawing helpers only; the app doesn't ship OpenCV."""
    import sys
    if "cv2" in sys.modules and sys.modules["cv2"] is not None:
        return
    try:
        import importlib.util
        if importlib.util.find_spec("cv2") is not None:
            return
    except Exception:  # noqa: BLE001
        pass
    import types
    sys.modules["cv2"] = types.ModuleType("cv2")


_install_matplotlib_stub()
