# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir build for AI Image Create (Windows x64). Built on GitHub Actions windows-latest (.github/workflows/build.yml).
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None
root = Path(SPECPATH)

datas = [
    (str(root / "aic" / "resources"), "aic/resources"),
    (str(root / "packaging" / "icon.ico"), "."),
    (str(root / "THIRD_PARTY.md"), "."),
    (str(root / "LICENSE"), "."),
]
binaries = []
hidden = ["onnxruntime", "mediapipe", "regex", "PIL.Image", "PIL.PngImagePlugin", "PIL.JpegImagePlugin", "PIL.WebPImagePlugin",
          "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"]
for pkg in ("mediapipe", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hidden += h

a = Analysis(
    [str(root / "aic" / "__main__.py")],
    pathex=[str(root)],
    binaries=binaries, datas=datas, hiddenimports=hidden,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "cv2", "IPython", "notebook", "pytest", "onnx", "onnxconverter_common", "sounddevice",
              "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtNetwork", "PySide6.QtOpenGL", "PySide6.QtPdf", "PySide6.QtSvg", "PySide6.QtDBus"],
    win_no_prefer_redirects=False, win_private_assemblies=False, cipher=block_cipher, noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AIImageCreate", debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=False, disable_windowed_traceback=False, argv_emulation=False,
          icon=str(root / "packaging" / "icon.ico"))
# console companion: --selftest / --generate / --bench print to the terminal
exe_console = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AIImageCreate_cli", debug=False, bootloader_ignore_signals=False,
                  strip=False, upx=False, console=True, disable_windowed_traceback=False, icon=str(root / "packaging" / "icon.ico"))
coll = COLLECT(exe, exe_console, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, upx_exclude=[], name="AIImageCreate")
