"""Dark theme for a 7-inch 1080p touchscreen at ~150% Windows scaling (1280x720 logical). Tap targets >= 48 px."""

QSS = """
QWidget { background:#12141a; color:#e8eaed; font-size:15px; font-family:"Segoe UI","Noto Sans",sans-serif; }
QMainWindow, QDialog { background:#12141a; }
QLabel#title { font-size:22px; font-weight:600; color:#fff; background:transparent; }
QLabel#subtitle { font-size:14px; color:#9aa0a6; background:transparent; }
QLabel#hint { font-size:13px; color:#80868b; background:transparent; }
QLabel#section { font-size:16px; font-weight:600; color:#e8eaed; background:transparent; }
QLabel#big { font-size:20px; font-weight:600; color:#fff; background:transparent; }
QPushButton { background:#2a2f3a; color:#e8eaed; border:1px solid #3c4454; border-radius:10px;
              padding:10px 18px; min-height:30px; font-size:15px; }
QPushButton:hover { background:#343b4a; }
QPushButton:pressed { background:#1e222b; }
QPushButton:disabled { color:#5f6368; background:#1a1d24; }
QPushButton#primary { background:#8b7bff; color:#0c0a1f; border:none; font-weight:700; font-size:17px; }
QPushButton#primary:hover { background:#a497ff; }
QPushButton#primary:disabled { background:#2d2952; color:#6f6a99; }
QPushButton#toggle { background:transparent; border:none; color:#b9b2ff; font-weight:600; text-align:left; padding:8px 4px; }
QPushButton#seg { background:#1f232c; border:1px solid #3c4454; border-radius:10px; font-size:15px; min-height:34px; }
QPushButton#seg:checked { background:#8b7bff; color:#0c0a1f; border:none; font-weight:600; }
QFrame#card { background:#1a1e27; border:1px solid #2a2f3a; border-radius:14px; }
QFrame#card QLabel, QFrame#card QCheckBox { background:transparent; }
QFrame#topbar { background:#0d0f14; border-bottom:1px solid #2a2f3a; }
QLabel#apptitle { font-size:19px; font-weight:600; color:#fff; background:transparent; }
QLabel#chip { background:#2a2552; color:#c9c2ff; border-radius:13px; padding:5px 12px; font-size:13px; }
QLabel#chip[state="cpu"] { background:#3b3324; color:#ffcf8a; }
QLabel#slot { background:#12151c; border:2px dashed #3c4454; border-radius:12px; color:#80868b; font-size:16px; }
QPlainTextEdit { background:#0f1117; border:1px solid #3c4454; border-radius:12px; padding:10px; font-size:19px; }
QPlainTextEdit:focus { border:2px solid #8b7bff; }
QProgressBar { background:#1a1e27; border:1px solid #2a2f3a; border-radius:9px; text-align:center; min-height:26px; color:#e8eaed; }
QProgressBar::chunk { background:#8b7bff; border-radius:8px; }
QSlider { min-height:44px; background:transparent; }
QSlider::groove:horizontal { height:8px; background:#2a2f3a; border-radius:4px; }
QSlider::sub-page:horizontal { background:#6f60e8; border-radius:4px; }
QSlider::handle:horizontal { width:32px; height:32px; margin:-12px 0; background:#8b7bff; border-radius:16px; }
QSpinBox, QLineEdit { background:#0f1117; border:1px solid #3c4454; border-radius:8px; padding:8px 10px; min-height:28px; font-size:16px; }
QSpinBox::up-button, QSpinBox::down-button { width:34px; }
QCheckBox { spacing:12px; font-size:15px; background:transparent; min-height:42px; }
QCheckBox::indicator { width:26px; height:26px; border:2px solid #5a6273; border-radius:7px; background:#0f1117; }
QCheckBox::indicator:checked { background:#8b7bff; border:2px solid #b9b2ff; }
QScrollArea { border:none; background:transparent; }
QScrollBar:vertical { width:14px; background:#12141a; }
QScrollBar::handle:vertical { background:#3c4454; border-radius:6px; min-height:40px; }
QPushButton:focus, QCheckBox:focus, QSlider:focus, QSpinBox:focus { outline:none; border:2px solid #ff8a3d; }
QMessageBox QLabel { background:transparent; }
QToolTip { background:#1a1e27; color:#e8eaed; border:1px solid #3c4454; }
"""
