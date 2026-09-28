# AI Image Create for Windows

Desktop port of the Android **AI Image Create** app (`com.vanu.aiimagecreate` 1.0.1). Made for the **ASUS ROG Ally X** (7″ 1080p touch, Windows 11, Radeon 780M, 24 GB), and works on any Windows 10/11 x64 PC with automatic CPU fallback.

Turn a photo into a new picture from a text prompt **alone**. Everything runs on your PC — nothing is uploaded.

| | |
|---|---|
| **Download** | [Latest release](https://github.com/vanukrishnans-source/ai-image-create-windows/releases/latest) — portable zip |
| **Models** | [ai-image-create-models / models-v1](https://github.com/vanukrishnans-source/ai-image-create-models/releases/tag/models-v1) (1.45 GB, downloaded once by the app) |
| **Android twin** | Same pipeline as the phone app (LCM-Dreamshaper-v7 4-step img2img + T2I-Adapter canny + face likeness + colour keeping + Real-ESRGAN 2× + NSFW filter) |

## Install (first run)

1. Download `AIImageCreate-1.0.0-win64.zip` from the latest release and unzip it somewhere with a few GB free (e.g. `C:\Apps\`).
2. Open the `AIImageCreate` folder and run **`AIImageCreate.exe`**.
3. **Windows SmartScreen** (the app isn’t code-signed): click **More info**, then **Run anyway**.
4. Tap **Download** for the one-time AI model download:
   - **1.45 GB** from the public `ai-image-create-models` release (SHA-256 checked, resumable).
   - Unpacks to about **4.8 GB** under `%LOCALAPPDATA%\AIImageCreate\models`.
   - On a DirectML GPU PC the app also prepares a **1.7 GB** half-precision UNet for speed (derived locally from the same weights — no extra download).
5. Accept the model licences (including CreativeML OpenRAIL-M use restrictions). The safety filter is always on.

You can also copy the release files to another PC and use **Import from folder…**.

## How to use

1. Attach a photo — **Open…**, drag-and-drop, or **Paste** (Ctrl+V).
2. Type what the new picture should be (e.g. `watercolor painting`, `anime style`, `make it a beach at sunset`).
3. Set **How much to change** (default 0.55 — balanced).
4. Tap **Create**.

Everything else lives under the collapsed **Options** panel:

| Option | Default |
|---|---|
| Quality Best / Fast | **Best** (8 AI steps; Fast = 5) |
| Output size Standard / Large | **Standard** (512 px long side, e.g. 576×384). Large = 768 px (~2.3× slower; faces may drift more). Aspect is always preserved — never stretched. |
| Sharpen + enlarge 2× | **on** |
| Keep face likeness | **on** |
| Keep the photo’s colours | **on** |
| Seed | new random each time |
| Safety filter strictness | **Relaxed** (block if score > 0.85). Standard = > 0.5. The filter cannot be turned off. |
| Processor | Auto (GPU / DirectML when available) |

**Result page:** drag the before/after slider, **Save** (to `Pictures\AIImageCreate`), **Open folder**, **Copy image**, **Regenerate** (new seed), **Use result as new input**.

### Safety filter (always on)

- Every picture goes through the Stable Diffusion safety checker. There is **no** setting, config file, registry key, environment variable, CLI flag or code path that skips it.
- Nudity terms (`nude, naked, nsfw, nipples, genitals, sexual`) are always appended (hidden) to the negative prompt.
- A blocked result shows a friendly message and **nothing is saved or shown**.

## Hardware notes (ROG Ally X)

| Path | Est. time per picture (576×384, Best, 2× on, warm) |
|---|---|
| **Radeon 780M · DirectML · fp16 UNet** | **~8–14 s** *(estimate — see below)* |
| Radeon 780M · DirectML · fp32 UNet | ~15–25 s *(estimate)* |
| Ally X CPU (Ryzen Z1 Extreme) | ~12–20 s *(estimate)* |
| This CI runner (4 vCPU, CPU only) | ~20–35 s *(measured)* |

**How the 780M estimates were made.** The Windows hosted runner has no usable GPU (Microsoft Hyper-V Video; DirectML is present but not used). Locally on an 8-vCPU Xeon the UNet costs ~1.5 s per eval at 576×384. Published RX 6600 DirectML SD1.5 numbers are ~0.22 s/eval (with CFG); the 780M is roughly 0.5× that throughput, and a Framework 7840U (same 780M) saw ~6× GPU vs CPU in ComfyUI. Combining those gives **~0.3 s/eval fp16** and **~0.65 s/eval fp32** on the 780M, and **~0.6–1.0 s/eval** on the Z1 Extreme CPU — labelled **estimates**, not Ally X measurements. Prefer Standard size + Best + 2× for quality; use Large only when you want more detail and can wait.

The app badge shows **GPU · DirectML · fp16** / **GPU · DirectML** / **CPU (GPU unavailable)** so you always know which path is active. If half precision ever produces NaN/Inf on a particular driver, the app falls back to fp32 permanently for that install (toggle it back on in Options).

## What’s inside the zip

```
AIImageCreate/
  AIImageCreate.exe          # the app (double-click)
  AIImageCreate_cli.exe      # same app with a console (for --selftest / scripting)
  _internal/                 # Python + ONNX Runtime + Qt + MediaPipe
  README.md  LICENSE  THIRD_PARTY.md
```

No installer, no admin rights, no services.

## CLI (optional)

```bat
AIImageCreate_cli.exe --selftest --photo photo.jpg --dml-smoke
AIImageCreate_cli.exe --generate photo.jpg out.png --prompt "oil painting" --device auto
AIImageCreate_cli.exe --bench --device auto
AIImageCreate_cli.exe --download
```

The NSFW filter runs in every mode that produces a picture.

## Building from source

Built and tested on GitHub Actions `windows-latest` (see `.github/workflows/build.yml`):

- Python 3.13 + PySide6 + `onnxruntime-directml` + MediaPipe + PyInstaller **onedir**
- Packaged-exe selftest (CPU generate from a private-person photo; NSFW check confirmed)
- Windowed-exe selftest, GUI launch smoke, DirectML smoke, offscreen UI screenshots at 150 % scaling
- Parity: packaged CPU pipeline vs the unmodified Python reference (`reference/sd_ref.py`, ORT 1.22.1) — PSNR must be > 60 dB (Android Kotlin achieved 81–84 dB; this port is bit-identical on the same ORT)
- Sample sheet of original + 5 prompt-only results
- Portable zip + SHA256SUMS published as a GitHub Release

```bat
pip install -r requirements-win.txt
pip install --no-deps mediapipe==1.0.1
pip install matplotlib          :: build-time only
pyinstaller AIImageCreate.spec --noconfirm
powershell -File packaging\make_zip.ps1
```

## Licences

- **App code:** MIT (see `LICENSE`)
- **Models:** see `THIRD_PARTY.md` — CreativeML OpenRAIL-M for the UNet and the safety checker, MIT / Apache-2.0 / BSD for the rest
- **Qt / PySide6:** LGPLv3 (dynamically linked)

Don’t use this app for illegal, harmful or non-consensual content. The safety filter is always on, but it isn’t perfect — you’re responsible for what you create.
