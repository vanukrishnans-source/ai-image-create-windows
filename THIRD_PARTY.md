# Third-party notices — AI Image Create for Windows

## Application stack
| Component | Licence | Notes |
|---|---|---|
| Python 3.13 | PSF | Bundled by PyInstaller |
| ONNX Runtime + DirectML EP | MIT | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) |
| PySide6 / Qt 6 (Essentials) | LGPLv3 | Dynamic linking; Qt source available from The Qt Company |
| MediaPipe Face Landmarker | Apache-2.0 | [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) |
| NumPy | BSD-3-Clause | |
| Pillow | HPND-style | |
| regex | Apache-2.0 | |
| PyInstaller | GPLv2 + exception / Apache-2.0 | Bootloader exception applies |

## AI models (downloaded on first run from [ai-image-create-models / models-v1](https://github.com/vanukrishnans-source/ai-image-create-models/releases/tag/models-v1))
| Asset | Origin | Licence |
|---|---|---|
| `unet` — LCM-Dreamshaper-v7 | [SimianLuo/LCM_Dreamshaper_v7](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7) (based on Lykon/dreamshaper-7 / Stable Diffusion) | **CreativeML OpenRAIL-M** (use restrictions apply) |
| `text_encoder` — CLIP ViT-L/14 | OpenAI CLIP / LAION | MIT |
| `vae_decoder` — SD VAE | Stability AI | MIT |
| `taesd_encoder` — TAESD | madebyollin/taesd | MIT |
| `adapter_canny` — T2I-Adapter canny | TencentARC / Diffusers | Apache-2.0 |
| `upscaler` — Real-ESRGAN general x4v3 | xinntao/Real-ESRGAN | BSD-3-Clause |
| `safety` — Stable Diffusion safety checker | CompVis / Stability AI | **CreativeML OpenRAIL-M** |
| Face Landmarker task | MediaPipe | Apache-2.0 |

### CreativeML OpenRAIL-M (summary)
You may use, redistribute and modify the model under the OpenRAIL-M licence. You **must not** use it for illegal, harmful, or non-consensual purposes (full list in the licence text attached to the Hugging Face / GitHub model cards). The safety filter shipped with this app is always on and cannot be disabled.

## Test photos (bundled for CI only)
See `testdata/SOURCES.md`.
