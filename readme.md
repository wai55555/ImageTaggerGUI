# ImageTaggerGUI (formerly PixAI Tagger ONNX GUI)
## [日本語](readme/readme_ja.md) [简体中文](readme/readme_zh_CN.md) [繁體中文](readme/readme_zh_TW.md) [Русский](readme/readme_ru.md)
This application is a GUI tool for automatically generating fast and accurate tags and captions for large collections of local images. It streamlines dataset organization and management with intuitive controls. In addition to multiple local tagger and captioner models, it supports Gemini, OpenAI, Claude, Groq, local VLMs, and other compatible services.

|![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/main_window_01.webp)|![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/main_window_02.webp)|
|:-:|:-:|

![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/gridview_01.webp)

## Overview

**ImageTaggerGUI** uses multiple local models—including the [ONNX version](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) of the image tagging model developed by [PixAI](https://pixai.art/)—and optional network VLMs to generate tags and captions for images.

This project started as a GUI for PixAI Tagger (formerly PixAI Tagger ONNX GUI) and now supports multiple local models and network VLMs.

The original PixAI Tagger supports over 13,000 rich tags, offering an advantage compared to common tagging models (e.g., wd-tagger's ~10,000 tags). This tool was developed to maximize its performance and support your image management.

It features automatic image tagging, tag browsing, individual editing, and powerful bulk editing functions (adding/deleting specific tags).

## Installation & Usage

1.  Download the latest `ImageTaggerGUI-vX.X.X.zip` from the release page.
2.  Unzip the file to your preferred location.
3.  Run `ImageTaggerGUI.exe` inside.
4.  Upgrading from v1.7.0 or earlier: extract over the same folder and your `config.ini`, models, GPU components and saved API keys carry over. The old `pixai_tagger_gui.exe` can be deleted.

## How to Use
Double-click an image and use the wheel or drag to move it around. Intuitive operation is possible.
Double-clicking, wheel, and dragging are also effective when displaying 3x3. Ctrl+wheel moves images back and forth.

1.  **Specify Folder**: Load target image groups by clicking the `Browse` button or by dragging and dropping an image folder.
2.  **Prepare Model (First time only)**: If `Download Start` is displayed instead of `TAG` button, click it to download the model.
3.  **Execute Tagging**: Press the `TAG` button to start tagging all images in the folder. If existing `.txt` files are found, a dialog will appear to confirm overwriting.
4.  **Check and Edit Results**:
    -   Select an image from the left list to display the image in the center and its tags on the right. In the tag input field, you can navigate images back and forth with Ctrl+Up/Down keys.
    -   Delete unnecessary tags by clicking the tag button on the right.
    -   The "Bulk Tags" section at the bottom allows adding/deleting tags across the entire folder.

## Features

### 1. Intuitive and Comfortable UI
-   **Drag & Drop Support**: Image folders, individual image files, and even `.txt` files containing tags can be loaded instantly by dragging and dropping them directly into the window.
-   **Lightweight Image Viewer**:
    -   Easy operation by simply selecting an image from the list.
    -   Double-click an image to display an enlarged window that can be freely resized and moved.
    -   Quickly switch between images using the mouse wheel or keyboard (arrow keys, WASD, etc.).

### 2. Powerful Automatic Tagging
-   **High-speed ONNX Runtime**: Employs an ONNX model that operates smoothly even on CPUs. Processes large volumes of images without stress.
-   **Automatic Model Download**: Download the selected model and its required metadata from the configured source with one click. No tedious manual setup is required.
-   **GPU Acceleration (NVIDIA, optional)**: With an NVIDIA GPU, a one-time download of the GPU components (~2GB) from the startup prompt, followed by a restart, lets the tagger run on CUDA from then on (~28x faster, measured on an RTX 4070). Skip the download - or run on a non-NVIDIA machine - and it keeps working on the CPU exactly as before. Captioning (Florence-2) always runs on the CPU due to its quantized model. Toggle with `[Behavior] onnx_device` in `config.ini` (`auto` / `cpu` / `cuda`).

### 3. Flexible and Advanced Tag Editing
-   **Individual Editing**:
    -   Existing tags for an image are displayed as buttons. Unnecessary tags can be deleted by simply clicking the button.
    -   New tags (multiple separated by commas are allowed) can be easily added.
-   **Powerful Bulk Editing**:
    -   Aggregates tags from all `.txt` files in a folder and displays them in order of frequency.
    -   **Bulk delete specific tags from all files** with a single button click.
    -   It is also possible to **bulk add specified tags to the beginning or end of all files**.
-   **Grid Editing View**:
    -   Transition to grid view from the `3x3` button. Confirm and edit tags while viewing multiple images at once.

### 4. Detailed Customization
-   **Tag Generation Adjustment**: Intuitively adjust model-specific **thresholds** and **maximum tag counts** for supported categories.
-   **Automatic Settings Save**: Window size and other non-secret settings are automatically saved to `config.ini` upon app exit (API keys are never written there - see VLM Captioning below). Resume work in the same environment next time.
-   **Multi-language Support**: Supports English, Japanese, French, German, Spanish, Russian, Simplified Chinese, Traditional Chinese, and Korean. The UI automatically switches according to the OS language settings.
-   **New Version Notification**: Checks for a new version on startup and, if one is out, shows a link to the download page (notification only - no automatic download or self-update). Can be disabled with `[Behavior] update_check = dismissed` in `config.ini`.

### 5. VLM Captioning (Optional)
![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/vlm_setting_01.webp)

-   **Natural-language captions via a networked VLM**: Next to the model selector, a "Use VLM connection" checkbox switches generation from the local model to a Vision-Language Model that writes a detailed English caption for each image - handy as training-dataset descriptions. Off by default; local tagging is unchanged.
-   **Built-in services with same-model fallback**: Gemini API, OpenRouter, Cloudflare, Groq, NVIDIA NIM, Hugging Face, Vercel AI Gateway, OpenAI, Anthropic, and xAI Grok are supported. If one service refuses or is rate limited, the next service offering the *same* model is tried automatically—it never silently switches to a different model. <!-- Mistral/Pixtral is commented out because its caption quality is currently too limited. -->
-   **Custom connections**: Add any OpenAI-compatible endpoint, including local servers such as Ollama, LM Studio, llama.cpp or vLLM.
-   **Keys stay out of `config.ini`**: Register an API key from the VLM settings dialog; it is checked with one real request and stored in the OS keyring (or read from a `.env` file / environment variable).
-   **Routes follow your selection**: Only enabled, authenticated routes you ordered are tried; provider billing and metered usage follow each service's terms. No route is treated as free. Detail level, sentence count, character-name policy and Markdown are adjustable, and captions combine with an existing `.txt` (prepend / append / overwrite) just like tagging output.

## ~~Planned~~
-   ~~GPU support (onnxruntime-directml, for AMD/Intel and other general-purpose GPUs)~~
    → Implemented for NVIDIA instead, via `onnxruntime-gpu` (CUDA) - see "GPU Acceleration" above. DirectML and CUDA are mutually exclusive, so supporting both would need a separate build; shelved.
-   ~~Batch processing for extra speed~~
    → ONNX Runtime's per-call overhead outweighed the gains, so this was dropped.

## License

This project is released under the **LGPLv3** and **Apache License 2.0** licenses.

## Acknowledgements

- This tool utilizes the excellent tagging model trained by [PixAI](https://pixai.art/). This application would not have been possible without the public release of Pixai Tagger. My heartfelt thanks. This project began as a GUI for PixAI Tagger (PixAI Tagger ONNX GUI). It was renamed ImageTaggerGUI as more models were added, but PixAI Tagger remains where it all started.
- The ONNX model used is publicly available on Hugging Face by [deepghs](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx). Thank you.
- Thanks also to the authors who publish the other supported models and their ONNX builds — [SmilingWolf](https://huggingface.co/SmilingWolf) (WD tagger family), [Misaka41Z](https://huggingface.co/Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2) (wd-eva02 2026 Canary ONNX), [Camais03](https://huggingface.co/Camais03/camie-tagger-v2) (camie-tagger v2), [cella110n](https://huggingface.co/cella110n/cl_tagger) (cl_tagger), [Grio43](https://huggingface.co/Grio43/OppaiOracle) (OppaiOracle), and [onnx-community](https://huggingface.co/onnx-community/Florence-2-base-ft) for the ONNX build of Florence-2 (Microsoft). Each model is used under its upstream license.
