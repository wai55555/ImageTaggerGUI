# PixAI Tagger ONNX GUI
此应用程序是一个GUI工具，用于为本地的大量图像自动生成快速准确的标签和描述文本。通过直观的操作，它极大地简化了数据集的组织和管理。除了多种本地标签和描述模型外，还支持 Gemini、OpenAI、Claude、Groq、本地 VLM 以及其他兼容服务。

|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_01.png)|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_02.jpg)|
|:-:|:-:|

![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/gridview_01.jpg)

## 概述 (Overview)

**PixAI Tagger ONNX GUI** 使用多种本地模型（包括 [PixAI](https://pixai.art/) 开发的图像标注模型的 [ONNX 版本](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx)）以及可选的网络 VLM，为图像生成标签和描述文本。

原始的 PixAI Tagger 支持超过 13,000 个丰富的标签，与常见的标注模型（例如 wd-tagger 的约 10,000 个标签）相比具有优势。此工具旨在最大限度地发挥其性能并支持您的图像管理。

它具有自动图像标注、标签浏览、单独编辑以及强大的批量编辑功能（添加/删除特定标签）。

## 安装与使用 (Installation & Usage)

1.  从发布页面下载最新的 `PixaiTaggerOnnxGui-vX.X.X.zip`。
2.  将 zip 文件解压到您喜欢的位置。
3.  运行其中的 `pixai_tagger_gui.exe`。

## 如何使用 (How to Use)
双击图像并使用滚轮或拖动来移动它。可以进行直观的操作。
在显示 3x3 时，双击、滚轮和拖动也有效。Ctrl+滚轮可以前后移动图像。

1.  **指定文件夹**: 点击 `浏览` 按钮或拖放图像文件夹来加载目标图像组。
2.  **准备模型 (仅限首次)**: 如果显示 `开始下载` 而不是 `TAG` 按钮，请点击它以下载模型。
3.  **执行标注**: 按下 `TAG` 按钮开始对文件夹中的所有图像进行标注处理。如果找到现有的 `.txt` 文件，将出现对话框确认是否覆盖。
4.  **检查和编辑结果**:
    -   从左侧列表中选择图像，在中心显示图像，右侧显示其标签。在标签输入字段中，您可以使用 Ctrl+上下键前后导航图像。
    -   通过点击右侧的标签按钮删除不需要的标签。
    -   底部 `批量标签` 部分允许在整个文件夹中添加/删除标签。

## 主要功能 (Features)

### 1. 直观舒适的用户界面
-   **拖放支持**: 图像文件夹、单个图像文件，甚至包含标签的 `.txt` 文件都可以通过直接拖放到窗口中立即加载。
-   **轻量级图像查看器**:
    -   只需从列表中选择图像即可轻松操作。
    -   双击图像可显示可自由调整大小和移动的放大窗口。
    -   使用鼠标滚轮或键盘（箭头键、WASD 等）快速切换图像进行查看。

### 2. 强大的自动标注
-   **高速 ONNX Runtime**: 采用 ONNX 模型，即使在 CPU 上也能流畅运行。轻松处理大量图像。
-   **自动模型下载**: 只需单击一个按钮，即可从配置的来源下载所选模型及其必要的元数据。无需繁琐的手动设置。
-   **GPU 加速（NVIDIA，可选）**: 如果检测到 NVIDIA GPU，只需在启动提示中一次性下载 GPU 组件（约 2GB），下载完成后重启应用程序，即可让标注器在 CUDA 上运行（RTX 4070 实测约快 28 倍）。不下载，或在非 NVIDIA 设备上，应用仍会像以前一样在 CPU 上正常运行。由于使用量化模型，图像描述（Florence-2）始终在 CPU 上运行。可通过 `config.ini` 中的 `[Behavior] onnx_device`（`auto` / `cpu` / `cuda`）切换。

### 3. 灵活高级的标签编辑
-   **单独编辑**:
    -   图像的现有标签以按钮形式显示。只需单击按钮即可删除不需要的标签。
    -   可以轻松添加新标签（允许使用逗号分隔的多个标签）。
-   **强大的批量编辑**:
    -   聚合文件夹中所有 `.txt` 文件中的标签，并按频率顺序显示。
    -   只需单击一个按钮，即可**从所有文件中批量删除特定标签**。
    -   还可以**将指定标签批量添加到所有文件的开头或结尾**。
-   **网格编辑视图**:
    -   从 `3x3` 按钮切换到网格视图。同时查看多张图像，确认和编辑标签。

### 4. 详细自定义
-   **标签生成调整**: 可针对模型支持的各个类别直观调整标签生成**阈值**和**最大标签数**。
-   **自动设置保存**: 窗口大小和所有设置在应用程序退出时自动保存到 `config.ini`。下次在相同的环境中继续工作。
-   **多语言支持**: 支持英语、日语、法语、德语、西班牙语、俄语、简体中文、繁体中文和韩语。用户界面根据操作系统语言设置自动切换。

### 5. VLM 图像描述（可选）
-   **通过网络 VLM 生成自然语言描述**: 启用模型选择框旁的“使用 VLM 连接”后，将使用视觉语言模型为每张图像生成详细的英文描述。默认关闭，不会影响本地标签功能。
-   **内置服务与同模型故障转移**: 支持 Gemini API、OpenRouter、Cloudflare、Groq、NVIDIA NIM、Hugging Face、Vercel AI Gateway、OpenAI、Anthropic 和 xAI Grok。服务拒绝请求或受到速率限制时，只会尝试提供*同一模型*的下一条连接，不会静默切换到其他模型。
-   **自定义连接**: 可以添加任何兼容 OpenAI API 的端点，包括 Ollama、LM Studio、llama.cpp 和 vLLM 等本地服务器。
-   **API 密钥不会写入 `config.ini`**: 密钥会通过真实请求进行检查，并保存到操作系统密钥环，或从 `.env` 文件／环境变量读取。
-   **仅使用您选择的连接**: 只会依次尝试已启用并完成认证的连接。API 费用、免费额度和速率限制以各服务商的条款为准；本应用不会将任何连接判定为免费。可调整详细程度、句数、角色名称和 Markdown，并可将生成内容添加到现有 `.txt` 的前面、后面或直接覆盖。

## ~~计划中 (Planned)~~
-   ~~GPU 支持（onnxruntime-directml，面向 AMD/Intel 等通用 GPU）~~
    → 改为通过 `onnxruntime-gpu`（CUDA）为 NVIDIA 实现（见上文「GPU 加速」）。DirectML 与 CUDA 互斥，若要同时支持两者需要单独构建，故暂不实现。
-   ~~批处理加速~~
    → ONNX Runtime 每次调用的开销抵消了批处理带来的收益，因此未采用。

## 许可证 (License)

本项目根据 **LGPLv3** 和 **Apache License 2.0** 许可证发布。

## 致谢 (Acknowledgements)

- 此工具利用了 [PixAI](https://pixai.art/) 训练的优秀标注模型。如果没有 Pixai Tagger 的公开发布，此应用程序将无法诞生。衷心感谢。
- 使用的 ONNX 模型由 [deepghs](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) 在 Hugging Face 上公开。谢谢。
- 同样感谢公开了其它已支持模型及其 ONNX 版本的作者们 — [SmilingWolf](https://huggingface.co/SmilingWolf)（WD tagger 系列）、[Misaka41Z](https://huggingface.co/Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2)（wd-eva02 2026 Canary 的 ONNX）、[Camais03](https://huggingface.co/Camais03/camie-tagger-v2)（camie-tagger v2）、[cella110n](https://huggingface.co/cella110n/cl_tagger)（cl_tagger）、[Grio43](https://huggingface.co/Grio43/OppaiOracle)（OppaiOracle），以及公开 Florence-2（Microsoft）ONNX 全套文件的 [onnx-community](https://huggingface.co/onnx-community/Florence-2-base-ft)。各模型均遵循其上游许可协议。
