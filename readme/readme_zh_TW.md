# PixAI Tagger ONNX GUI
此應用程式是一個GUI工具，用於為本地的大量圖像自動產生快速準確的標籤和描述文字。透過直觀的操作，它極大地簡化了資料集的組織和管理。除了多種本地標籤和描述模型外，還支援 Gemini、OpenAI、Claude、Groq、本地 VLM 以及其他相容服務。

|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_01.png)|![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/main_window_02.jpg)|
|:-:|:-:|

![](https://raw.githubusercontent.com/wai55555/PixaiTaggerOnnxGui/refs/heads/main/sample/gridview_01.jpg)

## 概述 (Overview)

**PixAI Tagger ONNX GUI** 使用多種本地模型（包括 [PixAI](https://pixai.art/) 開發的圖像標註模型的 [ONNX 版本](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx)）以及可選的網路 VLM，為圖像產生標籤和描述文字。

原始的 PixAI Tagger 支援超過 13,000 個豐富的標籤，與常見的標註模型（例如 wd-tagger 的約 10,000 個標籤）相比具有優勢。此工具旨在最大限度地發揮其性能並支持您的圖像管理。

它具有自動圖像標註、標籤瀏覽、單獨編輯以及強大的批量編輯功能（添加/刪除特定標籤）。

## 安裝與使用 (Installation & Usage)

1.  從發布頁面下載最新的 `PixaiTaggerOnnxGui-vX.X.X.zip`。
2.  將 zip 文件解壓縮到您喜歡的位置。
3.  運行其中的 `pixai_tagger_gui.exe`。

## 如何使用 (How to Use)
雙擊圖像並使用滾輪或拖動來移動它。可以進行直觀的操作。
在顯示 3x3 時，雙擊、滾輪和拖動也有效。Ctrl+滾輪可以前後移動圖像。

1.  **指定資料夾**: 點擊 `瀏覽` 按鈕或拖放圖像資料夾來載入目標圖像組。
2.  **準備模型 (僅限首次)**: 如果顯示 `開始下載` 而不是 `TAG` 按鈕，請點擊它以下載模型。
3.  **執行標註**: 按下 `TAG` 按鈕開始對資料夾中的所有圖像進行標註處理。如果找到現有的 `.txt` 文件，將出現對話框確認是否覆蓋。
4.  **檢查和編輯結果**:
    -   從左側列表中選擇圖像，在中心顯示圖像，右側顯示其標籤。在標籤輸入欄位中，您可以使用 Ctrl+上下鍵前後導航圖像。
    -   透過點擊右側的標籤按鈕刪除不需要的標籤。
    -   底部 `批量標籤` 部分允許在整個資料夾中添加/刪除標籤。

## 主要功能 (Features)

### 1. 直觀舒適的使用者介面
-   **拖放支援**: 圖像資料夾、單個圖像文件，甚至包含標籤的 `.txt` 文件都可以透過直接拖放到視窗中立即載入。
-   **輕量級圖像檢視器**:
    -   只需從列表中選擇圖像即可輕鬆操作。
    -   雙擊圖像可顯示可自由調整大小和移動的放大視窗。
    -   使用滑鼠滾輪或鍵盤（箭頭鍵、WASD 等）快速切換圖像進行查看。

### 2. 強大的自動標註
-   **高速 ONNX Runtime**: 採用 ONNX 模型，即使在 CPU 上也能流暢運行。輕鬆處理大量圖像。
-   **自動模型下載**: 只需點擊一個按鈕，即可從設定的來源下載所選模型及其必要的中繼資料。無需繁瑣的手動設定。
-   **GPU 加速（NVIDIA，可選）**: 如果偵測到 NVIDIA GPU，只需在啟動提示中一次性下載 GPU 元件（約 2GB），下載完成後重新啟動應用程式，即可讓標註器在 CUDA 上運行（RTX 4070 實測約快 28 倍）。不下載，或在非 NVIDIA 裝置上，應用程式仍會像以前一樣在 CPU 上正常運作。由於使用量化模型，圖像描述（Florence-2）一律在 CPU 上運行。可透過 `config.ini` 中的 `[Behavior] onnx_device`（`auto` / `cpu` / `cuda`）切換。

### 3. 靈活高級的標籤編輯
-   **單獨編輯**:
    -   圖像的現有標籤以按鈕形式顯示。只需點擊按鈕即可刪除不需要的標籤。
    -   可以輕鬆添加新標籤（允許使用逗號分隔的多個標籤）。
-   **強大的批量編輯**:
    -   聚合資料夾中所有 `.txt` 文件中的標籤，並按頻率順序顯示。
    -   只需點擊一個按鈕，即可**從所有文件中批量刪除特定標籤**。
    -   還可以**將指定標籤批量添加到所有文件的開頭或結尾**。
-   **網格編輯檢視**:
    -   從 `3x3` 按鈕切換到網格檢視。同時查看多張圖像，確認和編輯標籤。

### 4. 詳細自定義
-   **標籤生成調整**: 可針對模型支援的各個類別直觀調整標籤生成**閾值**和**最大標籤數**。
-   **自動設定保存**: 視窗大小和所有設定在應用程式退出時自動保存到 `config.ini`。下次在相同的環境中繼續工作。
-   **多語言支援**: 支援英語、日語、法語、德語、西班牙語、俄語、簡體中文、繁體中文和韓語。使用者介面會根據作業系統語言設定自動切換。
-   **新版本通知**: 啟動時檢查是否有新版本，如有則顯示下載頁面連結（僅通知，不會自動下載或自我更新）。可透過`config.ini`中的`[Behavior] update_check = dismissed`停用。

### 5. VLM 圖像描述（可選）
-   **透過網路 VLM 產生自然語言描述**: 啟用模型選擇框旁的「使用 VLM 連線」後，將使用視覺語言模型為每張圖像產生詳細的英文描述。預設關閉，不會影響本地標籤功能。
-   **內建服務與同模型故障轉移**: 支援 Gemini API、OpenRouter、Cloudflare、Groq、NVIDIA NIM、Hugging Face、Vercel AI Gateway、OpenAI、Anthropic 和 xAI Grok。服務拒絕請求或受到速率限制時，只會嘗試提供*同一模型*的下一條連線，不會在未告知的情況下切換到其他模型。
-   **自訂連線**: 可以新增任何相容 OpenAI API 的端點，包括 Ollama、LM Studio、llama.cpp 和 vLLM 等本地伺服器。
-   **API 金鑰不會寫入 `config.ini`**: 金鑰會透過真實請求進行檢查，並儲存到作業系統金鑰圈，或從 `.env` 檔案／環境變數讀取。
-   **僅使用您選擇的連線**: 只會依序嘗試已啟用並完成驗證的連線。API 費用、免費額度和速率限制以各服務商的條款為準；本應用程式不會將任何連線判定為免費。可調整詳細程度、句數、角色名稱和 Markdown，並可將產生內容加入現有 `.txt` 的前面、後面或直接覆寫。

## ~~計畫中 (Planned)~~
-   ~~GPU 支援（onnxruntime-directml，面向 AMD/Intel 等通用 GPU）~~
    → 改為透過 `onnxruntime-gpu`（CUDA）為 NVIDIA 實作（見上文「GPU 加速」）。DirectML 與 CUDA 互斥，若要同時支援兩者需要獨立建置，故暫不實作。
-   ~~批次處理加速~~
    → ONNX Runtime 每次呼叫的開銷抵銷了批次處理帶來的效益，因此未採用。

## 授權 (License)

本專案根據 **LGPLv3** 和 **Apache License 2.0** 授權發布。

## 致謝 (Acknowledgements)

- 此工具利用了 [PixAI](https://pixai.art/) 訓練的優秀標註模型。如果沒有 Pixai Tagger 的公開發布，此應用程式將無法誕生。衷心感謝。
- 使用的 ONNX 模型由 [deepghs](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) 在 Hugging Face 上公開。謝謝。
- 同樣感謝公開了其他已支援模型及其 ONNX 版本的作者們 — [SmilingWolf](https://huggingface.co/SmilingWolf)（WD tagger 系列）、[Misaka41Z](https://huggingface.co/Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2)（wd-eva02 2026 Canary 的 ONNX）、[Camais03](https://huggingface.co/Camais03/camie-tagger-v2)（camie-tagger v2）、[cella110n](https://huggingface.co/cella110n/cl_tagger)（cl_tagger）、[Grio43](https://huggingface.co/Grio43/OppaiOracle)（OppaiOracle），以及公開 Florence-2（Microsoft）ONNX 全套檔案的 [onnx-community](https://huggingface.co/onnx-community/Florence-2-base-ft)。各模型均遵循其上游授權條款。
