# ImageTaggerGUI（旧名: PixAI Tagger ONNX GUI）
## [English](readme/readme_en.md) [简体中文](readme/readme_zh_CN.md) [繁體中文](readme/readme_zh_TW.md) [Русский](readme/readme_ru.md)

ローカル環境にある大量の画像に対して、高速かつ正確なタグやキャプションを自動生成するGUIツールです。直感的な操作で、データセットの整理や管理を劇的に効率化します。複数のローカルtagger／captionerモデルに加え、Gemini・OpenAI・Claude・Grok・ローカルVLMなどに対応しています。

|![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/main_window_01.webp)|![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/main_window_02.webp)|
|:-:|:-:|

![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/gridview_01.webp)

## 概要 (Overview)

**ImageTaggerGUI** は、[PixAI](https://pixai.art/) が開発した画像タグ付けモデルの[ONNX版](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx)をはじめとする複数のローカルモデルと、任意のネットワークVLMを利用し、画像にタグやキャプションを付与します。

本プロジェクトは PixAI Tagger の GUI（旧名 PixAI Tagger ONNX GUI）として始まり、現在は複数のローカルモデルとネットワーク VLM に対応しています。

PixAI Taggerは13,000以上の豊富なタグに対応しており、一般的なタグ付けモデル（例：wd-taggerの約10,000タグ）と比較して有利です。このツールはその性能を最大限に引き出し、あなたの画像管理をサポートするために開発されました。

画像の自動タグ付けはもちろん、タグの閲覧、個別編集、そして強力な一括編集機能（タグの追加・削除）を備えています。

## インストールと実行 (Installation & Usage)

1.  リリースページから最新の `ImageTaggerGUI-vX.X.X.zip` をダウンロードします。
2.  zipファイルを好きな場所に解凍します。
3.  中の `ImageTaggerGUI.exe` を実行します。
4.  v1.7.0 以前から更新する場合は、同じフォルダに上書き展開すれば `config.ini`・モデル・GPU コンポーネント・保存済み API キーはそのまま使えます。旧 `pixai_tagger_gui.exe` は削除して構いません。

## 使い方 (How to Use)
画像をWクリックしてホイールやドラッグすれば大体OK、直感的操作で動かせます。
3x3表示時でも画像Wクリックとホイールとドラッグは有効です。ctrl+ホイールで画像を前後します。

1.  **フォルダの指定**: `参照`ボタンか、画像フォルダのドラッグ＆ドロップで、対象の画像群を読み込みます。
2.  **モデルの準備 (初回のみ)**: `TAG`ボタンの代わりに`ダウンロード開始`と表示されたら、クリックしてモデルをダウンロードします。
3.  **タグ付け実行**: `TAG`ボタンを押すと、フォルダ内の全画像に対してタグ付け処理が始まります。既存の`.txt`ファイルがあれば、上書きを確認するダイアログが表示されます。
4.  **結果の確認と編集**:
    - 左のリストで画像を選ぶと、中央に画像、右にその画像のタグが表示されます。タグ入力欄ではctrl+上下キーで画像を前後できます。
    - 不要なタグは、右側のタグボタンをクリックして削除します。
    - 下部にある「一括タグ」セクションでは、フォルダ全体にまたがるタグの追加・削除が可能です。

## 主な機能 (Features)

### 1. 直感的で快適なUI
- **ドラッグ＆ドロップ対応**: 画像フォルダ、単体の画像ファイル、さらにはタグが記載された`.txt`ファイルまで、ウィンドウに直接ドラッグ＆ドロップするだけで即座に読み込めます。
- **軽快な画像ビューア**:
    - リストから画像を選ぶだけの簡単操作。
    - 画像をダブルクリックすれば、自由にリサイズ・移動可能な拡大ウィンドウが表示されます。
    - マウスホイールやキーボード（矢印キー, WASDなど）で、サクサクと次々に画像を切り替えて確認できます。

### 2. パワフルな自動タグ付け
- **高速なONNX Runtime**: CPUでも軽快に動作するONNXモデルを採用。大量の画像もストレスなく処理します。
- **モデル自動ダウンロード**: 選択したモデルと必要なメタデータを、設定された配布元からボタン一つでダウンロードできます。面倒な手作業は不要です。
- **GPU対応（NVIDIA・任意）**: NVIDIA GPUがあると、起動時の確認ダイアログからGPUコンポーネント（2GB前後）を一度ダウンロードし、アプリを再起動すればtaggerをCUDAで実行できます（実測RTX 4070で約28倍高速）。ダウンロードしなくても、また非NVIDIA環境でも、従来どおりCPUで動作します。キャプション（Florence-2）は量子化モデルの都合でCPUのままです。切り替えは`config.ini`の`[Behavior] onnx_device`（`auto`／`cpu`／`cuda`）。

### 3. 柔軟で高度なタグ編集
- **個別編集**:
    - 画像の既存タグをボタンとして一覧表示。不要なタグはボタンをクリックするだけで削除できます。
    - 新しいタグ（カンマ区切りで複数可）を簡単に追加できます。
- **強力な一括編集**:
    - フォルダ内の全`.txt`ファイルからタグを集計し、出現頻度順に表示。
    - ボタンクリック一つで、**全ファイルから特定のタグを一括で削除**できます。
    - 指定したタグを、**全ファイルの先頭または末尾に一括で追加**することも可能です。
- **グリッド編集ビュー**:
    - `3x3`ボタンからグリッドビューに移行。複数の画像を一度に俯瞰しながら、タグの確認や編集ができます。

### 4. 細やかなカスタマイズ性
- **タグ生成の調整**: モデルが対応するカテゴリごとに、タグを生成する**しきい値**や**最大タグ数**を直感的に調整できます。
- **設定の自動保存**: ウィンドウサイズや各設定値は、アプリ終了時に`config.ini`へ自動保存。次回も同じ環境で作業を再開できます。
- **多言語対応**: 日本語、英語、フランス語、ドイツ語、スペイン語、ロシア語、簡体字中国語、繁体字中国語、韓国語をサポート。OSの言語設定に応じてUIが自動で切り替わります。
- **新バージョン通知**: 起動時に新しいバージョンがないか確認し、あればダウンロードページへの案内を表示します（通知のみ、自動ダウンロードや自己更新は行いません）。`config.ini`の`[Behavior] update_check = dismissed`で無効化できます。

### 5. VLMキャプション（任意）
![](https://raw.githubusercontent.com/wai55555/ImageTaggerGUI/refs/heads/main/sample/vlm_setting_01.webp)

- **ネットワークVLMによる自然言語キャプション**: モデル選択欄の隣にある「VLM接続を使う」チェックを入れると、生成をローカルモデルからVLM（Vision-Language Model）へ切り替え、各画像に詳細な英語キャプションを付けます。学習用データセットの説明文に向いています。既定はOFFで、通常のタグ付けには影響しません。
- **内蔵サービスと同一モデルフォールバック**: Gemini API・OpenRouter・Cloudflare・Groq・NVIDIA NIM・Hugging Face・Vercel AI Gateway・OpenAI・Anthropic・xAI Grokを利用できます。あるサービスが拒否・レート制限された場合、*同じ*モデルを提供する次のサービスへ自動で切り替えます。別モデルへ勝手に乗り換えることはありません。
- **カスタム接続**: Ollama・LM Studio・llama.cpp・vLLM などローカルサーバーを含む、OpenAI 互換エンドポイントを追加できます。
- **キーは `config.ini` に置きません**: VLM設定ダイアログでAPIキーを登録すると、実リクエスト1回で検証し、OSのキーリング（または `.env` ファイル / 環境変数）に保存します。
- **選択した経路を使用**: 有効化して並べた認証済み経路だけを順に試します。API料金・無料枠・レート制限は各サービスの条件に従い、アプリはどの経路も無料とは判定しません。詳細度・文数・キャラクター名の扱い・Markdownを調整でき、生成キャプションは既存`.txt`と組み合わせられます（前に追加／後に追加／上書き。タグ出力と同じ）。


## ~~実装予定 (To Be Released)~~
- ~~GPU利用（onnxruntime-directml、AMD/Intel等の汎用GPU向け）~~
  → NVIDIA向け`onnxruntime-gpu`（CUDA）で実装（上記「GPU対応」参照）。directmlとCUDAは排他関係のため、両対応には別ビルドが必要になり見送り。
- ~~バッチ処理対応による高速化~~
  → ONNX Runtimeではオーバーヘッドが発生し、バッチ処理による高速化効果は得られなかった。

## ライセンス (License)

このプロジェクトは **LGPLv3** と **Apache License 2.0** のライセンスの下で公開されています。

## 謝辞 (Acknowledgements)

- このツールは [PixAI](https://pixai.art/) 様によってトレーニングされた素晴らしいタグ付けモデルを利用しています。Pixai Taggerが公開されていなければ、このアプリケーションは生まれませんでした。心から感謝申し上げます。本プロジェクトは PixAI Tagger の GUI（PixAI Tagger ONNX GUI）として始まりました。対応モデルが増えたため ImageTaggerGUI に改名しましたが、出発点が PixAI Tagger であることに変わりはありません。
- ONNXモデルは [deepghs](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx)様 によってHugging Faceで公開されているものを利用しています。ありがとうございます。
- 追加対応しているモデルとその ONNX を公開してくださっている作者の皆さまにも感謝します。 [SmilingWolf](https://huggingface.co/SmilingWolf)（WD tagger系）、[Misaka41Z](https://huggingface.co/Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2)（wd-eva02 2026 Canary ONNX）、[Camais03](https://huggingface.co/Camais03/camie-tagger-v2)（camie-tagger v2）、[cella110n](https://huggingface.co/cella110n/cl_tagger)（cl_tagger）、[Grio43](https://huggingface.co/Grio43/OppaiOracle)（OppaiOracle）、および Florence-2（Microsoft）の ONNX 一式を公開する [onnx-community](https://huggingface.co/onnx-community/Florence-2-base-ft) の各氏。各モデルのライセンスは配布元に従います。
