from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from utils import log_dbg, GetString
from app_settings import AppSettings
from tagging_core import (
    _normalize_np_chw, BASE_DIR, ExistingFileMode, FileChange, OverwriteDecision,
    make_cpu_session_options, parse_existing_file_mode,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image
    import onnxruntime as ort  # type: ignore
    from tokenizers import Tokenizer  # type: ignore
else:
    try:
        import numpy as np
        from numpy.typing import NDArray
        from PIL import Image
        import onnxruntime as ort  # type: ignore
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        np = None
        NDArray = None
        Image = None
        ort = None
        Tokenizer = None

_get_string: GetString = lambda section, key, **kwargs: str(key)

@dataclass(frozen=True)
class CaptionerConfig:
    model_dir: Path
    image_size: int = 768
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    image_seq_length: int = 577
    bos_token_id: int = 0
    eos_token_id: int = 2
    pad_token_id: int = 1
    decoder_start_token_id: int = 2
    max_new_tokens: int = 200
    # Florence-2-base defaults. Custom captioner manifests can override these for
    # decoder graphs with a different KV-cache layout.
    decoder_layers: int = 6
    decoder_attention_heads: int = 12
    decoder_head_dim: int = 64
    tasks: dict[str, str] = field(default_factory=dict)
    default_task: str = "MORE_DETAILED_CAPTION"


def build_captioner_config(model_dir: Path, model_config: dict[str, Any]) -> CaptionerConfig:
    cap = model_config.get("captioner") if isinstance(model_config.get("captioner"), dict) else {}
    d = CaptionerConfig(model_dir=model_dir)
    decoder_layers = int(cap.get("decoder_layers", d.decoder_layers))
    attention_heads = int(cap.get("decoder_attention_heads", d.decoder_attention_heads))
    if decoder_layers <= 0 or attention_heads <= 0:
        raise ValueError("captioner decoder_layers and decoder_attention_heads must be positive")
    configured_head_dim = cap.get("decoder_head_dim")
    if configured_head_dim is None and cap.get("d_model") is not None:
        hidden_size = int(cap["d_model"])
        if hidden_size <= 0 or hidden_size % attention_heads:
            raise ValueError("captioner d_model must be positive and divisible by decoder_attention_heads")
        configured_head_dim = hidden_size // attention_heads
    head_dim = int(configured_head_dim if configured_head_dim is not None
                   else d.decoder_head_dim)
    if head_dim <= 0:
        raise ValueError("captioner decoder_head_dim must be positive")
    return CaptionerConfig(
        model_dir=model_dir,
        image_size=cap.get("image_size", d.image_size),
        normalize_mean=tuple(cap.get("normalize_mean", d.normalize_mean)),
        normalize_std=tuple(cap.get("normalize_std", d.normalize_std)),
        image_seq_length=cap.get("image_seq_length", d.image_seq_length),
        bos_token_id=cap.get("bos_token_id", d.bos_token_id),
        eos_token_id=cap.get("eos_token_id", d.eos_token_id),
        pad_token_id=cap.get("pad_token_id", d.pad_token_id),
        decoder_start_token_id=cap.get("decoder_start_token_id", d.decoder_start_token_id),
        max_new_tokens=cap.get("max_new_tokens", d.max_new_tokens),
        decoder_layers=decoder_layers,
        decoder_attention_heads=attention_heads,
        decoder_head_dim=head_dim,
        tasks=cap.get("tasks", {}) or d.tasks,
        default_task=cap.get("default_task", d.default_task),
    )


class Florence2Captioner:
    """
    Hand-rolled ONNX Runtime inference for Florence-2 (base-ft, int8/quantized), replacing
    the `transformers`/`optimum` generate() pipeline the reference implementation uses.
    Greedy decoding only - no beam search (design.md 8.2節, spec.md 7章).
    """

    def __init__(self, config: CaptionerConfig, get_string: GetString | None = None,
                 intra_op_num_threads: int = 0, onnx_device: str = "auto"):
        self.get_string = get_string if get_string else _get_string
        self.config = config

        if ort is None or Tokenizer is None or Image is None:
            raise ImportError(self.get_string("CaptionCore", "Required_Libraries_NotFound"))

        onnx_dir = config.model_dir / "onnx"
        # config.ini [Behavior] onnx_threads。0 なら None（ORT 既定 = 全コア）。
        so = make_cpu_session_options(intra_op_num_threads)
        if so is not None:
            log_dbg(f"Florence2Captioner: intra_op_num_threads capped at {so.intra_op_num_threads} ([Behavior] onnx_threads)")
        log_dbg(self.get_string("CaptionCore", "Info_Loading_Sessions", model_dir=str(config.model_dir)))

        model_files = ("vision_encoder_quantized.onnx", "embed_tokens_quantized.onnx",
                       "encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx")
        (self.vision_encoder, self.embed_tokens,
         self.encoder_model, self.decoder_model_merged) = self._load_sessions(
            [onnx_dir / name for name in model_files], so, onnx_device)
        self._decoder_output_names = [o.name for o in self.decoder_model_merged.get_outputs()]
        self.tokenizer = Tokenizer.from_file(str(config.model_dir / "tokenizer.json"))
        log_dbg(self.get_string("CaptionCore", "Info_Sessions_Loaded"))

    def _load_sessions(self, paths: list[Path], sess_options, onnx_device: str) -> list:
        """Florence-2 の 4 セッションは CPU 固定（onnx_device は現状無視）。

        Phase 4 実測（RTX 4070 / onnxruntime-gpu 1.23.2 / 2026-09-10）で、量子化(int8)の
        Florence-2 を CUDA EP に載せると **CPU より約4倍遅い**（CPU 1.7s / CUDA 6.9s、
        1キャプション）。原因は (1) int8 グラフに CUDA が対応できず大量の Memcpy ノードが
        挿入される（ORT 警告 "455 Memcpy nodes are added ..."）、(2) greedy 自己回帰
        デコードは 1 トークンごとの小さなカーネル起動で、GPU の起動オーバーヘッドが支配的。
        タガー（単発の重い畳み込み）は逆に CUDA で ~28倍速いので、そちらだけ GPU にする。
        非量子化(fp16)の Florence エクスポートを採用したら onnx_device を効かせる余地あり。
        """
        del onnx_device  # 明示的に未使用（fp16 エクスポート採用時に復活させる）
        return [ort.InferenceSession(str(p), sess_options=sess_options,
                                     providers=["CPUExecutionProvider"]) for p in paths]

    def _preprocess_image(self, image: "Image.Image") -> "NDArray[np.float32]":
        size = self.config.image_size
        resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        chw = arr.transpose((2, 0, 1))
        mean = np.asarray(self.config.normalize_mean, dtype=np.float32)
        std = np.asarray(self.config.normalize_std, dtype=np.float32)
        normalized = _normalize_np_chw(chw, mean, std)
        return np.expand_dims(normalized, axis=0)

    def generate(self, image: "Image.Image", task_prompt: str,
                 stop_checker: Callable[[], bool] | None = None) -> tuple[str, bool]:
        """
        1. vision_encoder(pixel_values) -> image_features
        2. embed_tokens(task_prompt tokens) -> text_embeds
        3. concat(image_features, text_embeds) -> encoder_model -> encoder_hidden_states
        4. greedy autoregressive decode with decoder_model_merged (KV cache)

        `stop_checker`, if given, is polled once per decoded token so a long generation
        on CPU responds to the Stop button without finishing the whole caption first.

        Returns (text, cancelled). `cancelled` is True when decoding was cut short by
        stop_checker - the text is then a truncated fragment and MUST NOT be written
        over the image's caption file.
        """
        pixel_values = self._preprocess_image(image)
        image_features = self.vision_encoder.run(None, {"pixel_values": pixel_values})[0]

        prompt_ids = self.tokenizer.encode(task_prompt).ids
        input_ids = np.array([prompt_ids], dtype=np.int64)
        text_embeds = self.embed_tokens.run(None, {"input_ids": input_ids})[0]

        combined_embeds = np.concatenate([image_features, text_embeds], axis=1).astype(np.float32)
        combined_attention_mask = np.ones((1, combined_embeds.shape[1]), dtype=np.int64)

        encoder_hidden_states = self.encoder_model.run(None, {
            "inputs_embeds": combined_embeds,
            "attention_mask": combined_attention_mask,
        })[0]

        generated_ids, cancelled = self._greedy_decode(encoder_hidden_states, combined_attention_mask, stop_checker)
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip(), cancelled

    def _greedy_decode(self, encoder_hidden_states: "NDArray[np.float32]", encoder_attention_mask: "NDArray[np.int64]",
                       stop_checker: Callable[[], bool] | None = None) -> tuple[list[int], bool]:
        batch_size = encoder_hidden_states.shape[0]
        current_token = self.config.decoder_start_token_id
        generated: list[int] = []

        # First pass: use_cache_branch=False, so the graph computes cross-attention KV
        # fresh from encoder_hidden_states. The past_key_values.* inputs are still
        # required by the graph even though unused on this branch, so feed zero-length
        # placeholders for all of them.
        past_kv: dict[str, "NDArray[np.float32]"] = {}
        empty = np.zeros((batch_size, self.config.decoder_attention_heads, 0,
                          self.config.decoder_head_dim), dtype=np.float32)
        for i in range(self.config.decoder_layers):
            past_kv[f"past_key_values.{i}.decoder.key"] = empty
            past_kv[f"past_key_values.{i}.decoder.value"] = empty
            past_kv[f"past_key_values.{i}.encoder.key"] = empty
            past_kv[f"past_key_values.{i}.encoder.value"] = empty

        use_cache_branch = np.array([False])

        for step in range(self.config.max_new_tokens):
            if stop_checker and stop_checker():
                return generated, True
            input_ids = np.array([[current_token]], dtype=np.int64)
            inputs_embeds = self.embed_tokens.run(None, {"input_ids": input_ids})[0]

            feed: dict[str, Any] = {
                "encoder_attention_mask": encoder_attention_mask,
                "encoder_hidden_states": encoder_hidden_states,
                "inputs_embeds": inputs_embeds,
                "use_cache_branch": use_cache_branch,
            }
            feed.update(past_kv)

            outputs = self.decoder_model_merged.run(self._decoder_output_names, feed)
            output_map = dict(zip(self._decoder_output_names, outputs))

            logits = output_map["logits"]
            next_token = int(np.argmax(logits[0, -1, :]))

            if next_token == self.config.eos_token_id:
                break
            generated.append(next_token)

            new_past_kv: dict[str, "NDArray[np.float32]"] = {}
            for i in range(self.config.decoder_layers):
                new_past_kv[f"past_key_values.{i}.decoder.key"] = output_map[f"present.{i}.decoder.key"]
                new_past_kv[f"past_key_values.{i}.decoder.value"] = output_map[f"present.{i}.decoder.value"]
                if step == 0:
                    # Cross-attention KV only needs to be computed once; reuse verbatim afterwards.
                    new_past_kv[f"past_key_values.{i}.encoder.key"] = output_map[f"present.{i}.encoder.key"]
                    new_past_kv[f"past_key_values.{i}.encoder.value"] = output_map[f"present.{i}.encoder.value"]
                else:
                    new_past_kv[f"past_key_values.{i}.encoder.key"] = past_kv[f"past_key_values.{i}.encoder.key"]
                    new_past_kv[f"past_key_values.{i}.encoder.value"] = past_kv[f"past_key_values.{i}.encoder.value"]
            past_kv = new_past_kv
            use_cache_branch = np.array([True])
            current_token = next_token

        return generated, False


def setup_captioner_from_settings(app_settings: AppSettings, get_string: GetString | None) -> tuple["Florence2Captioner | None", dict[str, Any]]:
    """Mirrors tagging_core.setup_tagger_from_settings, but for captioner (model_type="captioner") models."""
    _get_string_internal = get_string if get_string else _get_string
    try:
        import model_registry

        model_id = app_settings.model.model_id
        entry = model_registry.get_model_entry(model_id)
        if entry is None or entry.model_type != "captioner":
            log_dbg(f"setup_captioner_from_settings: model_id={model_id} is not a captioner entry.")
            return None, {}

        captioner_config = build_captioner_config(entry.model_dir, entry.config)
        settings_dict: dict[str, Any] = {
            "INPUT_DIR": Path(app_settings.paths.input_dir),
            "TASK": app_settings.caption.task,
            "EXISTING_FILE_MODE": parse_existing_file_mode(app_settings.behavior.existing_file_mode, _get_string_internal),
            "CAPTION_PLACEMENT": app_settings.caption.placement,
        }
        captioner = Florence2Captioner(captioner_config, get_string=_get_string_internal,
                                       intra_op_num_threads=app_settings.behavior.onnx_threads,
                                       onnx_device=app_settings.behavior.onnx_device)
        return captioner, settings_dict
    except Exception as e:
        log_dbg(f"Error during Captioner initialization: {type(e).__name__}: {e}")
        log_dbg(f"Traceback: {traceback.format_exc()}")
        return None, {}


def combine_caption(existing: str, caption: str, placement: str) -> str:
    """生成キャプションを既存内容と組み合わせる（2026-08-31 ユーザー要望）。

    danbooru タグ＋自然言語を1つの .txt に同居させるモデルがあるため、上書き以外に
    「前に追加」「後に追加」が要る。区切りは学習用キャプションの慣習に合わせて ", "。
    既存が空なら生成文のみ、生成文が空なら既存のみを返す。
    """
    existing = existing.strip().strip(",").strip()
    caption = caption.strip()
    if not existing:
        return caption
    if not caption:
        # 生成文が空なら、OVERWRITE でも既存を残す（空文字で潰さない）。docstring どおり。
        return existing
    if placement == "OVERWRITE":
        return caption
    if placement == "PREPEND":
        return f"{caption}, {existing}"
    return f"{existing}, {caption}"


def process_caption_loop(
    captioner: Florence2Captioner,
    settings: dict[str, Any],
    image_paths: list[Path],
    decision_resolver: Callable[[Path], OverwriteDecision] | None,
    log_gui: Callable[[str, str], None] | None,
    stop_checker: Callable[[], bool] | None,
    get_string: GetString | None,
    progress_cb: Callable[[int, int], None] | None = None,
    failed_paths: list[Path] | None = None,
) -> list[FileChange]:
    """Captioner counterpart of tagging_core.process_image_loop.

    既存ファイルの扱い（ASK / OVERWRITE / SKIP / APPEND）は「そのファイルに書くか」
    だけを決める。実際にどう書くか — 生成キャプションを既存内容の前に入れるか、後ろに
    足すか、丸ごと置き換えるか — は CAPTION_PLACEMENT が担当する（combine_caption）。
    """

    def core_log_gui(message: str, color: str = "black") -> None:
        if log_gui:
            log_gui(message, color)

    _get_string_internal = get_string if get_string else _get_string
    mode: ExistingFileMode = settings.get("EXISTING_FILE_MODE", ExistingFileMode.ASK)
    placement: str = settings.get("CAPTION_PLACEMENT", "OVERWRITE")
    # 「既存ファイルの扱い」で APPEND（＝常に追記）を選んでいるのに、キャプション挿入位置が
    # 既定の OVERWRITE のままだと、combine_caption が既存キャプションを丸ごと捨ててしまう
    # （PR#16 レビュー指摘: ラベルは「追記」なのに実際は上書き）。矛盾する組み合わせでは
    # 明示的な「追記」意図を優先し、既存内容を壊さない APPEND として扱う。PREPEND は尊重。
    if mode is ExistingFileMode.APPEND and placement == "OVERWRITE":
        placement = "APPEND"
    changed_files: list[FileChange] = []
    task_key = settings.get("TASK", captioner.config.default_task)
    task_prompt = captioner.config.tasks.get(task_key, captioner.config.tasks.get(captioner.config.default_task, "Describe with a paragraph what is shown in the image."))
    total = len(image_paths)
    # issue #10: 1画像=1 GUI ログだと大量画像で Qt キューが飽和する。個々の skip / 成功は
    # debug log だけに残し、GUI へは末尾のサマリ1行だけ出す。
    n_skipped = 0
    n_errors = 0
    n_unchanged = 0
    failed_seen = set(failed_paths or ())
    # progress_cb もクロススレッド signal なので毎画像発行を避け、全体で ~200 回に間引く
    # （PR#16 レビュー指摘）。最後の1枚は必ず発行して N/N に到達させる。
    # 天井除算: total//200 だと 201〜399 枚で step=1 になり間引きが効かない。
    progress_step = max(1, (total + 199) // 200)

    def mark_failed(path: Path) -> None:
        if failed_paths is not None and path not in failed_seen:
            failed_seen.add(path)
            failed_paths.append(path)

    def mark_remaining_failed(start: int) -> None:
        if failed_paths is None:
            return
        for pending in image_paths[start:]:
            if (mode is ExistingFileMode.SKIP
                    and pending.with_suffix(".txt").is_file()):
                continue
            mark_failed(pending)

    for i, image_path in enumerate(image_paths):
        if progress_cb and ((i + 1) % progress_step == 0 or i == total - 1):
            progress_cb(i + 1, total)
        if stop_checker and stop_checker():
            core_log_gui(_get_string_internal("TaggerCore", "Tagging_Process_Aborted_By_User"), "red")
            mark_remaining_failed(i)
            break

        base_name, _ = os.path.splitext(str(image_path))
        output_path = Path(base_name + ".txt")
        relative_path = image_path.relative_to(settings["INPUT_DIR"])
        current_index_str = f"[{i+1}/{len(image_paths)}]"

        if output_path.is_file():
            if mode is ExistingFileMode.SKIP:
                n_skipped += 1
                log_dbg(_get_string_internal("TaggerCore", "Tag_Skipped_Existing_File_Short", current_index_str=current_index_str, output_path_name=output_path.name))
                continue
            if mode is ExistingFileMode.ASK:
                if decision_resolver is None:
                    n_skipped += 1
                    log_dbg("process_caption_loop: ASK モードだが decision_resolver が未設定のためスキップします")
                    continue
                if stop_checker and stop_checker():
                    mark_remaining_failed(i)
                    break
                if decision_resolver(output_path) is OverwriteDecision.SKIP:
                    n_skipped += 1
                    log_dbg(_get_string_internal("TaggerCore", "Tag_Skipped_Existing_File_Short", current_index_str=current_index_str, output_path_name=output_path.name))
                    continue

        # ルーチンの「処理中」表示は GUI へ流さず debug log のみ（issue #10）。GUI は progress_cb 経由。
        log_dbg(_get_string_internal("TaggerCore", "Processing_Image", current_index_str=current_index_str, relative_path=str(relative_path)))

        try:
            with open(image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
        except Exception as e:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(f"Caption image load failed for {relative_path}: {type(e).__name__}: {e}")
            core_log_gui(_get_string_internal("TaggerCore", "Image_Load_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "red")
            continue

        try:
            caption, cancelled = captioner.generate(image, task_prompt, stop_checker)
        except Exception as e:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(f"Caption generation failed for {relative_path}: {type(e).__name__}: {e}")
            core_log_gui(_get_string_internal("TaggerCore", "Tag_Inference_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "red")
            continue

        if cancelled:
            # Stop was pressed mid-decode: `caption` is a truncated fragment. Writing it
            # would replace a good caption with a broken one, so abandon this image and
            # leave the loop - the outer stop check would end it on the next pass anyway.
            log_dbg(f"Caption generation cancelled for {relative_path}; not writing a partial caption.")
            core_log_gui(_get_string_internal("TaggerCore", "Caption_Cancelled_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "orange")
            mark_remaining_failed(i)
            break

        if not caption.strip():
            # 生成が空文字を返した（モデルの不具合・極端な画像など）。OVERWRITE だと
            # combine_caption が空文字をそのまま返し、既存キャプションを空ファイルで
            # 潰してしまう。新規側でも中身のない .txt を作って「生成済み」に見せてしまう。
            # 失敗扱いにして触らない（PR#16 レビュー指摘: coderabbit）。
            n_errors += 1
            mark_failed(image_path)
            log_dbg(f"Caption generation returned empty text for {relative_path}; leaving the file untouched.")
            core_log_gui(_get_string_internal("TaggerCore", "Caption_Empty_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "orange")
            continue

        previous_content: str | None = None
        if output_path.is_file():
            try:
                previous_content = output_path.read_text(encoding="utf-8")
            except Exception as e:
                # 既存だが読めないファイルはここで previous_content=None のまま書くと
                # 「元は存在しなかった」と undo に誤認され、undo でファイルごと削除して
                # 原本を破壊する（PR#16 レビュー指摘）。OVERWRITE でも同じ undo 経由の
                # 破壊が起きるので placement を問わず触らずスキップする。
                n_errors += 1
                mark_failed(image_path)
                log_dbg(f"caption: 既存ファイルの読み込みに失敗したためスキップします {output_path.name}: {type(e).__name__}: {e}")
                core_log_gui(_get_string_internal("TaggerCore", "Save_Failed_Short", current_index_str=current_index_str, output_path_name=output_path.name), "red")
                continue

        new_content = combine_caption(previous_content or "", caption, placement)
        if previous_content is not None and new_content == previous_content:
            # 内容が変わらないなら mtime も変えない。ルーチン結果なので GUI へは出さず debug log のみ（issue #10）。
            n_unchanged += 1
            log_dbg(_get_string_internal("TaggerCore", "Log_No_New_Tags", current_index_str=current_index_str, file_name=output_path.name))
            continue

        try:
            resolved_path = output_path.resolve()
            long_path_str = f"\\\\?\\{resolved_path}" if sys.platform == "win32" else str(resolved_path)
            with open(long_path_str, "w", encoding="utf-8") as f:
                f.write(new_content)
            changed_files.append(FileChange(
                path=output_path, previous_content=previous_content,
                new_content=new_content, was_append=(placement != "OVERWRITE" and previous_content is not None)))
            # ルーチンの出力成功は GUI へ流さず debug log のみ（issue #10）。
            log_dbg(_get_string_internal("TaggerCore", "Tag_Output_Success", current_index_str=current_index_str, output_path_name=output_path.name))
        except Exception as e:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(f"Caption save failed for {output_path.name}: {type(e).__name__}: {e}")
            core_log_gui(_get_string_internal("TaggerCore", "Save_Failed_Short", current_index_str=current_index_str, output_path_name=output_path.name), "red")

    n_appended = sum(1 for c in changed_files if c.was_append)
    core_log_gui(_get_string_internal(
        "TaggerCore", "Batch_Summary",
        written=len(changed_files), appended=n_appended, skipped=n_skipped,
        unchanged=n_unchanged, errors=n_errors, total=total), "blue")
    return changed_files
