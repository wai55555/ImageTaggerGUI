"""Phase 1 (REQ-1.2〜1.6, 1.9, 1.10): infer_batch_prepared の後処理ベクトル化の等価性テスト。

改修前（`9e21884` の `OnnxTagger.infer_batch_prepared` 後段）のロジックを
`_reference_predictions()` としてそのまま複写して保持し、改修後の実装と
「タグ名・スコア・カテゴリ・順序」が完全一致することを固定する。

スコアは ONNX セッションをスタブ化して与える（実モデル不要）。
"""
from __future__ import annotations

import itertools
from typing import Any, Mapping

import numpy as np
import pytest

import tagging_core
from tagging_core import (
    DEFAULT_TAG_OUTPUT_CAP,
    InferenceConfig,
    OnnxTagger,
    TagCategory,
    TagMeta,
    TagPrediction,
    TagResult,
)

SCORE_FLOOR = 1e-4


class _StubSession:
    """session.run() が固定のスコア配列を返すだけのスタブ。"""

    def __init__(self) -> None:
        self.arr: np.ndarray | None = None

    def run(self, _names, _feed):  # noqa: ANN001 - onnxruntime のシグネチャに合わせる
        return [self.arr]


def make_tagger(tags: list[TagMeta], activation: str = "none") -> OnnxTagger:
    """__init__ を通さずにタグだけを持つ OnnxTagger を組み立てる（実モデル不要）。"""
    tagger = object.__new__(OnnxTagger)
    tagger.inference_config = InferenceConfig(output_activation=activation)
    tagger.input_name = "images"
    tagger.output_name = "prediction"
    tagger.tags = tags
    tagger.tag_meta_lookup = {tag.name: tag for tag in tags}
    tagger.session = _StubSession()  # type: ignore[assignment]
    tagger._build_tag_arrays()
    return tagger


def run_tagger(
    tagger: OnnxTagger,
    scores: np.ndarray,
    *,
    thresholds: Mapping[TagCategory, float] | None = None,
    max_tags: Mapping[TagCategory, int] | None = None,
) -> list[TagResult]:
    rows = scores if scores.ndim == 2 else scores.reshape(1, -1)
    tagger.session.arr = rows  # type: ignore[union-attr]
    batch = np.zeros((rows.shape[0], 3, 4, 4), dtype=np.float32)
    return tagger.infer_batch_prepared(batch, None, thresholds=thresholds, max_tags=max_tags)


def _reference_predictions(
    tags: list[TagMeta],
    scores: np.ndarray,
    thresholds: Mapping[TagCategory, float] | None,
    max_tags: Mapping[TagCategory, int] | None,
) -> list[TagPrediction]:
    """9e21884 の `infer_batch_prepared` 後段そのもの（等価性の参照実装）。

    原典: `git show 9e21884:src/tagging_core.py` の `infer_batch_prepared` 内、
    `for scores in scores_batch:` 以降のループ。改修後も同じ観測結果になることを
    ここで固定するため、意図的に「遅いが読みやすい」元の形のまま残している。
    """
    cat_thresholds: Mapping[TagCategory, float] = thresholds if thresholds is not None else {}
    cat_limits: Mapping[TagCategory, int] = max_tags if max_tags is not None else {}
    hard_cap = sum(cat_limits.values()) if cat_limits else DEFAULT_TAG_OUTPUT_CAP
    score_floor = SCORE_FLOOR

    raw_predictions: list[TagPrediction] = []
    for tag_meta, score in zip(tags, scores):
        if score < score_floor:
            continue
        category = TagCategory(tag_meta.category)
        threshold = cat_thresholds.get(category, cat_thresholds.get(TagCategory.GENERAL, 0.0))
        if float(score) < threshold:
            continue
        raw_predictions.append(TagPrediction(
            name=tag_meta.name,
            score=float(score),
            category=category,
        ))
    ordered = sorted(raw_predictions, key=lambda pred: (-pred.score, pred.name))
    taken: list[TagPrediction] = []
    per_category: dict[TagCategory, int] = {}
    for prediction in ordered:
        if len(taken) >= hard_cap:
            break
        category = prediction.category
        limit = cat_limits.get(category)
        current = per_category.get(category, 0)
        if limit is not None and current >= limit:
            continue
        per_category[category] = current + 1
        taken.append(prediction)
    return taken


def as_tuples(predictions: list[TagPrediction]) -> list[tuple[str, Any, int]]:
    """比較用の正規化。NaN は自分自身と等しくないので、比較できる番兵に置き換える。"""
    normalized: list[tuple[str, Any, int]] = []
    for pred in predictions:
        score: Any = pred.score
        if score != score:  # NaN
            score = "nan"
        normalized.append((pred.name, score, int(pred.category)))
    return normalized


def compare(
    tagger: OnnxTagger,
    scores: np.ndarray,
    *,
    thresholds: Mapping[TagCategory, float] | None = None,
    max_tags: Mapping[TagCategory, int] | None = None,
) -> None:
    expected = _reference_predictions(tagger.tags, scores, thresholds, max_tags)
    actual = run_tagger(tagger, scores, thresholds=thresholds, max_tags=max_tags)[0].tags
    assert as_tuples(actual) == as_tuples(expected)


# ---------------------------------------------------------------------------
# ランダムなスコア・閾値・上限の組み合わせで参照実装と一致すること（REQ-1.4）
# ---------------------------------------------------------------------------

_THRESHOLD_PATTERNS: list[Mapping[TagCategory, float] | None] = [
    None,
    {},
    {TagCategory.GENERAL: 0.3},
    {TagCategory.CHARACTER: 0.7, TagCategory.GENERAL: 0.2},
    {TagCategory.ARTIST: 0.0, TagCategory.RATING: 0.9},
]

_LIMIT_PATTERNS: list[Mapping[TagCategory, int] | None] = [
    None,
    {},
    {TagCategory.GENERAL: 20},
    {TagCategory.GENERAL: 3, TagCategory.CHARACTER: 2, TagCategory.ARTIST: 2},
    {TagCategory.GENERAL: 0},
    {TagCategory.GENERAL: 5, TagCategory.CHARACTER: 0},
]


@pytest.mark.parametrize("seed", [0, 1, 7, 99, 1234, 987654])
@pytest.mark.parametrize(
    "thresholds,limits",
    list(itertools.product(_THRESHOLD_PATTERNS, _LIMIT_PATTERNS)),
)
def test_matches_reference_on_random_scores(seed, thresholds, limits):
    rng = np.random.default_rng(seed)
    n_tags = int(rng.integers(1, 60))
    # カテゴリをばらけさせ、名前も重複させる（同名・同スコアの同点を作る）
    tags = [
        TagMeta(
            name=f"tag{rng.integers(0, n_tags // 2 + 1)}",
            category=int(rng.integers(0, 10)),
        )
        for _ in range(n_tags)
    ]
    # 床未満・閾値近傍・重複値を混ぜる（同点の大量発生を作る）
    pool = np.array([0.0, SCORE_FLOOR, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0, 0.7])
    scores = pool[rng.integers(0, pool.size, size=n_tags)]
    tagger = make_tagger(tags)
    compare(tagger, scores, thresholds=thresholds, max_tags=limits)


def test_matches_reference_with_unicode_names():
    tags = [
        TagMeta(name="ねこ", category=int(TagCategory.GENERAL)),
        TagMeta(name="ねこ", category=int(TagCategory.GENERAL)),
        TagMeta(name="猫", category=int(TagCategory.CHARACTER)),
        TagMeta(name="cat", category=int(TagCategory.GENERAL)),
        TagMeta(name="Cat", category=int(TagCategory.GENERAL)),
        TagMeta(name="café", category=int(TagCategory.GENERAL)),
        TagMeta(name="cafe", category=int(TagCategory.GENERAL)),
    ]
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6])
    tagger = make_tagger(tags)
    compare(tagger, scores)
    # 同点はタグ名昇順（コードポイント順）、完全同点の「ねこ」同士は元のタグ順が保たれる
    result = run_tagger(tagger, scores)[0].tags
    assert [pred.name for pred in result] == ["cafe", "café", "Cat", "cat", "ねこ", "ねこ", "猫"]


def test_matches_reference_on_dense_random_scores():
    rng = np.random.default_rng(42)
    tags = [TagMeta(name=f"t{i}", category=int(rng.integers(0, 10))) for i in range(2000)]
    scores = rng.random(len(tags))
    tagger = make_tagger(tags)
    compare(tagger, scores, thresholds={TagCategory.GENERAL: 0.3},
            max_tags={TagCategory.GENERAL: 30, TagCategory.CHARACTER: 5})


# ---------------------------------------------------------------------------
# 境界ケース（§1.4）
# ---------------------------------------------------------------------------


def _five_tags() -> list[TagMeta]:
    return [
        TagMeta(name="a", category=int(TagCategory.GENERAL)),
        TagMeta(name="b", category=int(TagCategory.CHARACTER)),
        TagMeta(name="c", category=int(TagCategory.GENERAL)),
        TagMeta(name="d", category=int(TagCategory.ARTIST)),
        TagMeta(name="e", category=int(TagCategory.GENERAL)),
    ]


def test_all_scores_below_floor():
    tagger = make_tagger(_five_tags())
    compare(tagger, np.full(5, SCORE_FLOOR / 2))


def test_all_scores_above_threshold():
    tagger = make_tagger(_five_tags())
    compare(tagger, np.full(5, 0.99), thresholds={TagCategory.GENERAL: 0.3})


def test_score_exactly_at_floor_is_kept():
    tagger = make_tagger(_five_tags())
    result = run_tagger(tagger, np.full(5, SCORE_FLOOR))[0].tags
    assert len(result) == 5


def test_score_exactly_at_threshold_is_kept():
    """「未満が棄却」。閾値ちょうどは残る（float32 化すると 0.3 が 0.3000000119 になり反転する）。"""
    tagger = make_tagger(_five_tags())
    result = run_tagger(tagger, np.full(5, 0.3), thresholds={TagCategory.GENERAL: 0.3})[0].tags
    assert len(result) == 5
    compare(tagger, np.full(5, 0.3), thresholds={TagCategory.GENERAL: 0.3})


def test_category_limit_zero_only_makes_hard_cap_zero():
    """max_tags が非空なら全体上限は合計値になる（§1.1-6）。合計 0 なら 1 件も残らない。"""
    tagger = make_tagger(_five_tags())
    scores = np.full(5, 0.5)
    result = run_tagger(tagger, scores, max_tags={TagCategory.GENERAL: 0})[0].tags
    assert result == []
    compare(tagger, scores, max_tags={TagCategory.GENERAL: 0})


def test_category_limit_zero_skips_that_category_but_keeps_others():
    tagger = make_tagger(_five_tags())
    scores = np.full(5, 0.5)
    limits = {TagCategory.GENERAL: 0, TagCategory.ARTIST: 5}
    result = run_tagger(tagger, scores, max_tags=limits)[0].tags
    assert [pred.name for pred in result] == ["b", "d"]
    compare(tagger, scores, max_tags=limits)


def test_limits_sum_zero_yields_empty_result():
    tagger = make_tagger(_five_tags())
    scores = np.full(5, 0.5)
    result = run_tagger(tagger, scores, max_tags={TagCategory.GENERAL: 0, TagCategory.ARTIST: 0})[0].tags
    assert result == []
    compare(tagger, scores, max_tags={TagCategory.GENERAL: 0, TagCategory.ARTIST: 0})


def test_default_hard_cap_applies_without_limits():
    tags = [TagMeta(name=f"t{i:03d}", category=int(TagCategory.GENERAL)) for i in range(150)]
    tagger = make_tagger(tags)
    scores = np.linspace(1.0, 0.5, len(tags))
    result = run_tagger(tagger, scores)[0].tags
    assert len(result) == DEFAULT_TAG_OUTPUT_CAP
    compare(tagger, scores)


def test_no_tags_returns_empty_results_per_row():
    tagger = make_tagger([])
    results = run_tagger(tagger, np.zeros((3, 5)))
    assert len(results) == 3
    assert all(result.tags == [] for result in results)


def test_empty_batch_returns_empty_list():
    tagger = make_tagger(_five_tags())
    assert tagger.infer_batch_prepared(np.zeros((0, 3, 4, 4), dtype=np.float32)) == []


def test_scores_shorter_than_tags_are_truncated():
    tagger = make_tagger(_five_tags())
    compare(tagger, np.array([0.5, 0.5]))


def test_scores_longer_than_tags_are_ignored():
    tagger = make_tagger(_five_tags())
    compare(tagger, np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]))


def test_duplicate_name_and_score_keeps_original_tag_order():
    tags = [TagMeta(name="dup", category=int(TagCategory.GENERAL)) for _ in range(4)]
    tags[1] = TagMeta(name="other", category=int(TagCategory.GENERAL))
    tagger = make_tagger(tags)
    scores = np.full(4, 0.5)
    result = run_tagger(tagger, scores)[0].tags
    # 同点なのでタグ名昇順 → "dup" が 3 つ（元のタグ順を維持）並び、最後に "other"
    assert [pred.name for pred in result] == ["dup", "dup", "dup", "other"]
    compare(tagger, scores)


def test_rows_are_independent():
    tagger = make_tagger(_five_tags())
    rows = np.array([
        [SCORE_FLOOR / 10, 0.9, SCORE_FLOOR / 10, 0.9, SCORE_FLOOR / 10],
        [0.9, SCORE_FLOOR / 10, 0.9, SCORE_FLOOR / 10, 0.9],
    ])
    results = run_tagger(tagger, rows)
    assert [pred.name for pred in results[0].tags] == ["b", "d"]
    assert [pred.name for pred in results[1].tags] == ["a", "c", "e"]
    for row, result in zip(rows, results):
        assert as_tuples(result.tags) == as_tuples(_reference_predictions(tagger.tags, row, None, None))


@pytest.mark.parametrize("activation", ["none", "sigmoid"])
def test_output_activation_uses_the_same_postprocess(activation):
    """REQ-1.6: sigmoid の有無で後処理の分岐を増やさない。"""
    tagger = make_tagger(_five_tags(), activation=activation)
    scores = np.array([0.0, 1.0, 2.0, -1.0, 0.5])
    results = run_tagger(tagger, scores)
    post = scores if activation == "none" else 1.0 / (1.0 + np.exp(-scores))
    assert as_tuples(results[0].tags) == as_tuples(_reference_predictions(tagger.tags, post, None, None))


# ---------------------------------------------------------------------------
# NaN（REQ-1.10 / §6 U-1）: 現行実装へ委譲しているので参照実装と一致する
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nan_position", [0, 2, 4])
def test_nan_scores_match_reference(nan_position):
    tagger = make_tagger(_five_tags())
    scores = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    scores[nan_position] = np.nan
    compare(tagger, scores)


def test_nan_with_thresholds_and_limits_matches_reference():
    tagger = make_tagger(_five_tags())
    scores = np.array([np.nan, 0.6, 0.7, 0.8, np.nan])
    compare(
        tagger,
        scores,
        thresholds={TagCategory.GENERAL: 0.5},
        max_tags={TagCategory.GENERAL: 2, TagCategory.CHARACTER: 1},
    )


# ---------------------------------------------------------------------------
# 閾値ベクトル・上限ベクトルのフォールバック規則（REQ-1.3）
# ---------------------------------------------------------------------------


def test_threshold_vector_falls_back_to_general_then_zero():
    tagger = make_tagger(_five_tags())
    assert tagger._threshold_vector({TagCategory.GENERAL: 0.25})[int(TagCategory.ARTIST)] == 0.25
    no_general = tagger._threshold_vector({TagCategory.ARTIST: 0.5})
    assert no_general[int(TagCategory.GENERAL)] == 0.0
    assert no_general[int(TagCategory.ARTIST)] == 0.5
    assert no_general[int(TagCategory.CHARACTER)] == 0.0


def test_threshold_and_limit_vectors_ignore_unusable_keys():
    tagger = make_tagger(_five_tags())
    thresholds = {TagCategory.GENERAL: 0.5, "not-a-category": 0.9, 999: 0.9}  # type: ignore[dict-item]
    vector = tagger._threshold_vector(thresholds)
    assert vector[int(TagCategory.GENERAL)] == 0.5
    limits = {TagCategory.GENERAL: 2, "not-a-category": 5}  # type: ignore[dict-item]
    limit_vector = tagger._limit_vector(limits)
    assert limit_vector[int(TagCategory.GENERAL)] == 2
    assert np.isnan(limit_vector[int(TagCategory.ARTIST)])


def test_int_keys_behave_like_tag_category_keys():
    tagger = make_tagger(_five_tags())
    scores = np.full(5, 0.4)
    by_enum = run_tagger(tagger, scores, thresholds={TagCategory.GENERAL: 0.5})[0].tags
    by_int = run_tagger(tagger, scores, thresholds={0: 0.5})[0].tags  # type: ignore[dict-item]
    assert as_tuples(by_int) == as_tuples(by_enum)


def test_tag_category_ids_are_precomputed():
    tagger = make_tagger(_five_tags())
    assert tagger._tag_category_ids.dtype == np.intp
    assert tagger._tag_category_ids.tolist() == [int(tag.category) for tag in tagger.tags]
    assert tagger._category_slots >= len(tagging_core.TagCategory)


@pytest.mark.parametrize("seed", list(range(20)))
def test_matches_reference_when_thresholds_equal_actual_scores(seed):
    """閾値を実スコアから採る（境界ちょうどの比較で float32 化の誤差が出ないこと）。"""
    rng = np.random.default_rng(seed)
    n_tags = 40
    tags = [TagMeta(name=f"t{i}", category=int(rng.integers(0, 10))) for i in range(n_tags)]
    scores = rng.random(n_tags)
    thresholds = {
        TagCategory.GENERAL: float(scores[int(rng.integers(0, n_tags))]),
        TagCategory.ARTIST: float(scores[int(rng.integers(0, n_tags))]),
    }
    limits = {TagCategory.GENERAL: int(rng.integers(0, 15)), TagCategory.CHARACTER: int(rng.integers(0, 5))}
    tagger = make_tagger(tags)
    compare(tagger, scores, thresholds=thresholds, max_tags=limits)


def test_vectorized_path_is_used_for_normal_rows(monkeypatch):
    """NaN を含まない行は参照実装（旧ループ）へ委譲していないこと。"""
    called = {"reference": 0}
    original = OnnxTagger._predictions_for_row_reference

    def spy(self, *args, **kwargs):
        called["reference"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(OnnxTagger, "_predictions_for_row_reference", spy)
    tagger = make_tagger(_five_tags())
    run_tagger(tagger, np.full(5, 0.5))
    run_tagger(tagger, np.array([0.5, 0.6, 0.7, 0.8, 0.9]))
    assert called["reference"] == 0

    # NaN を含む行は委譲する（現行実装そのもので挙動を固定するため）
    run_tagger(tagger, np.array([np.nan, 0.6, 0.7, 0.8, 0.9]))
    assert called["reference"] == 1


# ---------------------------------------------------------------------------
# 先頭接頭辞だけで並べ替える高速路（候補が上限よりずっと多い行）の等価性
# ---------------------------------------------------------------------------


def _many_tags(count: int, categories: list[int]) -> list[TagMeta]:
    return [
        TagMeta(name=f"t{i:05d}", category=categories[i % len(categories)])
        for i in range(count)
    ]


def test_prefix_path_matches_reference_with_many_ties():
    """同点が大量にある行でも、接頭辞だけで確定した結果が参照実装と一致する。"""
    rng = np.random.default_rng(5)
    tags = _many_tags(2000, [int(TagCategory.GENERAL), int(TagCategory.CHARACTER), int(TagCategory.ARTIST)])
    # 0.5 の同点を大量に作る（実モデルで sigmoid が飽和する場合に相当）
    scores = np.where(rng.random(2000) < 0.7, 0.5, rng.random(2000))
    tagger = make_tagger(tags)
    for limits in (
        {TagCategory.GENERAL: 20, TagCategory.CHARACTER: 5},
        {TagCategory.GENERAL: 55, TagCategory.CHARACTER: 1},
        {TagCategory.GENERAL: 300},
    ):
        compare(tagger, scores, thresholds={TagCategory.GENERAL: 0.4}, max_tags=limits)


def test_prefix_path_falls_back_when_caps_cannot_be_filled(monkeypatch):
    """接頭辞では上限を埋め切れない行（上限に達しないカテゴリが接頭辞の外）は全件でやり直す。"""
    categories = [int(TagCategory.GENERAL)] * 400 + [int(TagCategory.ARTIST)]
    tags = _many_tags(401, [int(TagCategory.GENERAL)])[:400] + [
        TagMeta(name="zzz_artist", category=int(TagCategory.ARTIST))
    ]
    scores = np.concatenate([np.full(400, 0.9), np.array([0.1])])
    tagger = make_tagger(tags)

    calls: list[int] = []
    original = OnnxTagger._sort_candidates

    def spy(self, candidate_index, candidate_scores):
        calls.append(int(candidate_index.size))
        return original(self, candidate_index, candidate_scores)

    monkeypatch.setattr(OnnxTagger, "_sort_candidates", spy)
    limits = {TagCategory.GENERAL: 0, TagCategory.ARTIST: 3}
    result = run_tagger(tagger, scores, max_tags=limits)[0].tags
    # GENERAL は上限 0 で全部飛ばされ、ARTIST の 1 件だけが残る
    assert [pred.name for pred in result] == ["zzz_artist"]
    # 1 回目は接頭辞（小さい）、2 回目は全候補（401）で並べ替えている
    assert calls[-1] == 401
    compare(tagger, scores, max_tags=limits)


def test_prefix_path_with_threshold_and_all_categories():
    rng = np.random.default_rng(11)
    tags = _many_tags(1000, list(range(10)))
    scores = np.round(rng.random(1000), 1)  # 同点を作る
    tagger = make_tagger(tags)
    limits = {category: 3 for category in TagCategory}
    compare(tagger, scores, thresholds={TagCategory.GENERAL: 0.2}, max_tags=limits)
