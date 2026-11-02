from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from relbench import metrics as relbench_metrics
from relbench.base import Table
from relbench.base.task_recommendation import RecommendationTask

from avito_recsys.metrics import PAD, map_at_k, ndcg_at_k, recall_at_k, recs_to_array


def test_map_by_hand():
    """Попадания на местах 1 и 3: (1/1 + 2/3) / 2 = 5/6."""
    pred = np.array([[1, 2, 3]])
    assert map_at_k(pred, [[1, 3]], k=3) == pytest.approx(5 / 6)


def test_map_clips_denominator_to_k():
    pred = np.array([[1, 2]])
    assert map_at_k(pred, [[1, 2, 3, 4, 5]], k=2) == pytest.approx(1.0)


def test_map_averages_over_users():
    pred = np.array([[1, 2], [3, 4]])
    assert map_at_k(pred, [[1], [9]], k=2) == pytest.approx(0.5)


def test_users_without_true_are_skipped():
    pred = np.array([[1, 2], [3, 4]])
    true = [[1], []]
    assert map_at_k(pred, true, k=2) == pytest.approx(1.0)
    assert recall_at_k(pred, true, k=2) == pytest.approx(1.0)
    assert ndcg_at_k(pred, true, k=2) == pytest.approx(1.0)


def test_recall_by_hand():
    """В отличие от MAP, знаменатель не обрезан: из 5 ответов найдено 2."""
    pred = np.array([[1, 2, 3]])
    assert recall_at_k(pred, [[1, 3]], k=1) == pytest.approx(1 / 2)
    assert recall_at_k(pred, [[1, 3]], k=3) == pytest.approx(1.0)
    assert recall_at_k(pred, [[1, 2, 7, 8, 9]], k=3) == pytest.approx(2 / 5)


def test_ndcg_by_hand():
    """DCG = 1 + 1/log2(4) = 1.5, идеальный DCG = 1 + 1/log2(3)."""
    pred = np.array([[1, 2, 3]])
    assert ndcg_at_k(pred, [[1, 3]], k=3) == pytest.approx(1.5 / (1 + 1 / np.log2(3)))


def test_ndcg_clips_ideal_to_k():
    pred = np.array([[1, 2]])
    assert ndcg_at_k(pred, [[1, 2, 3, 4, 5]], k=2) == pytest.approx(1.0)


def test_no_hits_gives_zero():
    pred = np.array([[1, 2]])
    for metric in (map_at_k, recall_at_k, ndcg_at_k):
        assert metric(pred, [[9]], k=2) == 0.0


def test_k_cuts_longer_predictions():
    pred = np.array([[9, 1]])
    assert recall_at_k(pred, [[1]], k=1) == 0.0
    assert recall_at_k(pred, [[1]], k=2) == 1.0


def test_wrong_shapes_raise():
    pred = np.array([[1, 2]])
    with pytest.raises(ValueError):
        map_at_k(pred, [[1]], k=3)
    with pytest.raises(ValueError):
        map_at_k(pred, [[1], [2]], k=2)


def test_recs_to_array_pads_truncates_and_keeps_order():
    """Длинная выдача обрезается, короткая дополняется, пользователя 4 нет в словаре."""
    recs = {1: [10, 11, 12, 13], 2: [20], 3: []}
    pred = recs_to_array(recs, users=[2, 1, 3, 4], k=3)
    expected = [
        [20, PAD, PAD],
        [10, 11, 12],
        [PAD, PAD, PAD],
        [PAD, PAD, PAD],
    ]
    assert pred.tolist() == expected


def test_recs_to_array_finds_numpy_ids_and_arrays():
    pred = recs_to_array({1: np.array([10, 11])}, users=np.array([1]), k=2)
    assert pred.tolist() == [[10, 11]]


def test_recs_to_array_rejects_duplicates():
    with pytest.raises(ValueError):
        recs_to_array({1: [10, 10]}, users=[1], k=2)


def test_padding_is_not_a_hit():
    pred = recs_to_array({1: [5]}, users=[1], k=3)
    assert map_at_k(pred, [[5]], k=3) == pytest.approx(1.0)
    assert recall_at_k(pred, [[5, 6]], k=3) == pytest.approx(0.5)
    assert ndcg_at_k(pred, [[5]], k=3) == pytest.approx(1.0)


def _random_case(seed: int, k: int = 12, n_users: int = 500, n_ads: int = 40):
    """Случайные выдачи, в которых есть дубликаты, паддинг и пользователи без ответов."""
    rng = np.random.default_rng(seed)
    pred = rng.integers(PAD, n_ads, size=(n_users, k))
    true = [rng.choice(n_ads, size=rng.integers(0, 30), replace=False) for _ in range(n_users)]
    return pred, true


@pytest.mark.parametrize("seed", range(5))
def test_map_matches_relbench_metric(seed):
    pred, true = _random_case(seed)
    pred_isin = np.stack([np.isin(row, row_true) for row, row_true in zip(pred, true)])
    dst_count = np.array([len(row_true) for row_true in true])
    assert map_at_k(pred, true, k=12) == pytest.approx(relbench_metrics.map(pred_isin, dst_count))


def test_map_matches_task_evaluate():
    """Вызывает настоящий evaluate без скачивания датасета: ему нужны только эти атрибуты задачи."""
    pred, true = _random_case(seed=0)
    true = [row_true for row_true in true if len(row_true) > 0]
    pred = pred[: len(true)]
    task = SimpleNamespace(eval_k=12, dst_entity_col="AdID", metrics=[relbench_metrics.map])
    table = Table(
        df=pd.DataFrame({"AdID": true}),
        fkey_col_to_pkey_table={},
        pkey_col=None,
        time_col=None,
    )
    result = RecommendationTask.evaluate(task, pred, target_table=table)
    assert map_at_k(pred, true, k=12) == pytest.approx(result["map"])
