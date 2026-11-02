"""Оценка генераторов на срезе задачи."""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from avito_recsys.generators import Generator
from avito_recsys.metrics import map_at_k, ndcg_at_k, recall_at_k, recs_to_array

EVAL_K = 12
RECALL_KS = (12, 50, 100, 200)


def evaluate(
    generator: Generator,
    interactions: pd.DataFrame,
    task_slice: pd.DataFrame,
    recall_ks: Sequence[int] = RECALL_KS,
) -> dict[str, float]:
    """Обучает генератор на событиях до момента среза и считает метрики на его ответах.

    Срез должен содержать один момент времени, поэтому каждый пользователь встречается один раз.
    """
    cutoffs = task_slice.timestamp.unique()
    if len(cutoffs) != 1:
        raise ValueError(f"expected one timestamp in task slice, got {len(cutoffs)}")
    users = task_slice.UserID.tolist()
    true = task_slice.AdID.tolist()
    k = max(EVAL_K, *recall_ks)

    generator.fit(interactions, pd.Timestamp(cutoffs[0]))
    pred = recs_to_array(generator.recommend(users, k), users, k)

    result = {f"MAP@{EVAL_K}": map_at_k(pred, true, EVAL_K), f"NDCG@{EVAL_K}": ndcg_at_k(pred, true, EVAL_K)}
    result.update({f"Recall@{recall_k}": recall_at_k(pred, true, recall_k) for recall_k in recall_ks})
    return result
