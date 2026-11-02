"""Метрики top-k для рекомендаций.

MAP@k считается ровно как relbench.metrics.map, чтобы цифры совпадали с бенчмарком.
Везде pred имеет форму (n_users, >=k), а true содержит список объявлений на каждую строку pred.
Строки с пустым true в среднее не входят, как и в RelBench.
PAD заполняет выдачу короче k, настоящих объявлений с таким id нет.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

PAD = -1


def recs_to_array(recs: Mapping[int, Sequence[int]], users: Sequence[int], k: int) -> np.ndarray:
    """Строит массив (n_users, k) в порядке users, как ждёт task.evaluate.

    Пользователь без рекомендаций получает строку из PAD. Дубликаты в выдаче считаем ошибкой:
    RelBench засчитает каждый как отдельное попадание, и метрика окажется завышенной.
    """
    pred = np.full((len(users), k), PAD, dtype=np.int64)
    for i, user in enumerate(users):
        items = recs.get(user, [])[:k]
        if len(set(items)) != len(items):
            raise ValueError(f"duplicate ads in recommendations for user {user}")
        pred[i, : len(items)] = items
    return pred


def _hits(pred: np.ndarray, true: Sequence[Sequence[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает матрицу попаданий (n_users, k) и число правильных ответов на строку."""
    if len(pred) != len(true):
        raise ValueError(f"pred has {len(pred)} rows, true has {len(true)}")
    if pred.shape[1] < k:
        raise ValueError(f"pred has {pred.shape[1]} columns, need at least {k}")
    hits = np.zeros((len(pred), k), dtype=bool)
    for i, (row, row_true) in enumerate(zip(pred, true)):
        hits[i] = np.isin(row[:k], row_true)
    dst_count = np.array([len(row_true) for row_true in true], dtype=np.int64)
    has_true = dst_count > 0
    return hits[has_true], dst_count[has_true]


def map_at_k(pred: np.ndarray, true: Sequence[Sequence[int]], k: int) -> float:
    """Считает MAP@k: среднюю точность на позициях попаданий, усреднённую по пользователям.

    Знаменатель обрезан до k: иначе при 100 правильных ответах результат не достигнет 1.
    """
    hits, dst_count = _hits(pred, true, k)
    precision = np.cumsum(hits, axis=1) / (np.arange(k) + 1)
    return float(((precision * hits).sum(axis=1) / dst_count.clip(max=k)).mean())


def recall_at_k(pred: np.ndarray, true: Sequence[Sequence[int]], k: int) -> float:
    """Считает Recall@k: долю правильных ответов, попавших в первые k позиций.

    Знаменатель не обрезан, поэтому при числе ответов больше k результат не достигает 1.
    """
    hits, dst_count = _hits(pred, true, k)
    return float((hits.sum(axis=1) / dst_count).mean())


def ndcg_at_k(pred: np.ndarray, true: Sequence[Sequence[int]], k: int) -> float:
    """Считает NDCG@k с бинарной релевантностью.

    Идеальная выдача ставит все правильные ответы на первые места, не больше k штук.
    """
    hits, dst_count = _hits(pred, true, k)
    discounts = 1 / np.log2(np.arange(k) + 2)
    ideal = np.cumsum(discounts)[dst_count.clip(max=k) - 1]
    return float(((hits * discounts).sum(axis=1) / ideal).mean())
