"""Генераторы кандидатов с общим интерфейсом fit(interactions, cutoff) и recommend(users, k).

Базовый класс оставляет в обучении только события строго раньше cutoff, поэтому наследники
не могут случайно заглянуть в будущее. Для пользователя, про которого генератору нечего сказать,
recommend возвращает пустой список. Дозаполнение делает BlendGenerator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

SECONDS_PER_DAY = 86_400


class Generator(ABC):
    """Общий интерфейс всех генераторов."""

    def fit(self, interactions: pd.DataFrame, cutoff: pd.Timestamp) -> Generator:
        """Обучает генератор на событиях строго раньше cutoff."""
        self.cutoff = cutoff
        self._fit(interactions[interactions.ts < cutoff])
        return self

    @abstractmethod
    def _fit(self, events: pd.DataFrame) -> None:
        """Строит внутреннее состояние по уже отфильтрованным событиям."""

    @abstractmethod
    def recommend(self, users: Sequence[int], k: int) -> dict[int, list[int]]:
        """Возвращает до k разных объявлений на пользователя, лучшие первыми."""

    def _recent(self, events: pd.DataFrame, days: int) -> pd.DataFrame:
        """Оставляет события за последние days дней перед cutoff."""
        return events[events.ts >= self.cutoff - pd.Timedelta(days=days)]


def _ranked_ads(ad_scores: pd.Series) -> np.ndarray:
    """Сортирует объявления по убыванию скора, при равенстве по id, чтобы порядок не зависел от запуска."""
    return ad_scores.index.to_numpy()[np.lexsort((ad_scores.index.to_numpy(), -ad_scores.to_numpy()))]


def _decayed_scores(events: pd.DataFrame, cutoff: pd.Timestamp, half_life_days: float) -> pd.DataFrame:
    """Складывает веса событий по парам user_id, ad_id. Вес падает вдвое каждые half_life_days дней.

    Объявления каждого пользователя отсортированы по убыванию скора, при равенстве по id.
    """
    age_days = (cutoff - events.ts).dt.total_seconds() / SECONDS_PER_DAY
    scored = events[["user_id", "ad_id"]].assign(score=events.weight * 0.5 ** (age_days / half_life_days))
    scored = scored.groupby(["user_id", "ad_id"], as_index=False).score.sum()
    return scored.sort_values(["user_id", "score", "ad_id"], ascending=[True, False, True])


class HistoryGenerator(Generator):
    """Объявления из истории пользователя: вес события затухает с давностью.

    Вес падает вдвое каждые half_life_days дней, поэтому свежий просмотр важнее старого.
    """

    def __init__(self, half_life_days: float = 3.0):
        """Запоминает период полураспада веса в днях."""
        self.half_life_days = half_life_days

    def _fit(self, events: pd.DataFrame) -> None:
        """Запоминает объявления каждого пользователя в порядке убывания скора."""
        scored = _decayed_scores(events, self.cutoff, self.half_life_days)
        self.user_ads = scored.groupby("user_id").ad_id.agg(list).to_dict()

    def recommend(self, users: Sequence[int], k: int) -> dict[int, list[int]]:
        """Возвращает первые k объявлений из истории пользователя."""
        return {user: self.user_ads.get(user, [])[:k] for user in users}


class PopularGenerator(Generator):
    """Одна и та же выдача для всех: самые популярные объявления за последние days дней."""

    def __init__(self, days: int = 7):
        """Запоминает длину окна популярности в днях."""
        self.days = days

    def _fit(self, events: pd.DataFrame) -> None:
        """Считает суммарный вес событий по объявлениям внутри окна."""
        ad_scores = self._recent(events, self.days).groupby("ad_id").weight.sum()
        self.top_ads = _ranked_ads(ad_scores).tolist()

    def recommend(self, users: Sequence[int], k: int) -> dict[int, list[int]]:
        """Возвращает всем пользователям один и тот же топ."""
        top = self.top_ads[:k]
        return {user: top for user in users}


class SegmentPopularGenerator(Generator):
    """Популярное в сегментах, которые пользователь смотрел чаще всего.

    Сегмент задаётся колонками таблицы объявлений, например категорией или категорией и регионом.
    Скор объявления равен доле сегмента в истории пользователя, умноженной на популярность внутри сегмента.
    """

    def __init__(
        self,
        ads: pd.DataFrame,
        segment_cols: Sequence[str] = ("CategoryID",),
        top_segments: int = 3,
        days: int = 7,
        max_per_segment: int = 200,
    ):
        """Запоминает таблицу объявлений с колонками AdID и segment_cols и параметры отбора."""
        self.ads = ads
        self.segment_cols = list(segment_cols)
        self.top_segments = top_segments
        self.days = days
        self.max_per_segment = max_per_segment

    def _fit(self, events: pd.DataFrame) -> None:
        """Считает топ объявлений каждого сегмента и любимые сегменты каждого пользователя."""
        cols = self.segment_cols
        ad_segments = self.ads[["AdID", *cols]].dropna()
        events = events.merge(ad_segments, left_on="ad_id", right_on="AdID")

        recent = self._recent(events, self.days)
        ad_scores = recent.groupby([*cols, "ad_id"]).weight.sum().rename("score").reset_index()
        ad_scores = ad_scores.sort_values([*cols, "score", "ad_id"], ascending=[True] * len(cols) + [False, True])
        ad_scores = ad_scores.groupby(cols).head(self.max_per_segment)
        self.segment_ads = {
            segment: (group.ad_id.to_numpy(), group.score.to_numpy())
            for segment, group in ad_scores.groupby(cols)
        }

        weights = events.groupby(["user_id", *cols]).weight.sum().rename("weight").reset_index()
        weights["share"] = weights.weight / weights.groupby("user_id").weight.transform("sum")
        weights = weights.sort_values(["user_id", "share", *cols], ascending=[True, False] + [True] * len(cols))
        weights = weights.groupby("user_id").head(self.top_segments)
        self.user_segments = {
            user: list(zip(group[cols].itertuples(index=False, name=None), group.share))
            for user, group in weights.groupby("user_id")
        }

    def recommend(self, users: Sequence[int], k: int) -> dict[int, list[int]]:
        """Объединяет топы любимых сегментов пользователя и сортирует их по скору."""
        return {user: self._recommend_user(user, k) for user in users}

    def _recommend_user(self, user: int, k: int) -> list[int]:
        """Строит выдачу одного пользователя. Объявление лежит в одном сегменте, поэтому дублей нет."""
        parts = [
            (self.segment_ads[segment][0], share * self.segment_ads[segment][1])
            for segment, share in self.user_segments.get(user, [])
            if segment in self.segment_ads
        ]
        if not parts:
            return []
        ads = np.concatenate([part[0] for part in parts])
        scores = np.concatenate([part[1] for part in parts])
        return ads[np.lexsort((ads, -scores))][:k].tolist()


class Item2ItemGenerator(Generator):
    """Похожие объявления: два объявления похожи, если их смотрели одни и те же пользователи.

    Сходство косинусное: c / (sqrt(n_i * n_j) + shrink), где c это число общих пользователей, а n это
    число пользователей объявления. shrink снижает сходство пар с малым числом наблюдений.
    Скор кандидата равен сумме сходств с объявлениями пользователя, умноженных на их затухающий вес.
    Активные пользователи ограничены max_user_ads лучшими объявлениями: иначе число пар растёт квадратично.
    С exclude_seen=True просмотренное не предлагается: его уже покрывает HistoryGenerator.
    """

    def __init__(
        self,
        half_life_days: float = 3.0,
        max_user_ads: int = 50,
        top_neighbors: int = 50,
        shrink: float = 0.0,
        exclude_seen: bool = True,
        batch_size: int = 2_000,
    ):
        """Запоминает параметры обучения и размер пачки пользователей в recommend."""
        self.half_life_days = half_life_days
        self.max_user_ads = max_user_ads
        self.top_neighbors = top_neighbors
        self.shrink = shrink
        self.exclude_seen = exclude_seen
        self.batch_size = batch_size

    def _fit(self, events: pd.DataFrame) -> None:
        """Строит матрицу весов пользователь-объявление и матрицу сходства объявлений."""
        scored = _decayed_scores(events, self.cutoff, self.half_life_days)
        scored = scored.groupby("user_id").head(self.max_user_ads)

        user_rows, users = pd.factorize(scored.user_id, sort=True)
        ad_cols, self.ads = pd.factorize(scored.ad_id, sort=True)
        shape = (len(users), len(self.ads))
        self.user_index = {user: row for row, user in enumerate(users.tolist())}
        self.user_weights = sp.csr_matrix((scored.score.to_numpy(), (user_rows, ad_cols)), shape=shape)
        seen = sp.csr_matrix((np.ones(len(scored), dtype=np.int32), (user_rows, ad_cols)), shape=shape)
        self.similarity = self._build_similarity(seen)

    def _build_similarity(self, seen: sp.csr_matrix) -> sp.csr_matrix:
        """Считает косинусное сходство и оставляет top_neighbors соседей у каждого объявления."""
        co_views = (seen.T @ seen).tocoo()
        off_diagonal = co_views.row != co_views.col
        rows, cols = co_views.row[off_diagonal], co_views.col[off_diagonal]
        ad_users = np.asarray(seen.sum(axis=0)).ravel().astype(np.float64)
        scores = co_views.data[off_diagonal] / (np.sqrt(ad_users[rows] * ad_users[cols]) + self.shrink)

        order = np.lexsort((cols, -scores, rows))
        rows, cols, scores = rows[order], cols[order], scores[order]
        row_starts = np.searchsorted(rows, rows, side="left")
        keep = np.arange(len(rows)) - row_starts < self.top_neighbors
        return sp.csr_matrix((scores[keep], (rows[keep], cols[keep])), shape=co_views.shape)

    def recommend(self, users: Sequence[int], k: int) -> dict[int, list[int]]:
        """Считает скоры пачками, чтобы не держать в памяти большое произведение матриц."""
        known = [user for user in users if user in self.user_index]
        recs: dict[int, list[int]] = {user: [] for user in users}
        for start in range(0, len(known), self.batch_size):
            batch = known[start : start + self.batch_size]
            weights = self.user_weights[[self.user_index[user] for user in batch]]
            scores = (weights @ self.similarity).tocsr()
            for row, user in enumerate(batch):
                recs[user] = self._top_ads(scores, weights, row, k)
        return recs

    def _top_ads(self, scores: sp.csr_matrix, weights: sp.csr_matrix, row: int, k: int) -> list[int]:
        """Выбирает k лучших объявлений из строки скоров."""
        cols = scores.indices[scores.indptr[row] : scores.indptr[row + 1]]
        values = scores.data[scores.indptr[row] : scores.indptr[row + 1]]
        if self.exclude_seen:
            fresh = ~np.isin(cols, weights.indices[weights.indptr[row] : weights.indptr[row + 1]])
            cols, values = cols[fresh], values[fresh]
        return self.ads[cols[np.lexsort((cols, -values))[:k]]].tolist()


class BlendGenerator(Generator):
    """Склейка: берёт выдачу первого генератора и дозаполняет её следующими без повторов."""

    def __init__(self, generators: Sequence[Generator]):
        """Запоминает генераторы в порядке приоритета."""
        self.generators = list(generators)

    def _fit(self, events: pd.DataFrame) -> None:
        """Обучает все генераторы."""
        for generator in self.generators:
            generator.fit(events, self.cutoff)

    def recommend(self, users: Sequence[int], k: int) -> dict[int, list[int]]:
        """Склеивает выдачи по порядку и обрезает результат до k."""
        parts = [generator.recommend(users, k) for generator in self.generators]
        return {user: list(dict.fromkeys(ad for part in parts for ad in part[user]))[:k] for user in users}
