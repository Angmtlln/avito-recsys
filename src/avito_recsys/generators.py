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


class HistoryGenerator(Generator):
    """Объявления из истории пользователя: вес события затухает с давностью.

    Вес падает вдвое каждые half_life_days дней, поэтому свежий просмотр важнее старого.
    """

    def __init__(self, half_life_days: float = 3.0):
        """Запоминает период полураспада веса в днях."""
        self.half_life_days = half_life_days

    def _fit(self, events: pd.DataFrame) -> None:
        """Складывает затухающие веса по парам и сортирует объявления каждого пользователя."""
        age_days = (self.cutoff - events.ts).dt.total_seconds() / SECONDS_PER_DAY
        scored = events[["user_id", "ad_id"]].assign(score=events.weight * 0.5 ** (age_days / self.half_life_days))
        scored = scored.groupby(["user_id", "ad_id"], as_index=False).score.sum()
        scored = scored.sort_values(["user_id", "score", "ad_id"], ascending=[True, False, True])
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
