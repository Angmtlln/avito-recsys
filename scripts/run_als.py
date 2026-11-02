"""Подбирает параметры ALS и сравнивает его с другими генераторами на val. Test здесь не используется.

Без флагов меняет параметры по одному относительно базового варианта, это занимает около 11 минут.
С флагом --final считает лучшие сочетания и склейки с другими генераторами.
"""
import argparse
import time

import pandas as pd

from avito_recsys import data
from avito_recsys.evaluate import evaluate
from avito_recsys.generators import (
    ALSGenerator,
    BlendGenerator,
    Generator,
    HistoryGenerator,
    Item2ItemGenerator,
    SegmentPopularGenerator,
)

METRICS = ["MAP@12", "NDCG@12", "Recall@12", "Recall@50", "Recall@100", "Recall@200"]


def build_grid() -> dict[str, Generator]:
    """Собирает ALS с разными параметрами. Меняем по одному параметру относительно базового варианта."""
    generators: dict[str, Generator] = {"als base": ALSGenerator()}
    for factors in [32, 128]:
        generators[f"als factors={factors}"] = ALSGenerator(factors=factors)
    for regularization in [1, 100]:
        generators[f"als reg={regularization}"] = ALSGenerator(regularization=regularization)
    for alpha in [10, 100]:
        generators[f"als alpha={alpha}"] = ALSGenerator(alpha=alpha)
    generators["als half_life=30d"] = ALSGenerator(half_life_days=30)
    generators["als min_ad_users=3"] = ALSGenerator(min_ad_users=3)
    return generators


def build_final(ads: pd.DataFrame) -> dict[str, Generator]:
    """Собирает лучшие сочетания параметров ALS и его склейки с другими генераторами."""

    def i2i() -> Item2ItemGenerator:
        """Создаёт item2item с параметрами, подобранными на шаге 5."""
        return Item2ItemGenerator(top_neighbors=1000, shrink=20)

    def als() -> ALSGenerator:
        """Создаёт ALS с лучшими параметрами из сетки."""
        return ALSGenerator(factors=128, alpha=100)

    def region() -> SegmentPopularGenerator:
        """Создаёт популярное в категории и регионе."""
        return SegmentPopularGenerator(ads, ["CategoryID", "RegionID"])

    return {
        "als factors=128 alpha=100": als(),
        "als factors=256 alpha=100": ALSGenerator(factors=256, alpha=100),
        "history+i2i+region": BlendGenerator([HistoryGenerator(), i2i(), region()]),
        "history+als+region": BlendGenerator([HistoryGenerator(), als(), region()]),
        "history+i2i+als+region": BlendGenerator([HistoryGenerator(), i2i(), als(), region()]),
        "history+als+i2i+region": BlendGenerator([HistoryGenerator(), als(), i2i(), region()]),
    }


def main() -> None:
    """Оценивает варианты на val и печатает таблицу."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="evaluate best combinations and blends")
    args = parser.parse_args()

    interactions = data.load_interactions()
    val = data.load_task_table("val")
    generators = build_final(data.load_ads()) if args.final else build_grid()
    rows = {}
    for name, generator in generators.items():
        start = time.perf_counter()
        rows[name] = evaluate(generator, interactions, val)
        print(f"{name}: {time.perf_counter() - start:.1f}s Recall@200={rows[name]['Recall@200']:.4f}", flush=True)
    table = pd.DataFrame(rows).T[METRICS]
    print(table.sort_values("Recall@200", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
