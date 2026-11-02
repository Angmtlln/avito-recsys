"""Подбирает параметры бейзлайнов и сравнивает их на val. Test здесь не используется."""
import time

import pandas as pd

from avito_recsys import data
from avito_recsys.evaluate import evaluate
from avito_recsys.generators import (
    BlendGenerator,
    Generator,
    HistoryGenerator,
    PopularGenerator,
    SegmentPopularGenerator,
)

METRICS = ["MAP@12", "NDCG@12", "Recall@12", "Recall@50", "Recall@100", "Recall@200"]


def load_ads() -> pd.DataFrame:
    """Читает категорию и регион каждого объявления."""
    ads = data.load_table("AdsInfo")[["AdID", "CategoryID", "LocationID"]]
    locations = data.load_table("Location")[["LocationID", "RegionID"]]
    return ads.merge(locations, on="LocationID", how="left")


def build_generators(ads: pd.DataFrame) -> dict[str, Generator]:
    """Собирает отдельные бейзлайны с разными параметрами и их склейки."""
    category = ["CategoryID"]
    category_region = ["CategoryID", "RegionID"]
    generators: dict[str, Generator] = {}
    for half_life in [1, 3, 7, 30]:
        generators[f"history half_life={half_life}d"] = HistoryGenerator(half_life)
    for days in [1, 3, 7, 14]:
        generators[f"popular {days}d"] = PopularGenerator(days)
    for days in [3, 7, 14]:
        generators[f"category popular {days}d"] = SegmentPopularGenerator(ads, category, days=days)
        generators[f"category+region popular {days}d"] = SegmentPopularGenerator(ads, category_region, days=days)
    generators["blend history+popular"] = BlendGenerator([HistoryGenerator(), PopularGenerator()])
    generators["blend history+category+region+popular"] = BlendGenerator([
        HistoryGenerator(),
        SegmentPopularGenerator(ads, category_region),
        SegmentPopularGenerator(ads, category),
        PopularGenerator(),
    ])
    return generators


def main() -> None:
    """Оценивает все генераторы на val и печатает таблицу."""
    interactions = data.load_interactions()
    val = data.load_task_table("val")
    rows = {}
    for name, generator in build_generators(load_ads()).items():
        start = time.perf_counter()
        rows[name] = evaluate(generator, interactions, val)
        print(f"{name}: {time.perf_counter() - start:.1f}s MAP@12={rows[name]['MAP@12']:.4f}", flush=True)
    table = pd.DataFrame(rows).T[METRICS]
    print(table.sort_values("MAP@12", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
