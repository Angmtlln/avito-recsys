"""Подбирает параметры item2item и сравнивает его с бейзлайнами на val. Test здесь не используется."""
import time

import pandas as pd

from avito_recsys import data
from avito_recsys.evaluate import evaluate
from avito_recsys.generators import (
    BlendGenerator,
    Generator,
    HistoryGenerator,
    Item2ItemGenerator,
    SegmentPopularGenerator,
)

METRICS = ["MAP@12", "NDCG@12", "Recall@12", "Recall@50", "Recall@100", "Recall@200"]
BEST_NEIGHBORS = 1000
BEST_SHRINK = 20


def build_generators(ads: pd.DataFrame) -> dict[str, Generator]:
    """Собирает item2item с разными параметрами и его склейки с бейзлайнами."""
    region = ["CategoryID", "RegionID"]
    generators: dict[str, Generator] = {"history": HistoryGenerator()}
    for neighbors in [50, 200, 1000]:
        generators[f"i2i neighbors={neighbors}"] = Item2ItemGenerator(top_neighbors=neighbors)
    generators["i2i neighbors=1000 shrink=20"] = Item2ItemGenerator(top_neighbors=1000, shrink=20)
    generators["i2i cap=25"] = Item2ItemGenerator(max_user_ads=25)
    generators["i2i with seen"] = Item2ItemGenerator(exclude_seen=False)

    def best_i2i() -> Item2ItemGenerator:
        """Создаёт item2item с лучшими параметрами."""
        return Item2ItemGenerator(top_neighbors=BEST_NEIGHBORS, shrink=BEST_SHRINK)

    generators["history+i2i"] = BlendGenerator([HistoryGenerator(), best_i2i()])
    generators["history+region"] = BlendGenerator([HistoryGenerator(), SegmentPopularGenerator(ads, region)])
    generators["history+i2i+region"] = BlendGenerator([
        HistoryGenerator(), best_i2i(), SegmentPopularGenerator(ads, region),
    ])
    generators["history+region+i2i"] = BlendGenerator([
        HistoryGenerator(), SegmentPopularGenerator(ads, region), best_i2i(),
    ])
    return generators


def main() -> None:
    """Оценивает все варианты на val и печатает таблицу."""
    interactions = data.load_interactions()
    val = data.load_task_table("val")
    rows = {}
    for name, generator in build_generators(data.load_ads()).items():
        start = time.perf_counter()
        rows[name] = evaluate(generator, interactions, val)
        print(f"{name}: {time.perf_counter() - start:.1f}s Recall@200={rows[name]['Recall@200']:.4f}", flush=True)
    table = pd.DataFrame(rows).T[METRICS]
    print(table.sort_values("Recall@200", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
