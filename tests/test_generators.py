import pandas as pd
import pytest

from avito_recsys.evaluate import evaluate
from avito_recsys.generators import (
    ALSGenerator,
    BlendGenerator,
    HistoryGenerator,
    Item2ItemGenerator,
    PopularGenerator,
    SegmentPopularGenerator,
)

CUTOFF = pd.Timestamp("2015-05-08")


def make_events(rows: list[tuple[int, int, float, float]]) -> pd.DataFrame:
    """Строит события из строк (user_id, ad_id, дней до CUTOFF, вес)."""
    return pd.DataFrame({
        "user_id": [row[0] for row in rows],
        "ad_id": [row[1] for row in rows],
        "ts": [CUTOFF - pd.Timedelta(days=row[2]) for row in rows],
        "weight": [row[3] for row in rows],
    })


@pytest.mark.parametrize("generator", [
    HistoryGenerator(),
    PopularGenerator(),
    SegmentPopularGenerator(pd.DataFrame({"AdID": [1, 2], "CategoryID": [10, 10]})),
])
def test_events_at_or_after_cutoff_are_ignored(generator):
    """Событие ровно в момент среза уже относится к будущему."""
    events = make_events([(1, 1, 1, 1.0), (1, 2, 0, 1.0), (1, 2, -1, 1.0)])
    recs = generator.fit(events, CUTOFF).recommend([1], 5)
    assert recs[1] == [1]


def test_history_prefers_recent_events():
    events = make_events([(1, 1, 10, 1.0), (1, 2, 1, 1.0)])
    recs = HistoryGenerator(half_life_days=3).fit(events, CUTOFF).recommend([1], 2)
    assert recs[1] == [2, 1]


def test_history_decay_by_hand():
    """Через один период полураспада вес падает вдвое: 3 * 0.5 = 1.5 больше 1 * 1."""
    events = make_events([(1, 1, 0.0001, 1.0), (1, 2, 3, 3.0)])
    recs = HistoryGenerator(half_life_days=3).fit(events, CUTOFF).recommend([1], 2)
    assert recs[1] == [2, 1]


def test_history_sums_repeated_events_and_has_no_duplicates():
    events = make_events([(1, 1, 1, 1.0), (1, 2, 1, 1.0), (1, 2, 1, 1.0)])
    recs = HistoryGenerator().fit(events, CUTOFF).recommend([1], 5)
    assert recs[1] == [2, 1]


def test_history_unknown_user_gets_nothing():
    events = make_events([(1, 1, 1, 1.0)])
    assert HistoryGenerator().fit(events, CUTOFF).recommend([2], 5) == {2: []}


def test_popular_uses_only_recent_window():
    """Объявление 1 популярно, но все его события старше окна."""
    events = make_events([(1, 1, 10, 1.0), (2, 1, 10, 1.0), (1, 2, 1, 1.0)])
    recs = PopularGenerator(days=7).fit(events, CUTOFF).recommend([1], 5)
    assert recs[1] == [2]


def test_popular_counts_weights_and_breaks_ties_by_id():
    events = make_events([(1, 3, 1, 1.0), (2, 2, 1, 1.0), (3, 1, 1, 3.0)])
    recs = PopularGenerator().fit(events, CUTOFF).recommend([9], 3)
    assert recs[9] == [1, 2, 3]


def test_segment_popular_recommends_inside_favorite_segments():
    """Пользователь 1 смотрел категорию 10, поэтому популярное объявление 3 из категории 20 не получает."""
    ads = pd.DataFrame({"AdID": [1, 2, 3], "CategoryID": [10, 10, 20]})
    events = make_events([
        (1, 1, 1, 1.0),
        (2, 2, 1, 1.0),
        (2, 2, 1, 1.0),
        (3, 3, 1, 5.0),
    ])
    generator = SegmentPopularGenerator(ads, top_segments=1).fit(events, CUTOFF)
    assert generator.recommend([1], 5)[1] == [2, 1]


def test_segment_popular_score_by_hand():
    """Доли категорий 0.75 и 0.25. Скоры: объявление 1: 0.75 * 3, объявление 2: 0.25 * 10 больше."""
    ads = pd.DataFrame({"AdID": [1, 2], "CategoryID": [10, 20]})
    events = make_events([(1, 1, 1, 3.0), (2, 2, 1, 9.0), (1, 2, 1, 1.0)])
    generator = SegmentPopularGenerator(ads, top_segments=2).fit(events, CUTOFF)
    assert generator.recommend([1], 2)[1] == [2, 1]


def test_segment_popular_supports_several_columns():
    ads = pd.DataFrame({"AdID": [1, 2], "CategoryID": [10, 10], "RegionID": [1, 2]})
    events = make_events([(1, 1, 1, 1.0), (2, 2, 1, 5.0)])
    generator = SegmentPopularGenerator(ads, segment_cols=["CategoryID", "RegionID"]).fit(events, CUTOFF)
    assert generator.recommend([1], 5)[1] == [1]


def test_segment_popular_cold_user_gets_nothing():
    ads = pd.DataFrame({"AdID": [1], "CategoryID": [10]})
    events = make_events([(1, 1, 1, 1.0)])
    assert SegmentPopularGenerator(ads).fit(events, CUTOFF).recommend([2], 5) == {2: []}


def coview_events() -> pd.DataFrame:
    """Объявление 1 смотрели четверо. С объявлением 2 оно встречалось дважды, с объявлением 3 один раз."""
    return make_events([
        (1, 1, 1, 1.0), (1, 2, 1, 1.0),
        (2, 1, 1, 1.0), (2, 2, 1, 1.0),
        (3, 1, 1, 1.0), (3, 3, 1, 1.0),
        (4, 1, 1, 1.0),
    ])


def test_item2item_ranks_by_cosine_and_excludes_seen():
    """Сходства с объявлением 1: 2/sqrt(4*2) = 0.71 у объявления 2 и 1/sqrt(4*1) = 0.5 у объявления 3."""
    generator = Item2ItemGenerator().fit(coview_events(), CUTOFF)
    assert generator.recommend([4], 5)[4] == [2, 3]


def test_item2item_similarity_values_by_hand():
    """Объявление 1 смотрели четверо, 2 двое, 3 один, поэтому сходства 2/sqrt(8) и 1/sqrt(4)."""
    generator = Item2ItemGenerator().fit(coview_events(), CUTOFF)
    neighbors = generator.similarity[0].toarray().ravel()
    assert neighbors[1] == pytest.approx(2 / 8 ** 0.5)
    assert neighbors[2] == pytest.approx(0.5)
    assert neighbors[0] == 0


def test_item2item_can_keep_seen_ads():
    generator = Item2ItemGenerator(exclude_seen=False).fit(coview_events(), CUTOFF)
    assert 2 in generator.recommend([1], 5)[1]


def test_item2item_top_neighbors_limits_similarity():
    generator = Item2ItemGenerator(top_neighbors=1).fit(coview_events(), CUTOFF)
    assert generator.recommend([4], 5)[4] == [2]


def test_item2item_max_user_ads_keeps_best_scored():
    """Старое объявление 1 не входит в лимит, поэтому связи 1-2 не появляется."""
    events = make_events([(1, 1, 30, 1.0), (1, 2, 1, 1.0), (2, 1, 1, 1.0)])
    generator = Item2ItemGenerator(max_user_ads=1).fit(events, CUTOFF)
    assert generator.recommend([2], 5)[2] == []


def test_item2item_shrink_lowers_similarity():
    generator = Item2ItemGenerator(shrink=2.0).fit(coview_events(), CUTOFF)
    assert generator.similarity[0, 2] == pytest.approx(1 / (4 ** 0.5 + 2))


def test_item2item_ignores_future_events():
    """Совместный просмотр в момент среза ещё не должен создавать связь."""
    events = make_events([(1, 1, 1, 1.0), (1, 2, 0, 1.0), (2, 1, 1, 1.0)])
    generator = Item2ItemGenerator().fit(events, CUTOFF)
    assert generator.recommend([2], 5)[2] == []


def test_item2item_unknown_user_and_batches():
    generator = Item2ItemGenerator(batch_size=1).fit(coview_events(), CUTOFF)
    recs = generator.recommend([9, 4, 1], 5)
    assert recs[9] == []
    assert recs[4] == [2, 3]
    assert recs[1] == [3]


def two_block_events() -> pd.DataFrame:
    """Две группы вкусов: пользователи 1-20 смотрят объявления 1-5, пользователи 21-40 смотрят 6-10.

    Пользователь 1 не видел объявление 5, а пользователь 21 не видел объявление 10.
    """
    rows = []
    for user in range(1, 21):
        rows += [(user, ad, 1, 1.0) for ad in range(1, 6) if not (user == 1 and ad == 5)]
    for user in range(21, 41):
        rows += [(user, ad, 1, 1.0) for ad in range(6, 11) if not (user == 21 and ad == 10)]
    return make_events(rows)


def small_als(**kwargs) -> ALSGenerator:
    """Создаёт маленькую модель для тестов."""
    return ALSGenerator(factors=4, regularization=0.1, iterations=10, num_threads=1, **kwargs)


def test_als_recommends_inside_user_taste():
    generator = small_als().fit(two_block_events(), CUTOFF)
    recs = generator.recommend([1, 21], 1)
    assert recs[1] == [5]
    assert recs[21] == [10]


def test_als_excludes_seen_and_has_no_duplicates():
    generator = small_als().fit(two_block_events(), CUTOFF)
    recs = generator.recommend([1], 100)[1]
    assert not {1, 2, 3, 4} & set(recs)
    assert len(recs) == len(set(recs)) == 6


def test_als_can_keep_seen_ads():
    generator = small_als(exclude_seen=False).fit(two_block_events(), CUTOFF)
    assert set(generator.recommend([1], 5)[1]) == {1, 2, 3, 4, 5}


def test_als_drops_ads_with_too_few_users():
    events = pd.concat([two_block_events(), make_events([(1, 99, 1, 1.0)])])
    generator = small_als().fit(events, CUTOFF)
    assert 99 not in generator.ads


def test_als_ignores_future_events():
    events = pd.concat([two_block_events(), make_events([(1, 77, 0, 1.0), (2, 77, -1, 1.0)])])
    generator = small_als().fit(events, CUTOFF)
    assert 77 not in generator.ads


def test_als_unknown_user_gets_nothing_and_batches_work():
    generator = small_als(batch_size=1).fit(two_block_events(), CUTOFF)
    recs = generator.recommend([999, 1, 21], 1)
    assert recs == {999: [], 1: [5], 21: [10]}


def test_blend_fills_without_duplicates_and_cuts_to_k():
    events = make_events([(1, 1, 1, 1.0), (2, 1, 1, 5.0), (2, 2, 1, 2.0), (2, 3, 1, 1.0)])
    blend = BlendGenerator([HistoryGenerator(), PopularGenerator()]).fit(events, CUTOFF)
    assert blend.recommend([1], 2)[1] == [1, 2]
    assert blend.recommend([1], 3)[1] == [1, 2, 3]
    assert blend.recommend([9], 2)[9] == [1, 2]


def test_evaluate_matches_hand_count():
    """Выдача пользователя 1: [2, 1], ответ [1]: AP = 1/2, полнота 1 на обоих порогах."""
    events = make_events([(1, 1, 5, 1.0), (1, 2, 1, 1.0)])
    task_slice = pd.DataFrame({"UserID": [1], "timestamp": [CUTOFF], "AdID": [[1]]})
    result = evaluate(HistoryGenerator(), events, task_slice, recall_ks=(12, 50))
    assert result["MAP@12"] == pytest.approx(0.5)
    assert result["Recall@12"] == pytest.approx(1.0)
    assert result["Recall@50"] == pytest.approx(1.0)


def test_evaluate_rejects_several_timestamps():
    """Срезы train надо оценивать по одному, иначе пользователи смешаются."""
    task_slice = pd.DataFrame({
        "UserID": [1, 1],
        "timestamp": [CUTOFF, CUTOFF + pd.Timedelta(days=4)],
        "AdID": [[1], [2]],
    })
    with pytest.raises(ValueError):
        evaluate(HistoryGenerator(), make_events([(1, 1, 1, 1.0)]), task_slice)
