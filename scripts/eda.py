"""Разведочный анализ событий и таблиц задачи. Выводы записаны в README, графики в docs/figures.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from avito_recsys import config, data

QUANTILES = [0.5, 0.9, 0.99, 1.0]
FIGURES_DIR = config.ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NEUTRAL = "#8a8983"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_MUTED,
    "axes.edgecolor": GRID,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "legend.frameon": False,
})


def print_header(title: str) -> None:
    """Печатает заголовок раздела."""
    print(f"\n=== {title} ===")


def save_figure(fig: plt.Figure, name: str) -> None:
    """Сохраняет график в docs/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def events_by_day(interactions: pd.DataFrame) -> pd.DataFrame:
    """Считает число событий каждого типа по дням."""
    days = interactions.ts.dt.normalize().rename("day")
    return interactions.groupby([days, "event"]).size().unstack(fill_value=0)


def print_time_range(interactions: pd.DataFrame) -> None:
    """Показывает границы данных и число событий по дням."""
    print_header("Time range")
    print(interactions.groupby("event").ts.agg(["min", "max", "size"]))
    print(events_by_day(interactions).to_string())


def plot_events_by_day(interactions: pd.DataFrame, cutoffs: dict[str, list[pd.Timestamp]]) -> None:
    """Рисует события по дням и моменты срезов. Шкалы разные, поэтому у каждого типа своя панель."""
    by_day = events_by_day(interactions)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    panels = [("VisitStream", "Просмотры в день, тыс."), ("PhoneRequestsStream", "Запросы телефона в день, тыс.")]
    for ax, (event, title) in zip(axes, panels):
        ax.plot(by_day.index, by_day[event] / 1000, color=BLUE, linewidth=2)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        for split, moments in cutoffs.items():
            for moment in moments:
                ax.axvline(moment, color=NEUTRAL, linewidth=1, linestyle="--")
                if ax is axes[0]:
                    ax.annotate(split, (moment, 1), xycoords=("data", "axes fraction"), xytext=(3, -2),
                                textcoords="offset points", va="top", color=TEXT_MUTED, fontsize=9)
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.tight_layout()
    save_figure(fig, "events_by_day")


def print_long_tail(interactions: pd.DataFrame, n_ads_total: int) -> None:
    """Показывает распределение активности пользователей и популярности объявлений."""
    print_header("Activity and popularity")
    for col in ["user_id", "ad_id"]:
        counts = interactions[col].value_counts()
        top_1pct = counts.iloc[: max(1, len(counts) // 100)].sum() / counts.sum()
        print(f"{col}: n={len(counts):,} quantiles={counts.quantile(QUANTILES).to_dict()} "
              f"single_event={(counts == 1).mean():.3f} top_1pct_share={top_1pct:.3f}")
    n_ads = interactions.ad_id.nunique()
    print(f"ads with any event: {n_ads:,} of {n_ads_total:,} ({n_ads / n_ads_total:.3f})")


def plot_long_tail(interactions: pd.DataFrame) -> None:
    """Рисует долю пользователей и объявлений, у которых не меньше n событий."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [("user_id", "Пользователи", BLUE, 300), ("ad_id", "Объявления", ORANGE, 30)]
    for col, label, color, label_at in series:
        n_events = interactions[col].value_counts().value_counts().sort_index()
        share_at_least = n_events[::-1].cumsum()[::-1] / n_events.sum()
        ax.step(share_at_least.index, share_at_least, where="post", color=color, linewidth=2, label=label)
        label_y = share_at_least[share_at_least.index >= label_at].iloc[0]
        ax.annotate(label, (label_at, label_y), xytext=(8, 8), textcoords="offset points", color=TEXT_MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Число событий, n")
    ax.set_ylabel("Доля с числом событий не меньше n")
    ax.set_title("Длинный хвост: у большинства объявлений всего одно событие")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, "long_tail")


def explode_truth(task_slice: pd.DataFrame) -> pd.DataFrame:
    """Разворачивает срез задачи в строки user_id, ad_id."""
    truth = task_slice[["UserID", "AdID"]].explode("AdID")
    truth.columns = ["user_id", "ad_id"]
    return truth.astype("int64")


def slice_stats(task_slice: pd.DataFrame, interactions: pd.DataFrame) -> dict:
    """Считает для среза долю повторов, холодных пользователей и холодных объявлений.

    Используются только события строго раньше момента среза.
    """
    cutoff = task_slice.timestamp.iloc[0]
    before = interactions.loc[interactions.ts < cutoff, ["user_id", "ad_id", "ts"]]
    last_seen = before.groupby(["user_id", "ad_id"], sort=False).ts.max().rename("last_seen").reset_index()

    truth = explode_truth(task_slice).merge(last_seen, on=["user_id", "ad_id"], how="left")
    truth["is_repeat"] = truth.last_seen.notna()
    truth["ad_known"] = truth.ad_id.isin(before.ad_id.unique())
    per_user = truth.groupby("user_id").is_repeat.mean()
    days_since = (cutoff - truth.last_seen.dropna()).dt.days

    return {
        "cutoff": cutoff.date(),
        "users": len(task_slice),
        "true_ads_per_user_median": task_slice.AdID.map(len).median(),
        "cold_users": 1 - task_slice.UserID.isin(before.user_id.unique()).mean(),
        "repeat_share_of_true_ads": truth.is_repeat.mean(),
        "repeat_share_per_user": per_user.mean(),
        "users_with_any_repeat": (per_user > 0).mean(),
        "repeats_seen_within_7d": (days_since < 7).mean(),
        "cold_ads_share_of_true_ads": 1 - truth.ad_known.mean(),
    }


def plot_slices(stats: pd.DataFrame) -> None:
    """Рисует состав правильных ответов и долю пользователей без истории на каждом срезе."""
    labels = [cutoff.strftime("%d.%m") for cutoff in stats.cutoff]
    cold = stats.cold_ads_share_of_true_ads * 100
    repeat = stats.repeat_share_of_true_ads * 100
    segments = [
        ("Объявление без событий до среза", cold, ORANGE, "white"),
        ("Объявление смотрели другие", 100 - cold - repeat, BLUE, "white"),
        ("Пользователь уже смотрел", repeat, AQUA, TEXT),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5.2), width_ratios=[3, 2])

    bottom = pd.Series(0.0, index=stats.index)
    for label, values, color, text_color in segments:
        bars = left.bar(labels, values, bottom=bottom, width=0.55, color=color, edgecolor=SURFACE,
                        linewidth=2, label=label)
        for bar, value in zip(bars, values):
            if value >= 4.5:
                left.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%",
                          ha="center", va="center", color=text_color, fontsize=9)
        bottom += values
    left.set_ylim(0, 100)
    left.set_ylabel("Доля правильных ответов, %")
    left.set_xlabel("Срез (последний — val)")
    left.set_title("Откуда берутся правильные ответы")
    left.grid(axis="x", visible=False)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, columnspacing=1, handlelength=1)

    bars = right.bar(labels, stats.cold_users * 100, width=0.55, color=NEUTRAL)
    right.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    right.set_ylim(0, 100)
    right.set_ylabel("Доля пользователей, %")
    right.set_xlabel("Срез (последний — val)")
    right.set_title("Пользователи без истории")
    right.grid(axis="x", visible=False)
    fig.tight_layout()
    save_figure(fig, "task_slices")


def print_user_overlap(train: pd.DataFrame, val: pd.DataFrame, test_users: pd.Series) -> None:
    """Показывает пересечение пользователей между сплитами."""
    print_header("User overlap")
    train_users = set(train.UserID)
    val_users = set(val.UserID)
    print(f"train={len(train_users):,} val={len(val_users):,} test={test_users.nunique():,}")
    print(f"val users also in train: {val.UserID.isin(train_users).mean():.3f}")
    print(f"test users also in train: {test_users.isin(train_users).mean():.3f}")
    print(f"test users also in val: {test_users.isin(val_users).mean():.3f}")


def print_segments(interactions: pd.DataFrame, ads: pd.DataFrame) -> None:
    """Показывает долю событий по категориям и регионам."""
    print_header("Categories and regions")
    events = interactions[["ad_id"]].merge(ads, left_on="ad_id", right_on="AdID", how="left")
    print(f"events without category: {events.CategoryID.isna().mean():.3f}, "
          f"without region: {events.RegionID.isna().mean():.3f}")
    for col in ["CategoryID", "RegionID"]:
        shares = events[col].value_counts(normalize=True)
        print(f"{col}: n={len(shares)} top5={shares.head(5).round(3).to_dict()} "
              f"top10_share={shares.head(10).sum():.3f}")


def plot_segments(interactions: pd.DataFrame, ads: pd.DataFrame) -> None:
    """Рисует десять самых частых категорий и регионов по доле событий."""
    events = interactions[["ad_id"]].merge(ads, left_on="ad_id", right_on="AdID", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    panels = [("CategoryID", "Топ-10 категорий"), ("RegionID", "Топ-10 регионов")]
    for ax, (col, title) in zip(axes, panels):
        shares = events[col].value_counts(normalize=True).head(10)[::-1] * 100
        bars = ax.barh([str(int(value)) for value in shares.index], shares, height=0.6, color=BLUE)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_xlim(0, shares.max() * 1.18)
        ax.set_xlabel("Доля событий, %")
        ax.set_ylabel("ID")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "top_segments")


def segment_affinity(val: pd.DataFrame, interactions: pd.DataFrame, ads: pd.DataFrame, col: str, top_n: int) -> float:
    """Считает долю ответов val в top_n значений col, которые пользователь смотрел до среза.

    Пользователи без истории в расчёт не входят.
    """
    cutoff = val.timestamp.iloc[0]
    before = interactions.loc[interactions.ts < cutoff, ["user_id", "ad_id"]]
    before = before[before.user_id.isin(val.UserID)].merge(ads, left_on="ad_id", right_on="AdID")
    counts = before.groupby(["user_id", col]).size().rename("n").reset_index()
    top = counts.sort_values(["user_id", "n"], ascending=[True, False]).groupby("user_id").head(top_n)

    truth = explode_truth(val)
    truth = truth[truth.user_id.isin(top.user_id)].merge(ads, left_on="ad_id", right_on="AdID")
    hits = truth.merge(top[["user_id", col]], on=["user_id", col], how="left", indicator=True)
    return (hits._merge == "both").mean()


def cold_ads_shown_in_search(val: pd.DataFrame, interactions: pd.DataFrame, search: pd.DataFrame) -> float:
    """Считает долю холодных ответов val, которые хотя бы раз показывались в поиске до среза."""
    cutoff = val.timestamp.iloc[0]
    true_ads = explode_truth(val).ad_id
    cold = true_ads[~true_ads.isin(interactions.loc[interactions.ts < cutoff, "ad_id"].unique())]
    shown = search.loc[pd.to_datetime(search.SearchDate) < cutoff, "AdID"].dropna().unique()
    return cold.isin(shown).mean()


def main() -> None:
    """Печатает все разделы анализа и сохраняет графики."""
    interactions = data.load_interactions()
    ads = data.load_table("AdsInfo")[["AdID", "CategoryID", "LocationID"]]
    ads = ads.merge(data.load_table("Location")[["LocationID", "RegionID"]], on="LocationID", how="left")
    search = data.load_table("SearchStream")[["AdID", "SearchDate"]]
    train = data.load_task_table("train")
    val = data.load_task_table("val")
    test = data.load_task_table("test")[["UserID", "timestamp"]]
    test_users = test.UserID

    print_time_range(interactions)
    plot_events_by_day(interactions, {
        "train": sorted(train.timestamp.unique()),
        "val": [val.timestamp.iloc[0]],
        "test": [test.timestamp.iloc[0]],
    })
    print_long_tail(interactions, n_ads_total=len(ads))
    plot_long_tail(interactions)
    print_user_overlap(train, val, test_users)

    print_header("Task slices (only events before each cutoff)")
    slices = [group for _, group in train.groupby("timestamp")] + [val]
    stats = pd.DataFrame([slice_stats(s, interactions) for s in slices])
    print(stats.T.to_string())
    plot_slices(stats)

    print_segments(interactions, ads)
    plot_segments(interactions, ads)
    print(f"val true ads in user's top-3 categories: {segment_affinity(val, interactions, ads, 'CategoryID', 3):.3f}")
    print(f"val true ads in user's top-1 region: {segment_affinity(val, interactions, ads, 'RegionID', 1):.3f}")
    print(f"cold val true ads shown in search before cutoff: {cold_ads_shown_in_search(val, interactions, search):.3f}")


if __name__ == "__main__":
    main()
