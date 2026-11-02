"""Загрузка rel-avito из RelBench и подготовка таблицы взаимодействий.

Схема хранения на диске:
    data/raw/db/<Table>.parquet        таблицы базы
    data/raw/task/<split>.parquet      таблицы задачи user-ad-visit (train/val/test)
    data/raw/meta.json                 ключи, колонки времени, параметры задачи
    data/processed/interactions.parquet  все события user -> ad в одном формате

После одного скачивания весь остальной код работает с parquet и сеть не нужна.
"""
from __future__ import annotations

import json

import pandas as pd

from avito_recsys import config


def download(dataset_name: str = config.DATASET_NAME, task_name: str = config.TASK_NAME) -> None:
    """Скачивает датасет и задачу из RelBench и сохраняет их в parquet и meta.json."""
    import relbench  # импорт здесь, чтобы остальной код не зависел от relbench

    ds = relbench.load_dataset(dataset_name)
    db = ds.get_db(upto_test_timestamp=False)
    task = ds.load_task(task_name)

    db_dir = config.RAW_DIR / "db"
    task_dir = config.RAW_DIR / "task"
    db_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    tables_meta = {}
    for name, table in db.table_dict.items():
        table.df.to_parquet(db_dir / f"{name}.parquet", index=False)
        tables_meta[name] = {
            "pkey": table.pkey_col,
            "time_col": table.time_col,
            "fkeys": dict(table.fkey_col_to_pkey_table),
            "rows": len(table.df),
        }

    for split in ["train", "val", "test"]:
        df = task.get_table(split, mask_input_cols=False).df
        df.to_parquet(task_dir / f"{split}.parquet", index=False)

    meta = {
        "tables": tables_meta,
        "val_timestamp": str(ds.val_timestamp),
        "test_timestamp": str(ds.test_timestamp),
        "task": {
            "src_col": task.src_entity_col,
            "dst_col": task.dst_entity_col,
            "time_col": task.time_col,
            "eval_k": int(task.eval_k),
            "timedelta": str(task.timedelta),
        },
    }
    (config.RAW_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_meta() -> dict:
    """Читает ключи, колонки времени и параметры задачи из meta.json."""
    return json.loads((config.RAW_DIR / "meta.json").read_text())


def load_table(name: str) -> pd.DataFrame:
    """Читает таблицу базы по имени."""
    return pd.read_parquet(config.RAW_DIR / "db" / f"{name}.parquet")


def load_task_table(split: str) -> pd.DataFrame:
    """Читает таблицу задачи для сплита train, val или test."""
    return pd.read_parquet(config.RAW_DIR / "task" / f"{split}.parquet")


def _fkey_col(table_meta: dict, target_table: str) -> str:
    """Возвращает колонку внешнего ключа на target_table, ошибка если она не единственная."""
    cols = [c for c, t in table_meta["fkeys"].items() if t == target_table]
    if len(cols) != 1:
        raise ValueError(f"expected one fkey to {target_table}, got {cols}")
    return cols[0]


def build_interactions(tables: dict[str, pd.DataFrame], meta: dict) -> pd.DataFrame:
    """Склеивает таблицы событий в одну: user_id, ad_id, ts, event, weight.

    Названия колонок берём из метаданных RelBench, а не хардкодим.
    """
    parts = []
    for name, weight in config.EVENT_TABLES.items():
        tm = meta["tables"][name]
        df = tables[name]
        user_col = _fkey_col(tm, config.USER_TABLE)
        ad_col = _fkey_col(tm, config.AD_TABLE)
        # RelBench ставит NA вместо ссылок на сущности, которых нет в базе.
        # Такие события не к кому привязать, поэтому выбрасываем их.
        valid = df[user_col].notna() & df[ad_col].notna()
        if not valid.all():
            print(f"  {name}: dropped {(~valid).sum():,} of {len(df):,} rows with missing user/ad")
        df = df[valid]
        part = pd.DataFrame({
            "user_id": df[user_col].astype("int64"),
            "ad_id": df[ad_col].astype("int64"),
            "ts": pd.to_datetime(df[tm["time_col"]]),
        })
        part["event"] = name
        part["weight"] = weight
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("ts", kind="stable").reset_index(drop=True)


def prepare() -> pd.DataFrame:
    """Собирает таблицу взаимодействий из сырых данных и сохраняет её в parquet."""
    meta = load_meta()
    tables = {name: load_table(name) for name in config.EVENT_TABLES}
    interactions = build_interactions(tables, meta)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(config.PROCESSED_DIR / "interactions.parquet", index=False)
    return interactions


def load_interactions() -> pd.DataFrame:
    """Читает готовую таблицу взаимодействий."""
    return pd.read_parquet(config.PROCESSED_DIR / "interactions.parquet")
