import argparse

from avito_recsys import config, data


def main() -> None:
    """Скачивает данные, если их ещё нет, и собирает таблицу взаимодействий."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="download again even if data exists")
    args = parser.parse_args()

    if args.force or not (config.RAW_DIR / "meta.json").exists():
        print(f"Downloading {config.DATASET_NAME} / {config.TASK_NAME} ...")
        data.download()
    else:
        print("Raw data already downloaded, skipping (use --force to redownload)")

    meta = data.load_meta()
    for name, t in meta["tables"].items():
        print(f"  {name:22s} rows={t['rows']:>11,}  time_col={t['time_col']}  fkeys={t['fkeys']}")
    print("Task:", meta["task"])
    print("val/test timestamps:", meta["val_timestamp"], meta["test_timestamp"])

    interactions = data.prepare()
    print(f"Interactions: {len(interactions):,} rows, "
          f"{interactions.user_id.nunique():,} users, {interactions.ad_id.nunique():,} ads")
    print(interactions.groupby("event").size())


if __name__ == "__main__":
    main()
