from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # таблицы rel-avito как есть
PROCESSED_DIR = DATA_DIR / "processed"  # наши производные таблицы

DATASET_NAME = "rel-avito"
TASK_NAME = "user-ad-visit"

# Таблицы с событиями пользователь -> объявление и вес события.
# Запрос телефона продавца гораздо сильнее говорит об интересе, чем просмотр.
EVENT_TABLES = {
    "VisitStream": 1.0,
    "PhoneRequestsStream": 3.0,
}
USER_TABLE = "UserInfo"
AD_TABLE = "AdsInfo"
