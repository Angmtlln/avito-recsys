# avito-recsys

Двухэтапная рекомендательная система на реальных данных Avito: кандидатогенерация и ранжирование.

**Задача.** Для каждого пользователя предсказать объявления, которые он посмотрит в ближайшие 4 дня.
Это задача `user-ad-visit` из бенчмарка [RelBench](https://relbench.stanford.edu/datasets/rel-avito),
метрика MAP@k, разбиение по времени.

**Данные.** `rel-avito` собран из соревнования Avito Context Ad Clicks. Основные таблицы:
просмотры объявлений (`VisitStream`), запросы телефона продавца (`PhoneRequestsStream`),
объявления (`AdsInfo`), поиски (`SearchInfo`), пользователи, категории и регионы.

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_data.py
```

## План

- [x] Загрузка данных и таблица взаимодействий
- [ ] Метрики и EDA
- [ ] Бейзлайны: история пользователя, популярное
- [ ] Кандидатогенераторы: item2item, ALS, эмбеддинги заголовков
- [ ] Ранжирование на CatBoost
- [ ] Сервис FastAPI + gRPC, Airflow DAG, Docker
