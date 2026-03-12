Домашняя работа №5
# Taxi Trip Time Prediction

## Описание

Решение задачи регрессии на датасете **ECML/PKDD 15 — Porto Taxi Trajectory**.  
Цель: предсказать длительность поездки такси в секундах по данным GPS-трека.

**Метрика:** RMSLE (Root Mean Squared Log Error)

---

## Результаты

| Модель | RMSLE | MAE (сек) | R² |
|---|---|---|---|
| Dummy (baseline) | 0.6699 | 322.3 | 0.000 |
| RandomForest | 0.2811 | 146.0 | 0.824 |
| CatBoost | 0.2809 | 148.4 | 0.824 |
| XGBoost | 0.2632 | 137.5 | 0.846 |
| LightGBM | 0.2616 | 137.7 | 0.848 |
| LightGBM (tuned) | 0.2534 | 132.5 | 0.857 |
| **Stacking** | **0.2528** | **132.1** | **0.858** |

Улучшение vs Dummy: **62.3% по RMSLE**

---

## Данные

**Источник:** [Kaggle — PKDD 15: Taxi Trip Time Prediction](https://www.kaggle.com/c/pkdd-15-taxi-trip-time-prediction-ii)

Датасет содержит ~1.7 млн поездок такси в Порту (Португалия) за 2013–2014 гг.

**Ключевые поля:**

| Поле | Описание |
|---|---|
| `TRIP_ID` | Уникальный идентификатор поездки |
| `CALL_TYPE` | Тип вызова (A — центр, B — стоянка, C — улица) |
| `TAXI_ID` | Идентификатор водителя |
| `TIMESTAMP` | Время начала поездки (Unix) |
| `DAY_TYPE` | Тип дня (A — рабочий, B — праздник, C — канун) |
| `POLYLINE` | GPS-координаты маршрута (каждые 15 сек) |

**Целевая переменная:** `trip_time = (len(POLYLINE) - 1) × 15` секунд

---

## Feature Engineering

**Временные признаки:**
- `hour`, `dow`, `month` — час, день недели, месяц
- `is_weekend`, `is_rush` — бинарные флаги
- `hour_sin/cos`, `dow_sin/cos` — циклические проекции

**Географические признаки:**
- `start_lon/lat`, `end_lon/lat` — координаты начала и конца
- `direct_dist_km` — расстояние по прямой (Haversine)
- `total_dist_km` — суммарная длина маршрута
- `straightness` — прямолинейность (direct / total)
- `mean_lon/lat` — центр маршрута
- `log_direct_dist`, `log_total_dist` — логарифмированные дистанции
- `bbox_area` — площадь ограничивающего прямоугольника

**Категориальные признаки:**
- `call_type_enc`, `day_type_enc` — LabelEncoder
- `taxi_freq` — frequency encoding водителя

> `n_points` и `speed_proxy` исключены из признаков — они вычисляются из target (leakage)

---

## Модели

- **Dummy** — наивный baseline (среднее значение)
- **RandomForest** — 500 деревьев
- **LightGBM** — градиентный бустинг, победитель среди одиночных моделей
- **XGBoost** — градиентный бустинг
- **CatBoost** — градиентный бустинг
- **LightGBM (tuned)** — параметры подобраны через Optuna (50 trials)
- **Stacking** — мета-модель Ridge поверх OOF-предсказаний всех моделей (кроме Dummy)

---