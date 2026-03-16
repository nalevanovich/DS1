# Walmart M5 — Анализ временного ряда продаж

Учебный проект по анализу временных рядов на датасете Walmart M5 (Kaggle).  
Магазин: **CA_1** (Калифорния). Период: **2011-01-29 — 2016-06-19**.

---

## Структура проекта

```
📁 проект/
├── 📓 notebook.ipynb          ← главный ноутбук
│
├── 📁 data/                   ← данные с Kaggle (не включены в репозиторий)
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   └── sell_prices.csv
│
├── data_loader.py             ← загрузка и описание данных
├── eda.py                     ← разведочный анализ
├── preprocessing.py           ← предобработка
├── features.py                ← инженерия признаков
├── anomaly.py                 ← обнаружение аномалий
├── metrics.py                 ← метрики качества
├── visualization.py           ← все графики и дашборд
│
└── 📁 models/
    ├── arima_model.py
    ├── prophet_model.py
    ├── linear_model.py
    ├── rf_model.py
    ├── xgb_model.py
    ├── adaboost_model.py
    └── lstm_model.py
```

---

## Данные

Датасет **Walmart M5 Forecasting** с Kaggle:  
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Скачать 3 файла и положить в папку `data/`:
- `sales_train_validation.csv` — продажи 30 490 товаров за 1913 дней
- `calendar.csv` — даты, праздники, SNAP-флаги
- `sell_prices.csv` — цены товаров по неделям

**Почему один магазин CA_1:**  
Полный датасет (58 млн строк) требует production-инфраструктуры.  
CA_1 содержит все характерные паттерны: тренд, недельную сезонность,  
влияние праздников и SNAP — достаточно для полного анализа.

---

## Установка

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install statsmodels prophet xgboost torch optuna
```

---

## Пайплайн (порядок ячеек в ноутбуке)

| Шаг | Модуль | Что делает |
|-----|--------|------------|
| 1 | `data_loader` | Загрузка, merge sales + calendar + prices |
| 2 | `eda` | Иерархия, сезонность, события, аномалии, декомпозиция |
| 3 | `preprocessing` | Заполнение пропусков, флаг событий |
| 4 | `anomaly` | Z-score, IQR, Isolation Forest, STL |
| 5 | `features` | Лаги, скользящие статистики, временные признаки |
| 6 | Модели | 7 моделей + тюнинг + стэккинг |
| 7 | `metrics` | MAE, RMSE, MAPE, R² |
| 8 | `visualization` | Residuals, важность признаков, дашборд |

---

## Модели

| Модель | MAPE% | R² | Особенности |
|--------|-------|----|-------------|
| **LinearReg** | **4.7%** | **0.91** | Лучшая — лаги линейно объясняют продажи |
| XGBoost | 5.1% | 0.91 | Тюнинг Optuna не улучшил дефолт |
| RandomForest | 5.6% | 0.88 | Стабильная, интерпретируемая |
| LSTM | 7.2% | 0.71 | Видит только сырой ряд без признаков |
| AdaBoost | 7.2% | 0.82 | Слабее бустинга на деревьях |
| ARIMA | 9.4% | 0.70 | order=(7,1,1), только автокорреляции |
| Prophet | 13.7% | 0.14 | Не подходит для горизонта 28 дней |

---

## Ключевые находки

**Признаки:**
- `is_weekend` — самый важный признак (XGBoost, RF)
- `lag_28`, `lag_35` — продажи месяц назад сильно коррелируют
- `is_anomaly` — флаг STL-аномалий вошёл в топ-6 признаков

**Аномалии (STL-метод, 38 дней):**
- 5 дней закрытия магазина — Christmas, Thanksgiving
- 38 дней нетипичных продаж — праздники, акции, SNAP

**Стэккинг:**
- Не дал улучшения — корреляция предсказаний всех моделей 0.90-0.99
- Все модели выучили одинаковый паттерн (недельная сезонность)

**Тюнинг XGBoost (Optuna, 100 trials):**
- Default MAPE 5.05% → Tuned MAPE 5.22% — дефолт оказался лучше
- Причина: Optuna оптимизировал под CV, тест имеет другой паттерн

---

## Инженерия признаков

```python
# Лаги (кратно 7 — недельная сезонность)
lag_7, lag_14, lag_28, lag_35, lag_42

# Скользящие статистики (с .shift(1) против утечки данных)
ma_7, ma_14, ma_28
std_7, std_14, std_28
min_7, max_7, ...

# Временные признаки
day_of_week, is_weekend, month, quarter
sin_wday, cos_wday  # цикличность

# Событийные
has_event, event_lag1, event_lead1
is_anomaly  # STL-флаг
```

---

## Требования

- Python 3.10+
- PyTorch (CPU достаточно для LSTM на этом датасете)
- Prophet устанавливается отдельно: `pip install prophet`
- Optuna: `pip install optuna`