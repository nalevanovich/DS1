Домашняя работа №4

# Sentiment Analysis of Tweets

Анализ тональности твитов на основе датасета [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (1.6 млн твитов). Проект включает полный ML-пайплайн: от загрузки и предобработки данных до обучения и сравнения нескольких классификаторов.

---

## Структура проекта
```
├── data_loader.py          # Загрузка и описание датасета
├── data_processor.py       # Извлечение признаков и очистка текста
├── notebook.ipynb          # Основной ноутбук
├── catboost_final.cbm      # Сохранённая модель CatBoost
├── lgbm_sentiment.txt      # Сохранённая модель LightGBM
├── xgb_sentiment.json      # Сохранённая модель XGBoost
├── sgd_sentiment.pkl       # Сохранённая модель Linear SVM (SGD)
├── dt_sentiment.pkl        # Сохранённая модель Decision Tree
├── tfidf.pkl               # Сохранённый TF-IDF векторизатор
├── svd.pkl                 # Сохранённый SVD трансформер
└── scaler.pkl              # Сохранённый StandardScaler
```

---

## Датасет

**Sentiment140** — 1.6 млн твитов, размеченных по тональности:
- `0` — негативный твит
- `1` — позитивный твит (в оригинале `4`, заменяется при загрузке)

Для обучения использовалась стратифицированная выборка **400 000 твитов** (200k на класс).

---

## Пайплайн

### 1. Предобработка текста
- Удаление ссылок, упоминаний (`@user`), пунктуации
- Приведение к нижнему регистру
- Удаление стоп-слов (NLTK)

### 2. Извлечение признаков
Из сырого текста (до очистки) извлекаются числовые признаки:
- `has_happy` — наличие позитивного смайлика
- `has_sad` — наличие негативного смайлика
- `char_count` — длина текста в символах
- `word_count` — количество слов
- `is_caps` — количество слов капсом
- `excl_count` — количество восклицательных знаков

### 3. Векторизация
- **TF-IDF** (`max_features=50000`, `ngram_range=(1, 2)`)
- **SVD** (`n_components=400`) — сжатие для градиентного бустинга
- Итоговая матрица признаков: SVD (400) + числовые (6) = **406 признаков**

### 4. Модели
| Модель | Данные | Подбор параметров |
|--------|--------|-------------------|
| CatBoost | `cleaned_text` + числовые (сырые) | Optuna |
| LightGBM | SVD + числовые | Optuna |
| XGBoost | SVD + числовые | Optuna |
| Linear SVM (SGD) | SVD + числовые (нормализованные) | Optuna |
| Decision Tree | SVD + числовые | Optuna |

---

## Результаты

| Модель | Accuracy | F1 (weighted) |
|--------|----------|---------------|
| CatBoost | ~0.78 | ~0.78 |
| LightGBM | ~0.75 | ~0.75 |
| XGBoost | ~0.74 | ~0.74 |
| Linear SVM (SGD) | ~0.74 | ~0.74 |
| Decision Tree | ~0.67 | ~0.67 |

> CatBoost показал лучший результат благодаря встроенной обработке текстовых признаков через `text_features`.

---

## Установка
```bash
pip install catboost lightgbm xgboost scikit-learn pandas numpy matplotlib seaborn nltk joblib optuna
```

---

## Запуск

1. Скачай датасет с [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. Положи файл `training.1600000.processed.noemoticon.csv` в папку проекта
3. Запусти все клетки ноутбука `notebook.ipynb` по порядку

---

## Технологии

- **Python 3.12**
- **CatBoost, LightGBM, XGBoost** — градиентный бустинг
- **scikit-learn** — SGD, Decision Tree, TF-IDF, SVD, метрики
- **Optuna** — подбор гиперпараметров
- **NLTK** — стоп-слова
- **Matplotlib, Seaborn** — визуализация