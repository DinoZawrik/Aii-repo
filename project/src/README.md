# Исходный код проекта

## Структура модулей

```
src/
├── __init__.py           # Главный модуль проекта
├── data/                 # Работа с данными
│   ├── loader.py         # Загрузка датасетов (CSV, train/test split)
│   └── preprocessor.py   # Предобработка текстов + TF-IDF
├── models/               # ML модели
│   ├── baseline.py       # LogisticRegression, RandomForest
│   └── transformer.py    # DistilBERT для sentiment analysis
├── training/             # Обучение
│   └── train.py          # Pipeline обучения + CLI
├── service/              # API
│   └── app.py            # FastAPI приложение
└── utils/                # Утилиты
    ├── config.py         # Конфигурация из .env
    └── logging_config.py # Structlog логирование
```

## Команды запуска

### Обучение модели
```bash
python -m src.training.train --model-type baseline --data-file sentiment_data.csv
```

### Запуск API
```bash
python -m src.service.app
```

## Основные классы

| Модуль | Класс | Описание |
|--------|-------|----------|
| `data.loader` | `DataLoader` | Загрузка CSV, train/test split |
| `data.preprocessor` | `TextPreprocessor` | Очистка текста |
| `data.preprocessor` | `TfidfFeatureExtractor` | TF-IDF векторизация |
| `models.baseline` | `BaselineModel` | Sklearn классификаторы |
| `models.transformer` | `TransformerModel` | HuggingFace модели |
| `training.train` | `TrainingPipeline` | Pipeline обучения |
| `utils.config` | `Config` | Конфигурация |
