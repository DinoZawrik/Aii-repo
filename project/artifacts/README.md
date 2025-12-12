# Артефакты проекта

## Структура

```
artifacts/
├── models/               # Сохранённые модели
│   └── logistic_regression/
│       ├── model.pkl           # Sklearn модель
│       ├── feature_extractor.pkl  # TF-IDF vectorizer
│       └── preprocessor_config.json  # Настройки препроцессора
└── (graphs, reports - генерируются ноутбуками)
```

## Генерация артефактов

### Обучение модели
```bash
python -m src.training.train --model-type baseline --data-file sentiment_data.csv
```

### Из ноутбуков
При запуске `notebooks/02_model_experiments.ipynb` модель сохраняется автоматически.

## Содержимое после обучения

| Файл | Описание |
|------|----------|
| `model.pkl` | Обученная sklearn модель |
| `feature_extractor.pkl` | TF-IDF vectorizer |
| `preprocessor_config.json` | Настройки предобработки |

## Примечание

- Большие файлы моделей (`.bin`, `.safetensors`) исключены из git через `.gitignore`
- Для transformer моделей используется отдельная папка `artifacts/models/transformer/`
