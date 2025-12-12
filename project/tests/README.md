# Тесты проекта

## Структура тестов

```
tests/
├── __init__.py           # Описание тест-пакета
├── conftest.py           # Pytest fixtures
├── test_preprocessor.py  # Тесты предобработки текста
├── test_models.py        # Тесты ML моделей
└── test_api.py           # Тесты FastAPI endpoints
```

## Запуск тестов

```bash
cd project

# Все тесты
pytest tests/ -v

# С покрытием
pytest tests/ --cov=src --cov-report=html

# Конкретный файл
pytest tests/test_models.py -v
```

## Описание тестов

### test_preprocessor.py
- Тесты `TextPreprocessor`: lowercase, удаление URL/email/пунктуации
- Тесты `TfidfFeatureExtractor`: fit, transform, fit_transform

### test_models.py
- Тесты `BaselineModel`: инициализация, train, predict, evaluate, save/load
- Поддержка LogisticRegression и RandomForest

### test_api.py
- Тесты endpoints: `/`, `/health`, `/predict`, `/predict/batch`
- Валидация request/response схем
