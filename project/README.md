# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

## Sentiment Analysis API - Анализ тональности текстов

---

## 1. Паспорт проекта

- **Название проекта:** Sentiment Analysis API
- **Автор:** Новиков Максим Петрович
- **Группа:** БСБО-05-23
- **Контакт:** @telegram

- **Краткое описание:**
  Проект посвящён построению сервиса анализа тональности текстовых отзывов.
  Используются методы машинного обучения (TF-IDF + Logistic Regression) для классификации текстов на положительные и отрицательные.
  Результат – REST API, который принимает текст и возвращает предсказание тональности с уровнем уверенности.

---

## 2. Структура проекта

```
project/
├── README.md                 # Этот файл
├── report.md                 # Отчёт по проекту
├── self-checklist.md         # Чеклист самопроверки
├── requirements.txt          # Зависимости Python
├── Dockerfile                # Docker образ
├── docker-compose.yml        # Docker Compose конфигурация
├── .env.example              # Пример переменных окружения
├── .gitignore                # Игнорируемые файлы
│
├── src/                      # Основной код
│   ├── __init__.py
│   ├── data/                 # Работа с данными
│   │   ├── loader.py         # Загрузка датасетов
│   │   └── preprocessor.py   # Предобработка текстов
│   ├── models/               # ML модели
│   │   ├── baseline.py       # Baseline модель (LogReg, RF)
│   │   └── transformer.py    # Transformer модель
│   ├── training/             # Обучение
│   │   └── train.py          # Скрипт обучения
│   ├── service/              # API сервис
│   │   └── app.py            # FastAPI приложение
│   └── utils/                # Утилиты
│       ├── config.py         # Конфигурация
│       └── logging_config.py # Логирование
│
├── data/                     # Данные
│   └── sentiment_data.csv    # Демо-датасет
│
├── notebooks/                # Jupyter ноутбуки
│   ├── 01_eda.ipynb          # Разведочный анализ
│   └── 02_model_experiments.ipynb  # Эксперименты с моделями
│
├── configs/                  # Конфигурации
│   ├── model_config.yaml     # Конфиг моделей
│   └── training_config.yaml  # Конфиг обучения
│
├── tests/                    # Тесты
│   ├── test_preprocessor.py  # Тесты предобработки
│   ├── test_models.py        # Тесты моделей
│   └── test_api.py           # Тесты API
│
└── artifacts/                # Артефакты
    └── models/               # Сохранённые модели
```

---

## 3. Требования и установка

### 3.1. Требования

- Python >= 3.10
- pip или uv для управления зависимостями

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Создать виртуальное окружение
python -m venv .venv

# Активировать окружение:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3. Переменные окружения

Скопируйте `.env.example` в `.env` и настройте при необходимости:

```bash
cp .env.example .env
```

Основные переменные:
- `MODEL_TYPE` - тип модели (baseline/transformer)
- `MODEL_PATH` - путь к сохранённой модели
- `API_HOST` - хост API (0.0.0.0)
- `API_PORT` - порт API (8000)

---

## 4. Как запустить проект

### 4.1. Обучение модели

```bash
cd project
source .venv/bin/activate  # или .venv\Scripts\activate на Windows

# Обучение baseline модели (Logistic Regression)
python -m src.training.train --model-type baseline --data-file sentiment_data.csv

# Или с Random Forest
python -m src.training.train --model-type baseline --baseline-algo random_forest
```

### 4.2. Запуск сервиса (локально)

```bash
cd project
source .venv/bin/activate

# Запуск API сервера
python -m src.service.app
```

Сервис будет доступен на http://localhost:8000

### 4.3. Запуск через Docker

```bash
cd project

# Сборка образа
docker build -t sentiment-api .

# Запуск контейнера
docker run -p 8000:8000 sentiment-api
```

Или через Docker Compose:

```bash
docker-compose up --build
```

---

## 5. API Endpoints

### GET /health
Проверка здоровья сервиса.

```bash
curl http://localhost:8000/health
```

Ответ:
```json
{
  "status": "healthy",
  "model_type": "baseline",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### POST /predict
Предсказание тональности для одного текста.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

Ответ:
```json
{
  "text": "This product is amazing!",
  "sentiment": "positive",
  "confidence": 0.95,
  "label": 1,
  "latency_ms": 12.5
}
```

### POST /predict/batch
Пакетное предсказание для нескольких текстов.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible service."]}'
```

### Swagger UI
Интерактивная документация API: http://localhost:8000/docs

---

## 6. Данные

Для обучения используется демонстрационный датасет `data/sentiment_data.csv`:
- 100 текстовых отзывов
- Сбалансированные классы (50% положительных, 50% отрицательных)
- Колонки: `text` (текст отзыва), `label` (0 = негативный, 1 = позитивный)

---

## 7. Тесты

```bash
cd project
source .venv/bin/activate

# Запуск всех тестов
pytest tests/ -v

# Запуск с coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 8. Демонстрация на защите

План демонстрации:

1. **Структура проекта** - показать организацию кода в `src/`
2. **EDA ноутбук** - `notebooks/01_eda.ipynb` с анализом данных
3. **Эксперименты** - `notebooks/02_model_experiments.ipynb` со сравнением моделей
4. **Запуск сервиса** - `python -m src.service.app`
5. **Тестирование API** - запросы через Swagger UI на `/docs`
6. **Тесты** - запуск `pytest tests/ -v`

---

## 9. Ограничения и дальнейшая работа

### Текущие ограничения:
- Небольшой демо-датасет (100 записей)
- Только бинарная классификация (positive/negative)
- Нет аутентификации в API

### Возможные улучшения:
- Добавить transformer модель (BERT/DistilBERT)
- Расширить датасет реальными отзывами
- Добавить нейтральный класс
- Реализовать мониторинг и метрики (Prometheus)
- Добавить авторизацию (JWT)

---
