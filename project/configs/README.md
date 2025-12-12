# Конфигурационные файлы

## Содержание

### model_config.yaml
Параметры моделей:
- Baseline: `model_type`, `max_iter`, TF-IDF настройки
- Transformer: `model_name`, `max_length`, training settings

### training_config.yaml
Параметры обучения:
- Эксперимент: `name`, `seed`
- Данные: `path`, `test_size`, `val_size`
- Метрики: `accuracy`, `precision`, `recall`, `f1`

### .env.example (в корне project/)
Переменные окружения:
- `MODEL_TYPE` - тип модели (baseline/transformer)
- `MODEL_PATH` - путь к модели
- `API_HOST`, `API_PORT` - настройки сервера
- `LOG_LEVEL` - уровень логирования

## Использование

```python
from src.utils.config import Config, get_config

# Загрузка из .env
config = get_config()

# Загрузка YAML
model_config = Config.load_yaml('configs/model_config.yaml')
```

## Важно

- **НЕ** храните реальные секреты в `.env` файлах в репозитории
- `.env` добавлен в `.gitignore`
- Используйте `.env.example` как шаблон
