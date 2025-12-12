# Ноутбуки проекта

## Содержание

### 01_eda.ipynb - Разведочный анализ данных
- Загрузка и описание датасета
- Распределение классов (визуализация)
- Анализ длины текстов
- Частотный анализ слов
- Выводы и рекомендации для моделирования

### 02_model_experiments.ipynb - Эксперименты с моделями
- Подготовка данных (train/test split)
- TF-IDF векторизация
- Обучение baseline моделей:
  - Logistic Regression
  - Random Forest
  - SVM (Linear)
- Cross-validation
- Сравнение моделей по метрикам
- Выбор финальной модели
- Сохранение модели

## Результаты экспериментов

| Модель | Accuracy | F1-Score |
|--------|----------|----------|
| Logistic Regression | ~0.95 | ~0.95 |
| Random Forest | ~0.90 | ~0.90 |
| SVM (Linear) | ~0.95 | ~0.95 |

**Финальная модель:** Logistic Regression

## Запуск ноутбуков

```bash
cd project
jupyter notebook notebooks/
```
