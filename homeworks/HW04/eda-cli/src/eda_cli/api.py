"""
FastAPI application for dataset quality assessment.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_basic_stats, compute_quality_flags

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ---------- /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (compute_basic_stats + compute_quality_flags)
    и возвращает оценку качества данных.

    Связывает CLI EDA из HW03 с HTTP-сервисом из HW04.
    """

    start = perf_counter()

    # Проверка content type (мягкая проверка, т.к. браузеры могут отправлять разное)
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream", None):
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv)."
        )

    try:
        # FastAPI даёт file.file как file-like объект для pandas
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать CSV: {exc}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV-файл не содержит данных (пустой DataFrame)."
        )

    # Используем EDA-ядро из HW03
    basic_stats = compute_basic_stats(df)
    quality_flags = compute_quality_flags(df)

    # compute_quality_flags возвращает quality_score в диапазоне [0, 100]
    # Преобразуем в [0, 1]
    score = quality_flags.get("quality_score", 0.0) / 100.0
    score = max(0.0, min(1.0, score))

    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    # Включаем как Python bool, так и numpy bool (np.bool_)
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in quality_flags.items()
        if isinstance(value, (bool, np.bool_))
    }

    # Размеры датасета
    n_rows = basic_stats["n_rows"]
    n_cols = basic_stats["n_cols"]

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv: Новый эндпоинт из HW04 ----------


class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества данных."""

    flags: dict[str, bool] = Field(
        ...,
        description="Полный набор булевых флагов качества данных, включая эвристики из HW03",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Интегральная оценка качества данных (0..100)",
    )
    details: dict = Field(
        default_factory=dict,
        description="Дополнительная информация о флагах (списки проблемных колонок и т.д.)",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )


@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Полный набор флагов качества данных из HW03",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Эндпоинт, который принимает CSV-файл и возвращает полный набор
    флагов качества данных, включая новые эвристики из HW03:

    - has_missing: наличие пропущенных значений
    - has_duplicates: наличие дубликатов строк
    - has_constant_columns: наличие колонок с одинаковыми значениями
    - has_high_cardinality_categoricals: категориальные признаки с большим числом уникальных значений
    - has_many_zero_values: числовые колонки с большой долей нулей
    - has_outliers: наличие выбросов в числовых колонках

    Этот эндпоинт реализован в рамках HW04 и использует доработки из HW03.
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream", None):
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv)."
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать CSV: {exc}"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV-файл не содержит данных (пустой DataFrame)."
        )

    # Используем compute_quality_flags из HW03
    quality_flags = compute_quality_flags(df)

    # Извлекаем булевые флаги
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in quality_flags.items()
        if isinstance(value, (bool, np.bool_))
    }

    # Собираем детали (списки проблемных колонок и т.д.)
    details = {}

    if quality_flags.get("constant_columns"):
        details["constant_columns"] = quality_flags["constant_columns"]

    if quality_flags.get("high_cardinality_categoricals"):
        details["high_cardinality_categoricals"] = [
            item["column"] for item in quality_flags["high_cardinality_categoricals"]
        ]

    if quality_flags.get("zero_heavy_columns"):
        details["zero_heavy_columns"] = [
            {"column": item["column"], "zero_rate": item["zero_rate"]}
            for item in quality_flags["zero_heavy_columns"]
        ]

    if quality_flags.get("outlier_columns"):
        details["outlier_columns"] = [
            {"column": item["column"], "n_outliers": item["n_outliers"]}
            for item in quality_flags["outlier_columns"]
        ]

    if quality_flags.get("missing_by_column"):
        problematic_missing = {
            col: rate for col, rate in quality_flags["missing_by_column"].items()
            if rate > 0.05
        }
        if problematic_missing:
            details["high_missing_columns"] = problematic_missing

    latency_ms = (perf_counter() - start) * 1000.0

    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"score={quality_flags.get('quality_score', 0):.1f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityFlagsResponse(
        flags=flags_bool,
        quality_score=quality_flags.get("quality_score", 0.0),
        details=details,
        latency_ms=latency_ms,
    )
