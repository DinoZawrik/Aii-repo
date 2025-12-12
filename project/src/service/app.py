"""
FastAPI application for sentiment analysis service.

This module implements the REST API for sentiment prediction using
trained models.
"""

import time
from pathlib import Path
from typing import Optional, List
import pickle
import json

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..models.baseline import BaselineModel
from ..models.transformer import TransformerModel
from ..data.preprocessor import TextPreprocessor
from ..utils.config import get_config
from ..utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using ML models",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_extractor = None
model_type = None


class PredictionRequest(BaseModel):
    """Request model for /predict endpoint."""
    text: str = Field(..., description="Text to analyze for sentiment")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is amazing! I love it!"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(..., description="List of texts to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "This is great!",
                    "This is terrible.",
                    "Not sure about this one."
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    text: str
    sentiment: str
    confidence: float
    label: int
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_type: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, preprocessor, feature_extractor, model_type

    config = get_config()
    model_type = config.model_type
    model_path = Path(config.model_path)

    logger.info("Loading model", model_type=model_type, path=str(model_path))

    try:
        if model_type == 'baseline' or model_type.startswith('logistic') or model_type.startswith('random'):
            # Load baseline model
            model = BaselineModel(model_type='logistic_regression')
            model.load(str(model_path))

            # Load feature extractor
            feature_path = model_path / 'feature_extractor.pkl'
            with open(feature_path, 'rb') as f:
                feature_extractor = pickle.load(f)

            # Load preprocessor config
            preprocessor_config_path = model_path / 'preprocessor_config.json'
            if preprocessor_config_path.exists():
                with open(preprocessor_config_path, 'r') as f:
                    preprocessor_config = json.load(f)
                preprocessor = TextPreprocessor(**preprocessor_config)
            else:
                preprocessor = TextPreprocessor()

            logger.info("Baseline model loaded successfully")

        elif model_type == 'transformer':
            # Load transformer model
            model = TransformerModel()
            model.load(str(model_path))
            logger.info("Transformer model loaded successfully")

        else:
            logger.warning("Unknown model type, using stub model", model_type=model_type)

    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        # Don't fail startup, just use stub mode
        model = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status including model information.
    """
    return HealthResponse(
        status="healthy",
        model_type=model_type or "none",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict sentiment for a single text.

    Args:
        request: Prediction request with text.

    Returns:
        Prediction response with sentiment and confidence.
    """
    start_time = time.time()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health endpoint."
        )

    try:
        logger.info("Processing prediction request", text_length=len(request.text))

        # Preprocess and predict based on model type
        if model_type == 'baseline' or model_type.startswith('logistic') or model_type.startswith('random'):
            # Baseline model
            clean_text = preprocessor.preprocess_batch([request.text])[0]
            features = feature_extractor.transform([clean_text])
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)

            label = int(predictions[0])
            confidence = float(probabilities[0][label])

        else:
            # Transformer model
            predictions = model.predict([request.text])
            probabilities = model.predict_proba([request.text])

            label = int(predictions[0])
            confidence = float(probabilities[0][label])

        # Map label to sentiment
        sentiment = "positive" if label == 1 else "negative"

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "Prediction complete",
            sentiment=sentiment,
            confidence=f"{confidence:.4f}",
            latency_ms=f"{latency_ms:.2f}"
        )

        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            label=label,
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts.

    Args:
        request: Batch prediction request with list of texts.

    Returns:
        Batch prediction response.
    """
    start_time = time.time()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health endpoint."
        )

    try:
        logger.info("Processing batch prediction", num_texts=len(request.texts))

        # Preprocess and predict
        if model_type == 'baseline' or model_type.startswith('logistic') or model_type.startswith('random'):
            clean_texts = preprocessor.preprocess_batch(request.texts)
            features = feature_extractor.transform(clean_texts)
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
        else:
            predictions = model.predict(request.texts)
            probabilities = model.predict_proba(request.texts)

        # Build response
        results = []
        for i, text in enumerate(request.texts):
            label = int(predictions[i])
            confidence = float(probabilities[i][label])
            sentiment = "positive" if label == 1 else "negative"

            results.append(PredictionResponse(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                label=label,
                latency_ms=0  # Individual latency not tracked in batch
            ))

        total_latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "Batch prediction complete",
            num_predictions=len(results),
            total_latency_ms=f"{total_latency_ms:.2f}"
        )

        return BatchPredictionResponse(
            predictions=results,
            total_latency_ms=total_latency_ms
        )

    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs"
        }
    }


def main():
    """Run the API server."""
    import uvicorn

    config = get_config()

    logger.info(
        "Starting API server",
        host=config.api_host,
        port=config.api_port
    )

    uvicorn.run(
        "src.service.app:app",
        host=config.api_host,
        port=config.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
