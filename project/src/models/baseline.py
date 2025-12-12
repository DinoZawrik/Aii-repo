"""
Baseline model implementation.

This module implements baseline sentiment analysis models using traditional
machine learning algorithms (Logistic Regression, Random Forest).
"""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BaselineModel:
    """Baseline sentiment analysis model using sklearn classifiers."""

    def __init__(
        self,
        model_type: str = 'logistic_regression',
        **model_kwargs
    ):
        """
        Initialize baseline model.

        Args:
            model_type: Type of model ('logistic_regression' or 'random_forest').
            **model_kwargs: Additional arguments for the model.
        """
        self.model_type = model_type

        if model_type == 'logistic_regression':
            default_kwargs = {'max_iter': 1000, 'random_state': 42}
            default_kwargs.update(model_kwargs)
            self.model = LogisticRegression(**default_kwargs)
        elif model_type == 'random_forest':
            default_kwargs = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            default_kwargs.update(model_kwargs)
            self.model = RandomForestClassifier(**default_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.is_trained = False
        logger.info("Baseline model initialized", model_type=model_type)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, float]:
        """
        Train the baseline model.

        Args:
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(
            "Training baseline model",
            model_type=self.model_type,
            train_size=len(X_train)
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Compute training metrics
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)

        logger.info("Training complete", train_accuracy=f"{train_acc:.4f}")

        return {
            'train_accuracy': train_acc
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features.

        Returns:
            Predicted labels.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features.

        Returns:
            Class probabilities.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features.
            y_test: Test labels.

        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating model", test_size=len(X_test))

        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary'
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        logger.info(
            "Evaluation complete",
            accuracy=f"{accuracy:.4f}",
            precision=f"{precision:.4f}",
            recall=f"{recall:.4f}",
            f1=f"{f1:.4f}"
        )

        return metrics

    def save(self, save_path: str) -> None:
        """
        Save model to disk.

        Args:
            save_path: Path to save the model.
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_file = save_dir / 'model.pkl'

        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

        logger.info("Model saved", path=str(model_file))

    def load(self, load_path: str) -> None:
        """
        Load model from disk.

        Args:
            load_path: Path to load the model from.
        """
        model_file = Path(load_path) / 'model.pkl'

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)

        self.is_trained = True
        logger.info("Model loaded", path=str(model_file))
