"""
Training pipeline for sentiment analysis models.

This module provides the main training pipeline for both baseline
and transformer models.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from ..data.loader import DataLoader
from ..data.preprocessor import TextPreprocessor, TfidfFeatureExtractor
from ..models.baseline import BaselineModel
from ..models.transformer import TransformerModel
from ..utils.config import Config
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Training pipeline for sentiment analysis."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize training pipeline.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or Config()
        logger.info("Training pipeline initialized", config=self.config.to_dict())

    def train_baseline(
        self,
        data_file: str,
        model_type: str = 'logistic_regression',
        test_size: float = 0.2,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train baseline model.

        Args:
            data_file: Path to data file.
            model_type: Type of baseline model.
            test_size: Proportion for test set.
            save_path: Path to save trained model.

        Returns:
            Dictionary with training results and metrics.
        """
        logger.info("Starting baseline model training", model_type=model_type)

        # Load data
        loader = DataLoader()
        train_texts, test_texts, train_labels, test_labels = loader.load_train_test_split(
            data_file, test_size=test_size
        )

        # Preprocess text
        preprocessor = TextPreprocessor()
        train_texts_clean = preprocessor.preprocess_batch(train_texts)
        test_texts_clean = preprocessor.preprocess_batch(test_texts)

        # Extract features
        feature_extractor = TfidfFeatureExtractor()
        X_train = feature_extractor.fit_transform(train_texts_clean)
        X_test = feature_extractor.transform(test_texts_clean)

        # Train model
        model = BaselineModel(model_type=model_type)
        train_metrics = model.train(X_train, np.array(train_labels))

        # Evaluate
        test_metrics = model.evaluate(X_test, np.array(test_labels))

        # Save model
        if save_path is None:
            save_path = f"artifacts/models/{model_type}"
        model.save(save_path)

        # Save feature extractor
        import pickle
        feature_path = Path(save_path) / 'feature_extractor.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_extractor, f)

        # Save preprocessor config
        preprocessor_config = {
            'lowercase': preprocessor.lowercase,
            'remove_punctuation': preprocessor.remove_punctuation,
            'remove_numbers': preprocessor.remove_numbers,
            'remove_extra_spaces': preprocessor.remove_extra_spaces
        }
        config_path = Path(save_path) / 'preprocessor_config.json'
        with open(config_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)

        results = {
            'model_type': model_type,
            'train_size': len(train_texts),
            'test_size': len(test_texts),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'save_path': save_path
        }

        logger.info("Baseline training complete", results=results)
        return results

    def train_transformer(
        self,
        data_file: str,
        model_name: str = 'distilbert-base-uncased',
        test_size: float = 0.2,
        val_size: float = 0.1,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train transformer model.

        Args:
            data_file: Path to data file.
            model_name: Name of pre-trained transformer.
            test_size: Proportion for test set.
            val_size: Proportion for validation set (from training data).
            save_path: Path to save trained model.

        Returns:
            Dictionary with training results and metrics.
        """
        logger.info("Starting transformer model training", model_name=model_name)

        # Load data
        loader = DataLoader()
        train_texts, test_texts, train_labels, test_labels = loader.load_train_test_split(
            data_file, test_size=test_size
        )

        # Split training data into train and validation
        if val_size > 0:
            from sklearn.model_selection import train_test_split
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=val_size, random_state=42
            )
        else:
            val_texts, val_labels = None, None

        # Initialize model
        model = TransformerModel(
            model_name=model_name,
            max_length=self.config.max_length
        )

        # Train
        if save_path is None:
            save_path = "artifacts/models/transformer"

        train_result = model.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            output_dir=save_path,
            num_epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate
        )

        # Evaluate on test set
        test_metrics = model.evaluate(test_texts, test_labels)

        # Save model
        model.save(save_path)

        results = {
            'model_name': model_name,
            'train_size': len(train_texts),
            'val_size': len(val_texts) if val_texts else 0,
            'test_size': len(test_texts),
            'train_result': train_result,
            'test_metrics': test_metrics,
            'save_path': save_path
        }

        logger.info("Transformer training complete", results=results)
        return results


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['baseline', 'transformer'],
        default='baseline',
        help='Type of model to train'
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='sample/sentiment_data.csv',
        help='Path to data file'
    )
    parser.add_argument(
        '--baseline-algo',
        type=str,
        choices=['logistic_regression', 'random_forest'],
        default='logistic_regression',
        help='Algorithm for baseline model'
    )
    parser.add_argument(
        '--transformer-name',
        type=str,
        default='distilbert-base-uncased',
        help='Name of transformer model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for model'
    )

    args = parser.parse_args()

    # Setup logging
    from ..utils.logging_config import setup_logging
    setup_logging()

    # Create pipeline
    pipeline = TrainingPipeline()

    # Train model
    if args.model_type == 'baseline':
        results = pipeline.train_baseline(
            data_file=args.data_file,
            model_type=args.baseline_algo,
            test_size=args.test_size,
            save_path=args.output_dir
        )
    else:
        results = pipeline.train_transformer(
            data_file=args.data_file,
            model_name=args.transformer_name,
            test_size=args.test_size,
            save_path=args.output_dir
        )

    # Print results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
