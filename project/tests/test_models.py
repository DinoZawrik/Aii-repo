"""
Tests for machine learning models.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline import BaselineModel


class TestBaselineModel:
    """Tests for BaselineModel class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 50)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 50)
        self.y_test = np.random.randint(0, 2, 20)

    def test_logistic_regression_initialization(self):
        """Test LogisticRegression model initialization."""
        model = BaselineModel(model_type='logistic_regression')
        assert model.model_type == 'logistic_regression'
        assert not model.is_trained

    def test_random_forest_initialization(self):
        """Test RandomForest model initialization."""
        model = BaselineModel(model_type='random_forest')
        assert model.model_type == 'random_forest'
        assert not model.is_trained

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises error."""
        with pytest.raises(ValueError):
            BaselineModel(model_type='invalid_model')

    def test_train_logistic_regression(self):
        """Test training LogisticRegression model."""
        model = BaselineModel(model_type='logistic_regression')
        metrics = model.train(self.X_train, self.y_train)

        assert model.is_trained
        assert 'train_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1

    def test_train_random_forest(self):
        """Test training RandomForest model."""
        model = BaselineModel(model_type='random_forest')
        metrics = model.train(self.X_train, self.y_train)

        assert model.is_trained
        assert 'train_accuracy' in metrics

    def test_predict_after_training(self):
        """Test prediction after training."""
        model = BaselineModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.X_test)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_without_training_raises_error(self):
        """Test that prediction without training raises error."""
        model = BaselineModel(model_type='logistic_regression')

        with pytest.raises(ValueError):
            model.predict(self.X_test)

    def test_predict_proba(self):
        """Test probability prediction."""
        model = BaselineModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)

        probas = model.predict_proba(self.X_test)

        assert probas.shape == (len(self.X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_evaluate(self):
        """Test model evaluation."""
        model = BaselineModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)

        metrics = model.evaluate(self.X_test, self.y_test)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_save_and_load(self):
        """Test model save and load."""
        model = BaselineModel(model_type='logistic_regression')
        model.train(self.X_train, self.y_train)

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Save model
            model.save(temp_dir)

            # Load model
            loaded_model = BaselineModel(model_type='logistic_regression')
            loaded_model.load(temp_dir)

            # Compare predictions
            original_preds = model.predict(self.X_test)
            loaded_preds = loaded_model.predict(self.X_test)

            assert np.array_equal(original_preds, loaded_preds)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_untrained_model_raises_error(self):
        """Test that saving untrained model raises error."""
        model = BaselineModel(model_type='logistic_regression')

        with pytest.raises(ValueError):
            model.save('/tmp/test_model')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
