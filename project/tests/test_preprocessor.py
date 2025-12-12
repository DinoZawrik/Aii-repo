"""
Tests for data preprocessing module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import TextPreprocessor, TfidfFeatureExtractor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.preprocessor = TextPreprocessor()

    def test_clean_text_lowercase(self):
        """Test lowercase conversion."""
        text = "HELLO WORLD"
        result = self.preprocessor.clean_text(text)
        assert result == "hello world"

    def test_clean_text_removes_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        result = self.preprocessor.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result

    def test_clean_text_removes_email(self):
        """Test email removal."""
        text = "Contact us at test@example.com"
        result = self.preprocessor.clean_text(text)
        assert "@" not in result
        assert "example.com" not in result

    def test_clean_text_removes_punctuation(self):
        """Test punctuation removal."""
        text = "Hello, World! How are you?"
        result = self.preprocessor.clean_text(text)
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_clean_text_removes_extra_spaces(self):
        """Test extra spaces removal."""
        text = "Hello    World   Test"
        result = self.preprocessor.clean_text(text)
        assert "  " not in result

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        texts = ["Hello World!", "TEST TEXT", "  Multiple   Spaces  "]
        result = self.preprocessor.preprocess_batch(texts)
        assert len(result) == 3
        assert all(isinstance(t, str) for t in result)

    def test_clean_text_preserves_numbers_by_default(self):
        """Test that numbers are preserved by default."""
        text = "Product 123 is great"
        result = self.preprocessor.clean_text(text)
        assert "123" in result

    def test_clean_text_removes_numbers_when_configured(self):
        """Test number removal when configured."""
        preprocessor = TextPreprocessor(remove_numbers=True)
        text = "Product 123 is great"
        result = preprocessor.clean_text(text)
        assert "123" not in result


class TestTfidfFeatureExtractor:
    """Tests for TfidfFeatureExtractor class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.extractor = TfidfFeatureExtractor(max_features=100)
        self.sample_texts = [
            "this is a great product",
            "terrible experience very bad",
            "amazing quality love it",
            "worst purchase ever made",
            "highly recommend excellent"
        ]

    def test_fit_transform(self):
        """Test fit_transform method."""
        features = self.extractor.fit_transform(self.sample_texts)
        assert features.shape[0] == len(self.sample_texts)
        assert features.shape[1] <= 100
        assert self.extractor.is_fitted

    def test_transform_after_fit(self):
        """Test transform after fitting."""
        self.extractor.fit(self.sample_texts)
        new_texts = ["new text to transform"]
        features = self.extractor.transform(new_texts)
        assert features.shape[0] == 1

    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        with pytest.raises(ValueError):
            self.extractor.transform(["some text"])

    def test_feature_values_are_normalized(self):
        """Test that TF-IDF values are in valid range."""
        features = self.extractor.fit_transform(self.sample_texts)
        assert features.min() >= 0
        assert features.max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
