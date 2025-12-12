"""
Data preprocessing module.

This module handles text preprocessing including cleaning, normalization,
and feature extraction.
"""

import re
import string
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TextPreprocessor:
    """Text preprocessing for sentiment analysis."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_extra_spaces: bool = True
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Convert text to lowercase.
            remove_punctuation: Remove punctuation marks.
            remove_numbers: Remove numbers from text.
            remove_extra_spaces: Remove extra whitespace.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_spaces = remove_extra_spaces

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Input text.

        Returns:
            Cleaned text.
        """
        # Convert to string if not already
        text = str(text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra spaces
        if self.remove_extra_spaces:
            text = ' '.join(text.split())

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of cleaned texts.
        """
        logger.info("Preprocessing batch of texts", count=len(texts))
        cleaned_texts = [self.clean_text(text) for text in texts]
        logger.info("Preprocessing complete")
        return cleaned_texts


class TfidfFeatureExtractor:
    """TF-IDF feature extraction for baseline models."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialize TF-IDF feature extractor.

        Args:
            max_features: Maximum number of features.
            ngram_range: Range of n-grams to consider.
            min_df: Minimum document frequency.
            max_df: Maximum document frequency.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        self.is_fitted = False

    def fit(self, texts: List[str]) -> None:
        """
        Fit the vectorizer on training texts.

        Args:
            texts: List of training texts.
        """
        logger.info("Fitting TF-IDF vectorizer", num_texts=len(texts))
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(
            "TF-IDF vectorizer fitted",
            vocabulary_size=len(self.vectorizer.vocabulary_)
        )

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF features.

        Args:
            texts: List of texts to transform.

        Returns:
            TF-IDF feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")

        logger.info("Transforming texts to TF-IDF features", num_texts=len(texts))
        features = self.vectorizer.transform(texts).toarray()
        logger.info("Transformation complete", feature_shape=features.shape)
        return features

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts in one step.

        Args:
            texts: List of texts.

        Returns:
            TF-IDF feature matrix.
        """
        logger.info("Fitting and transforming TF-IDF", num_texts=len(texts))
        features = self.vectorizer.fit_transform(texts).toarray()
        self.is_fitted = True
        logger.info(
            "Fit-transform complete",
            feature_shape=features.shape,
            vocabulary_size=len(self.vectorizer.vocabulary_)
        )
        return features
