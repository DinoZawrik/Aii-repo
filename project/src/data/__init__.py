"""
Data module for sentiment analysis.

This module provides data loading and preprocessing functionality.
"""

from .loader import DataLoader
from .preprocessor import TextPreprocessor, TfidfFeatureExtractor

__all__ = ["DataLoader", "TextPreprocessor", "TfidfFeatureExtractor"]
