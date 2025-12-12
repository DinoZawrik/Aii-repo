"""
Data loading module.

This module handles loading datasets from various sources (CSV, databases, etc.)
and preparing them for training and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Data loader for sentiment analysis datasets."""

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize data loader.

        Args:
            data_path: Path to data directory.
        """
        if data_path:
            self.data_path = Path(data_path)
        else:
            # Default to project data directory
            project_root = Path(__file__).parent.parent.parent
            self.data_path = project_root / 'data'

    def load_csv(
        self,
        filename: str,
        text_column: str = 'text',
        label_column: str = 'label',
        sep: str = ','
    ) -> Tuple[List[str], List[int]]:
        """
        Load dataset from CSV file.

        Args:
            filename: Name of the CSV file in data directory.
            text_column: Name of the column containing text data.
            label_column: Name of the column containing labels.
            sep: Separator character.

        Returns:
            Tuple of (texts, labels).
        """
        file_path = self.data_path / filename
        logger.info("Loading data from CSV", path=str(file_path))

        try:
            df = pd.read_csv(file_path, sep=sep)
            logger.info("Data loaded successfully", rows=len(df), columns=list(df.columns))

            # Validate columns
            if text_column not in df.columns:
                raise ValueError(f"Text column '{text_column}' not found in CSV")
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found in CSV")

            # Extract texts and labels
            texts = df[text_column].astype(str).tolist()
            labels = df[label_column].astype(int).tolist()

            # Log class distribution
            unique, counts = np.unique(labels, return_counts=True)
            class_dist = dict(zip(unique, counts))
            logger.info("Class distribution", distribution=class_dist)

            return texts, labels

        except Exception as e:
            logger.error("Failed to load data", error=str(e), path=str(file_path))
            raise

    def load_train_test_split(
        self,
        filename: str,
        test_size: float = 0.2,
        random_state: int = 42,
        text_column: str = 'text',
        label_column: str = 'label',
        stratify: bool = True
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Load data and split into train/test sets.

        Args:
            filename: Name of the CSV file.
            test_size: Proportion of data for testing.
            random_state: Random seed for reproducibility.
            text_column: Name of text column.
            label_column: Name of label column.
            stratify: Whether to stratify split by labels.

        Returns:
            Tuple of (train_texts, test_texts, train_labels, test_labels).
        """
        texts, labels = self.load_csv(filename, text_column, label_column)

        logger.info("Splitting data", test_size=test_size, stratify=stratify)

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if stratify else None
        )

        logger.info(
            "Data split complete",
            train_size=len(train_texts),
            test_size=len(test_texts)
        )

        return train_texts, test_texts, train_labels, test_labels

    def create_sample_dataset(
        self,
        output_path: str,
        num_samples: int = 1000
    ) -> None:
        """
        Create a sample sentiment dataset for demonstration.

        Args:
            output_path: Path to save the sample dataset.
            num_samples: Number of samples to generate.
        """
        logger.info("Creating sample dataset", num_samples=num_samples)

        # Positive sentiment examples
        positive_templates = [
            "This is an excellent product! Highly recommended.",
            "I absolutely love this! Best purchase ever.",
            "Amazing quality and great value for money.",
            "Fantastic! Exceeded all my expectations.",
            "Very satisfied with this purchase. Would buy again.",
            "Outstanding service and great product quality.",
            "Brilliant! This is exactly what I needed.",
            "Wonderful experience. Five stars all the way!",
            "I'm so happy with this! Totally worth it.",
            "Impressive quality. Highly recommended to everyone.",
        ]

        # Negative sentiment examples
        negative_templates = [
            "Terrible product. Complete waste of money.",
            "Very disappointed with this purchase.",
            "Poor quality and terrible customer service.",
            "Don't waste your money on this garbage.",
            "Awful experience. Would not recommend to anyone.",
            "This is the worst product I've ever bought.",
            "Completely useless. Total disappointment.",
            "Horrible! Nothing works as advertised.",
            "Save your money and look elsewhere.",
            "Extremely poor quality. Very dissatisfied.",
        ]

        # Generate samples
        texts = []
        labels = []

        for _ in range(num_samples // 2):
            # Positive sample
            texts.append(np.random.choice(positive_templates))
            labels.append(1)

            # Negative sample
            texts.append(np.random.choice(negative_templates))
            labels.append(0)

        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'label': labels
        })

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save
        output_file = self.data_path / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        logger.info("Sample dataset created", path=str(output_file), samples=len(df))
