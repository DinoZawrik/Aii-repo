"""
Transformer model implementation.

This module implements sentiment analysis using pre-trained transformer models
from Hugging Face (e.g., BERT, DistilBERT, RoBERTa).
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.

        Args:
            texts: List of text strings.
            labels: List of integer labels.
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from dataset.

        Args:
            idx: Index of the item.

        Returns:
            Dictionary with input_ids, attention_mask, and labels.
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class TransformerModel:
    """Transformer-based sentiment analysis model."""

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        num_labels: int = 2,
        max_length: int = 512
    ):
        """
        Initialize transformer model.

        Args:
            model_name: Name of pre-trained model from Hugging Face.
            num_labels: Number of output classes.
            max_length: Maximum sequence length.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        # Initialize tokenizer and model
        logger.info("Loading tokenizer and model", model_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info("Model initialized", device=str(self.device))

    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        output_dir: str = 'artifacts/models/transformer',
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ) -> Dict[str, Any]:
        """
        Train the transformer model.

        Args:
            train_texts: Training texts.
            train_labels: Training labels.
            val_texts: Validation texts (optional).
            val_labels: Validation labels (optional).
            output_dir: Directory to save model.
            num_epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.

        Returns:
            Dictionary with training history.
        """
        logger.info(
            "Starting training",
            train_size=len(train_texts),
            val_size=len(val_texts) if val_texts else 0,
            epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )

        eval_dataset = None
        if val_texts and val_labels:
            eval_dataset = SentimentDataset(
                val_texts, val_labels, self.tokenizer, self.max_length
            )

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            evaluation_strategy='epoch' if eval_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model='accuracy' if eval_dataset else None,
        )

        # Define compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary'
            )
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics if eval_dataset else None
        )

        # Train
        train_result = trainer.train()

        logger.info("Training complete")

        return {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics.get('train_runtime', 0),
        }

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions on texts.

        Args:
            texts: List of texts.

        Returns:
            Predicted labels.
        """
        logger.info("Making predictions", num_texts=len(texts))

        # Create dataset
        # Use dummy labels for prediction
        dummy_labels = [0] * len(texts)
        dataset = SentimentDataset(
            texts, dummy_labels, self.tokenizer, self.max_length
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=32)

        # Predict
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        logger.info("Predictions complete")
        return np.array(predictions)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            texts: List of texts.

        Returns:
            Class probabilities.
        """
        logger.info("Predicting probabilities", num_texts=len(texts))

        # Create dataset
        dummy_labels = [0] * len(texts)
        dataset = SentimentDataset(
            texts, dummy_labels, self.tokenizer, self.max_length
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=32)

        # Predict
        self.model.eval()
        probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())

        logger.info("Probability prediction complete")
        return np.array(probabilities)

    def evaluate(
        self,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_texts: Test texts.
            test_labels: Test labels.

        Returns:
            Dictionary with evaluation metrics.
        """
        logger.info("Evaluating model", test_size=len(test_texts))

        predictions = self.predict(test_texts)
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='binary'
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
        Save model and tokenizer.

        Args:
            save_path: Path to save directory.
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        logger.info("Model saved", path=str(save_dir))

    def load(self, load_path: str) -> None:
        """
        Load model and tokenizer.

        Args:
            load_path: Path to load directory.
        """
        load_dir = Path(load_path)

        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(load_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_dir)
        self.model.to(self.device)

        logger.info("Model loaded", path=str(load_dir))
