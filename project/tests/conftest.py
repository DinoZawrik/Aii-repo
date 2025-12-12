"""
Pytest configuration and fixtures for Sentiment Analysis tests.
"""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing."""
    return [
        "This product is amazing! I love it!",
        "Terrible experience. Would not recommend.",
        "Great quality and fast shipping.",
        "Worst purchase ever. Complete waste of money.",
        "Excellent service. Very satisfied.",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    """Sample labels for testing."""
    return [1, 0, 1, 0, 1]


@pytest.fixture(scope="session")
def positive_text():
    """Sample positive text."""
    return "This is an excellent product! Highly recommended!"


@pytest.fixture(scope="session")
def negative_text():
    """Sample negative text."""
    return "Terrible quality. Very disappointed with this purchase."
