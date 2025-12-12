"""
Tests for FastAPI endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient

from eda_cli.api import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_health(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "dataset-quality"
    assert data["version"] == "0.2.0"


def test_quality_endpoint_valid(client):
    """Test /quality endpoint with valid data."""
    request_data = {
        "n_rows": 1000,
        "n_cols": 10,
        "max_missing_share": 0.05,
        "numeric_cols": 7,
        "categorical_cols": 3
    }

    response = client.post("/quality", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "quality_score" in data
    assert "ok_for_model" in data
    assert "message" in data
    assert "latency_ms" in data
    assert "flags" in data
    assert "dataset_shape" in data

    # Check score is in valid range [0, 1]
    assert 0.0 <= data["quality_score"] <= 1.0

    # Check latency is positive
    assert data["latency_ms"] >= 0.0

    # Check dataset_shape matches input
    assert data["dataset_shape"]["n_rows"] == 1000
    assert data["dataset_shape"]["n_cols"] == 10


def test_quality_endpoint_high_quality(client):
    """Test /quality with high-quality dataset parameters."""
    request_data = {
        "n_rows": 5000,
        "n_cols": 15,
        "max_missing_share": 0.01,
        "numeric_cols": 10,
        "categorical_cols": 5
    }

    response = client.post("/quality", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Should pass quality threshold (>= 0.7)
    assert data["ok_for_model"] is True
    assert data["quality_score"] >= 0.7


def test_quality_endpoint_low_quality(client):
    """Test /quality with low-quality dataset parameters."""
    request_data = {
        "n_rows": 100,  # Too few rows
        "n_cols": 150,  # Too many columns
        "max_missing_share": 0.6,  # Too many missing values
        "numeric_cols": 0,  # No numeric columns
        "categorical_cols": 150
    }

    response = client.post("/quality", json=request_data)
    assert response.status_code == 200

    data = response.json()
    # Should fail quality threshold
    assert data["ok_for_model"] is False
    assert data["quality_score"] < 0.7

    # Check flags indicate problems
    assert data["flags"]["too_few_rows"] is True
    assert data["flags"]["too_many_columns"] is True
    assert data["flags"]["too_many_missing"] is True


def test_quality_validation_error_negative_rows(client):
    """Test /quality with invalid negative rows."""
    request_data = {
        "n_rows": -100,  # Invalid: negative
        "n_cols": 10,
        "max_missing_share": 0.05,
        "numeric_cols": 7,
        "categorical_cols": 3
    }

    response = client.post("/quality", json=request_data)
    assert response.status_code == 422  # Validation error


def test_quality_validation_error_invalid_missing_share(client):
    """Test /quality with invalid max_missing_share > 1.0."""
    request_data = {
        "n_rows": 1000,
        "n_cols": 10,
        "max_missing_share": 1.5,  # Invalid: > 1.0
        "numeric_cols": 7,
        "categorical_cols": 3
    }

    response = client.post("/quality", json=request_data)
    assert response.status_code == 422  # Validation error


def test_quality_validation_error_missing_field(client):
    """Test /quality with missing required field."""
    request_data = {
        "n_rows": 1000,
        "n_cols": 10,
        # Missing max_missing_share
        "numeric_cols": 7,
        "categorical_cols": 3
    }

    response = client.post("/quality", json=request_data)
    assert response.status_code == 422  # Validation error


def test_quality_from_csv_valid(client):
    """Test /quality-from-csv with valid CSV file."""
    # Create a simple CSV file in memory
    csv_content = """col1,col2,col3,col4,col5
1,2.5,a,10,x
2,3.5,b,20,y
3,4.5,c,30,z
4,5.5,d,40,x
5,6.5,e,50,y
6,7.5,a,60,z
7,8.5,b,70,x
8,9.5,c,80,y
9,10.5,d,90,z
10,11.5,e,100,x
"""

    # Create file-like object
    file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/quality-from-csv",
        files={"file": ("test.csv", file, "text/csv")}
    )

    assert response.status_code == 200

    data = response.json()
    assert "quality_score" in data
    assert "ok_for_model" in data
    assert "message" in data
    assert "latency_ms" in data
    assert "flags" in data
    assert "dataset_shape" in data

    # Check score is in valid range [0, 1]
    assert 0.0 <= data["quality_score"] <= 1.0

    # Check dataset shape
    assert data["dataset_shape"]["n_rows"] == 10
    assert data["dataset_shape"]["n_cols"] == 5


def test_quality_from_csv_with_missing_values(client):
    """Test /quality-from-csv with CSV containing missing values."""
    csv_content = """col1,col2,col3
1,2.5,a
2,,b
3,4.5,
4,5.5,d
5,,e
"""

    file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/quality-from-csv",
        files={"file": ("test_missing.csv", file, "text/csv")}
    )

    assert response.status_code == 200

    data = response.json()
    # Should detect missing values
    assert "has_missing" in data["flags"]


def test_quality_from_csv_empty_file(client):
    """Test /quality-from-csv with empty CSV."""
    csv_content = ""

    file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/quality-from-csv",
        files={"file": ("empty.csv", file, "text/csv")}
    )

    # Should return 400 error for empty file
    assert response.status_code == 400
    # Error message can be about parsing or empty data
    assert "CSV" in response.json()["detail"]


def test_quality_from_csv_invalid_format(client):
    """Test /quality-from-csv with invalid CSV format."""
    csv_content = "This is not a valid CSV file"

    file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/quality-from-csv",
        files={"file": ("invalid.csv", file, "text/csv")}
    )

    # Should return 400 error for invalid format
    # Note: Some invalid CSVs might be parsed as single-column DataFrames
    # which then fail the empty check
    assert response.status_code == 400
    assert "CSV" in response.json()["detail"]


def test_quality_from_csv_with_duplicates(client):
    """Test /quality-from-csv detects duplicate rows."""
    csv_content = """col1,col2,col3
1,2.5,a
2,3.5,b
1,2.5,a
4,5.5,d
"""

    file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/quality-from-csv",
        files={"file": ("duplicates.csv", file, "text/csv")}
    )

    assert response.status_code == 200

    data = response.json()
    # Should detect duplicates
    assert "has_duplicates" in data["flags"]


def test_quality_from_csv_large_dataset(client):
    """Test /quality-from-csv with a larger dataset."""
    # Generate larger CSV
    rows = ["col1,col2,col3"]
    for i in range(100):
        rows.append(f"{i},{i*1.5},cat{i%5}")

    csv_content = "\n".join(rows)
    file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/quality-from-csv",
        files={"file": ("large.csv", file, "text/csv")}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["dataset_shape"]["n_rows"] == 100
    assert data["dataset_shape"]["n_cols"] == 3


def test_openapi_docs_accessible(client):
    """Test that OpenAPI documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_json_accessible(client):
    """Test that OpenAPI JSON schema is accessible."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/quality" in schema["paths"]
    assert "/quality-from-csv" in schema["paths"]
