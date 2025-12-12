# EDA CLI - Exploratory Data Analysis Tool

**Автор:** Новиков Максим Петрович
**Группа:** БСБО-05-23

Command-line tool for quick and comprehensive exploratory data analysis of CSV datasets.

## Features

- **Quick Overview**: Get instant statistics and quality metrics
- **Comprehensive Reports**: Generate detailed Markdown reports with visualizations
- **Data Quality Checks**: Advanced heuristics for detecting data issues
- **Multiple Visualizations**: Histograms, boxplots, correlation matrices, and more
- **Flexible CLI**: Easy-to-use command-line interface with many options

## Installation

Using `uv`:

```bash
cd homeworks/HW03/eda-cli
uv sync
```

## Available Commands

### 1. `overview` - Quick Dataset Overview

Display basic statistics, quality metrics, and potential issues.

**Usage:**
```bash
uv run eda-cli overview data/example.csv
uv run eda-cli overview data/mydata.csv --sep ";" --encoding "latin-1"
```

**Options:**
- `--sep`: CSV separator (default: comma)
- `--encoding`: File encoding (default: utf-8)

### 2. `report` - Generate Comprehensive EDA Report

Create a detailed Markdown report with visualizations.

**Usage:**
```bash
uv run eda-cli report data/example.csv
uv run eda-cli report data/example.csv --out-dir my_reports
uv run eda-cli report data/example.csv --title "Sales Data Analysis"
uv run eda-cli report data/example.csv --max-hist-columns 10 --json-summary
```

**Options:**
- `--out-dir`: Output directory for report and plots (default: "reports")
- `--sep`: CSV separator (default: comma)
- `--encoding`: File encoding (default: utf-8)
- **`--max-hist-columns`**: Maximum number of columns to include in histograms (default: 6) ⭐ NEW
- **`--top-k-categories`**: Number of top categories to show in bar plots (default: 10) ⭐ NEW
- **`--title`**: Title for the report (default: "EDA Report") ⭐ NEW
- **`--min-missing-share`**: Minimum missing rate to flag column as problematic (default: 0.05) ⭐ NEW
- **`--json-summary`**: Also generate JSON summary file ⭐ NEW

**Example with all new parameters:**
```bash
uv run eda-cli report data/example.csv \
  --title "Customer Dataset - Q4 2024" \
  --max-hist-columns 12 \
  --top-k-categories 15 \
  --min-missing-share 0.10 \
  --json-summary \
  --out-dir reports/q4_analysis
```

This command will:
- Set custom report title
- Show distributions for up to 12 numeric columns
- Display top 15 values for categorical columns
- Flag columns with >10% missing values as problematic
- Generate both Markdown report and JSON summary

### 3. `head` - Display First N Rows

Show the beginning of the dataset.

**Usage:**
```bash
uv run eda-cli head data/example.csv
uv run eda-cli head data/example.csv --n 20
```

**Options:**
- `--n`: Number of rows to display (default: 5)

### 4. `sample` - Random Sample

Display a random sample of rows.

**Usage:**
```bash
uv run eda-cli sample data/example.csv
uv run eda-cli sample data/example.csv --n 20 --seed 42
```

**Options:**
- `--n`: Number of rows to sample (default: 10)
- `--seed`: Random seed for reproducibility

## API Usage (HTTP Service)

Starting from HW04, this tool also provides a FastAPI HTTP service for remote data quality assessment.

### Starting the API Server

```bash
# Development mode with auto-reload
uv run uvicorn eda_cli.api:app --reload

# Production mode
uv run uvicorn eda_cli.api:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **Server**: http://localhost:8000
- **Interactive Documentation (Swagger UI)**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### API Endpoints

#### 1. `GET /health` - Health Check

Check if the service is running.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.2.0"
}
```

#### 2. `POST /quality` - Quality Assessment from Aggregated Features

Assess dataset quality based on summary statistics.

**Request Body:**
```json
{
  "n_rows": 1000,
  "n_cols": 10,
  "max_missing_share": 0.05,
  "numeric_cols": 7,
  "categorical_cols": 3
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/quality \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 1000,
    "n_cols": 10,
    "max_missing_share": 0.05,
    "numeric_cols": 7,
    "categorical_cols": 3
  }'
```

**Response:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.85,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 1.23,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 1000,
    "n_cols": 10
  }
}
```

#### 3. `POST /quality-from-csv` - Quality Assessment from CSV File

Upload a CSV file for comprehensive quality analysis using the EDA core.

**Example:**
```bash
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/example.csv"
```

**Response:**
```json
{
  "ok_for_model": true,
  "quality_score": 0.92,
  "message": "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам).",
  "latency_ms": 45.67,
  "flags": {
    "has_missing": false,
    "has_duplicates": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_many_zero_values": false,
    "has_outliers": true
  },
  "dataset_shape": {
    "n_rows": 150,
    "n_cols": 5
  }
}
```

### Testing the API

Run API tests:

```bash
uv run pytest tests/test_api.py -v
```

### Integration with CLI

The API endpoints use the same core EDA logic as the CLI commands:
- `/quality-from-csv` → uses `compute_basic_stats()` and `compute_quality_flags()` from `core.py`
- Quality score is normalized to 0-1 range (CLI returns 0-100)
- All advanced quality checks (constants, high cardinality, zero-heavy, outliers) are available

## Advanced Data Quality Checks

This tool includes several advanced heuristics for detecting data quality issues:

### 1. Constant Columns ⭐ NEW
Detects columns where all values are identical, which provide no information for analysis.

```python
# Example: Column 'status' has value 'active' for all rows
```

### 2. High Cardinality Categoricals ⭐ NEW
Identifies categorical columns with unusually high number of unique values, which may indicate:
- ID columns mistakenly treated as categories
- Free-text fields
- Data quality issues

```python
# Threshold: >50 unique values or >50% of total rows
```

### 3. Zero-Heavy Columns ⭐ NEW
Flags numeric columns with excessive zero values (>30%), which may indicate:
- Sparse features
- Missing value encoding issues
- Data collection problems

### 4. Outlier Detection ⭐ NEW
Uses IQR method (3×IQR threshold) to identify extreme outliers in numeric columns.

## Report Output

The generated report includes:

1. **Dataset Overview**: Dimensions, types, memory usage
2. **Data Quality Assessment**:
   - Overall quality score (0-100)
   - Missing value analysis
   - Duplicate detection
   - All advanced quality checks
3. **Numeric Features**: Distributions, boxplots
4. **Categorical Features**: Top values, frequency charts
5. **Correlations**: Correlation matrix heatmap
6. **Column Details**: Complete column inventory

## Testing

Run the test suite:

```bash
uv run pytest
uv run pytest -v  # Verbose output
uv run pytest -v --cov=eda_cli  # With coverage report
```

## Example Workflow

### CLI Workflow

```bash
# 1. Quick check of new dataset
uv run eda-cli overview data/sales.csv

# 2. Generate full report with custom settings
uv run eda-cli report data/sales.csv \
  --out-dir reports/sales_analysis \
  --title "Sales Data - 2024 Q4" \
  --max-hist-columns 15 \
  --json-summary

# 3. Check a sample of the data
uv run eda-cli sample data/sales.csv --n 50 --seed 42
```

### API Workflow (HW04)

```bash
# 1. Start the API server
uv run uvicorn eda_cli.api:app --reload

# 2. Check health
curl http://localhost:8000/health

# 3. Assess quality from aggregated statistics
curl -X POST http://localhost:8000/quality \
  -H "Content-Type: application/json" \
  -d '{
    "n_rows": 5000,
    "n_cols": 15,
    "max_missing_share": 0.02,
    "numeric_cols": 10,
    "categorical_cols": 5
  }'

# 4. Upload and analyze CSV file
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/sales.csv"

# 5. Run API tests
uv run pytest tests/test_api.py -v
```

## Project Structure

```
eda-cli/
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── src/
│   └── eda_cli/
│       ├── __init__.py     # Package initialization
│       ├── core.py         # Core EDA logic and quality checks
│       ├── viz.py          # Visualization functions
│       ├── cli.py          # CLI interface
│       └── api.py          # FastAPI HTTP service ⭐ NEW (HW04)
├── tests/
│   ├── test_core.py        # Core logic tests
│   └── test_api.py         # API endpoint tests ⭐ NEW (HW04)
├── data/
│   └── example.csv         # Example dataset
└── reports/                # Generated reports (git-ignored)
```

## Requirements

- Python >=3.10
- pandas >=2.0.0
- numpy >=1.24.0
- matplotlib >=3.7.0
- click >=8.1.0
- fastapi >=0.104.0 ⭐ NEW (HW04)
- uvicorn[standard] >=0.24.0 ⭐ NEW (HW04)
- python-multipart >=0.0.6 ⭐ NEW (HW04)

## Development

Install dev dependencies:

```bash
uv sync --all-extras
```

Run tests with coverage:

```bash
uv run pytest --cov=eda_cli --cov-report=html
```

## License

MIT License

## Author

Новиков Максим Петрович
БСБО-05-23
