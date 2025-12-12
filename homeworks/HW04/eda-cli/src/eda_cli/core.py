"""
Core EDA functionality: data quality checks, statistics, and analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def load_dataset(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath, **kwargs)


def compute_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistics about the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with basic statistics
    """
    stats = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 ** 2),
    }

    # Numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    stats['n_numeric'] = len(numeric_cols)
    stats['n_categorical'] = len(categorical_cols)
    stats['numeric_columns'] = numeric_cols
    stats['categorical_columns'] = categorical_cols

    return stats


def compute_quality_flags(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute data quality flags and heuristics.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with quality flags and metrics
    """
    flags = {}

    # Basic quality checks
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    flags['has_missing'] = missing_cells > 0
    flags['missing_rate'] = missing_cells / total_cells if total_cells > 0 else 0
    flags['missing_by_column'] = df.isna().mean().to_dict()

    # Duplicate rows
    n_duplicates = df.duplicated().sum()
    flags['has_duplicates'] = n_duplicates > 0
    flags['n_duplicates'] = int(n_duplicates)
    flags['duplicate_rate'] = n_duplicates / len(df) if len(df) > 0 else 0

    # NEW HEURISTIC 1: Constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            constant_cols.append(col)

    flags['has_constant_columns'] = len(constant_cols) > 0
    flags['constant_columns'] = constant_cols
    flags['n_constant_columns'] = len(constant_cols)

    # NEW HEURISTIC 2: High cardinality categoricals
    high_cardinality_cols = []
    cardinality_threshold = max(50, len(df) * 0.5)  # 50 or 50% of rows

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count > cardinality_threshold:
            high_cardinality_cols.append({
                'column': col,
                'unique_count': int(unique_count),
                'unique_rate': unique_count / len(df)
            })

    flags['has_high_cardinality_categoricals'] = len(high_cardinality_cols) > 0
    flags['high_cardinality_categoricals'] = high_cardinality_cols

    # NEW HEURISTIC 3: Many zero values in numeric columns
    zero_heavy_cols = []
    zero_threshold = 0.3  # 30% zeros

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        zero_rate = (df[col] == 0).sum() / len(df)
        if zero_rate > zero_threshold:
            zero_heavy_cols.append({
                'column': col,
                'zero_rate': float(zero_rate),
                'zero_count': int((df[col] == 0).sum())
            })

    flags['has_many_zero_values'] = len(zero_heavy_cols) > 0
    flags['zero_heavy_columns'] = zero_heavy_cols

    # NEW HEURISTIC 4: Suspicious outliers (extreme values)
    outlier_cols = []

    for col in numeric_cols:
        if df[col].notna().sum() < 2:  # Need at least 2 non-null values
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:  # All values the same
            continue

        # Extreme outliers: beyond 3*IQR
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            outlier_cols.append({
                'column': col,
                'n_outliers': len(outliers),
                'outlier_rate': len(outliers) / len(df),
                'bounds': (float(lower_bound), float(upper_bound))
            })

    flags['has_outliers'] = len(outlier_cols) > 0
    flags['outlier_columns'] = outlier_cols

    # Compute overall quality score (0-100)
    score = 100.0

    # Penalties
    if flags['missing_rate'] > 0.05:
        score -= min(20, flags['missing_rate'] * 100)

    if flags['duplicate_rate'] > 0.01:
        score -= min(10, flags['duplicate_rate'] * 100)

    if flags['has_constant_columns']:
        score -= min(10, len(constant_cols) * 2)

    if flags['has_high_cardinality_categoricals']:
        score -= min(10, len(high_cardinality_cols) * 3)

    if flags['has_many_zero_values']:
        score -= min(10, len(zero_heavy_cols) * 2)

    flags['quality_score'] = max(0, score)

    return flags


def compute_column_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Compute detailed statistics for a single column.

    Args:
        df: Input DataFrame
        column: Column name

    Returns:
        Dictionary with column statistics
    """
    stats = {
        'name': column,
        'dtype': str(df[column].dtype),
        'n_total': len(df),
        'n_missing': int(df[column].isna().sum()),
        'missing_rate': float(df[column].isna().mean()),
        'n_unique': int(df[column].nunique()),
    }

    if pd.api.types.is_numeric_dtype(df[column]):
        # Numeric statistics
        stats['min'] = float(df[column].min()) if df[column].notna().any() else None
        stats['max'] = float(df[column].max()) if df[column].notna().any() else None
        stats['mean'] = float(df[column].mean()) if df[column].notna().any() else None
        stats['median'] = float(df[column].median()) if df[column].notna().any() else None
        stats['std'] = float(df[column].std()) if df[column].notna().any() else None
        stats['q25'] = float(df[column].quantile(0.25)) if df[column].notna().any() else None
        stats['q75'] = float(df[column].quantile(0.75)) if df[column].notna().any() else None
    else:
        # Categorical statistics
        value_counts = df[column].value_counts()
        stats['top_values'] = value_counts.head(10).to_dict()
        stats['mode'] = str(df[column].mode()[0]) if len(df[column].mode()) > 0 else None

    return stats


def get_correlations(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.

    Args:
        df: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    return numeric_df.corr(method=method)


def identify_potential_targets(df: pd.DataFrame) -> List[str]:
    """
    Identify columns that might be good prediction targets.

    Args:
        df: Input DataFrame

    Returns:
        List of potential target column names
    """
    targets = []

    for col in df.columns:
        col_lower = col.lower()

        # Check for common target keywords
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 'y']
        if any(keyword in col_lower for keyword in target_keywords):
            targets.append(col)
            continue

        # Binary columns (good for classification)
        if df[col].nunique() == 2:
            targets.append(col)

    return targets


def generate_summary_text(df: pd.DataFrame, stats: Dict[str, Any],
                         quality_flags: Dict[str, Any]) -> str:
    """
    Generate a text summary of the dataset.

    Args:
        df: Input DataFrame
        stats: Basic statistics dictionary
        quality_flags: Quality flags dictionary

    Returns:
        Formatted summary text
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET OVERVIEW")
    lines.append("=" * 60)
    lines.append(f"Rows: {stats['n_rows']:,}")
    lines.append(f"Columns: {stats['n_cols']}")
    lines.append(f"Memory: {stats['memory_usage_mb']:.2f} MB")
    lines.append("")

    lines.append("Column Types:")
    lines.append(f"  - Numeric: {stats['n_numeric']}")
    lines.append(f"  - Categorical: {stats['n_categorical']}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("DATA QUALITY")
    lines.append("=" * 60)
    lines.append(f"Overall Quality Score: {quality_flags['quality_score']:.1f}/100")
    lines.append("")

    if quality_flags['has_missing']:
        lines.append(f"⚠ Missing Values: {quality_flags['missing_rate']*100:.2f}% of cells")
    else:
        lines.append("✓ No missing values")

    if quality_flags['has_duplicates']:
        lines.append(f"⚠ Duplicate Rows: {quality_flags['n_duplicates']} ({quality_flags['duplicate_rate']*100:.2f}%)")
    else:
        lines.append("✓ No duplicate rows")

    if quality_flags['has_constant_columns']:
        lines.append(f"⚠ Constant Columns: {quality_flags['n_constant_columns']}")
        for col in quality_flags['constant_columns']:
            lines.append(f"    - {col}")
    else:
        lines.append("✓ No constant columns")

    if quality_flags['has_high_cardinality_categoricals']:
        lines.append(f"⚠ High Cardinality Categoricals: {len(quality_flags['high_cardinality_categoricals'])}")
        for item in quality_flags['high_cardinality_categoricals'][:3]:
            lines.append(f"    - {item['column']}: {item['unique_count']} unique values")

    if quality_flags['has_many_zero_values']:
        lines.append(f"⚠ Zero-Heavy Columns: {len(quality_flags['zero_heavy_columns'])}")
        for item in quality_flags['zero_heavy_columns'][:3]:
            lines.append(f"    - {item['column']}: {item['zero_rate']*100:.1f}% zeros")

    if quality_flags['has_outliers']:
        lines.append(f"⚠ Columns with Outliers: {len(quality_flags['outlier_columns'])}")
        for item in quality_flags['outlier_columns'][:3]:
            lines.append(f"    - {item['column']}: {item['n_outliers']} outliers ({item['outlier_rate']*100:.1f}%)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
