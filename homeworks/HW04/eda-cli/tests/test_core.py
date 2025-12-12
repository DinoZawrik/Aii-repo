"""
Tests for core EDA functionality.
"""

import pytest
import pandas as pd
import numpy as np
from eda_cli import core


def test_load_dataset_basic(tmp_path):
    """Test basic dataset loading."""
    # Create a simple CSV
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.to_csv(csv_path, index=False)

    # Load it
    loaded_df = core.load_dataset(str(csv_path))
    assert len(loaded_df) == 3
    assert list(loaded_df.columns) == ['a', 'b']


def test_compute_basic_stats():
    """Test basic statistics computation."""
    df = pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [10, 20, 30, 40, 50],
        'cat1': ['a', 'b', 'c', 'a', 'b'],
        'cat2': ['x', 'y', 'x', 'y', 'x']
    })

    stats = core.compute_basic_stats(df)

    assert stats['n_rows'] == 5
    assert stats['n_cols'] == 4
    assert stats['n_numeric'] == 2
    assert stats['n_categorical'] == 2
    assert set(stats['numeric_columns']) == {'num1', 'num2'}
    assert set(stats['categorical_columns']) == {'cat1', 'cat2'}


def test_quality_flags_no_issues():
    """Test quality flags on clean data."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50],
        'c': ['x', 'y', 'z', 'x', 'y']
    })

    flags = core.compute_quality_flags(df)

    assert flags['has_missing'] is False
    assert flags['has_duplicates'] is False
    assert flags['quality_score'] >= 90  # Should be high


def test_quality_flags_with_missing():
    """Test quality flags detect missing values."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4, 5],
        'b': [10, np.nan, 30, np.nan, 50]
    })

    flags = core.compute_quality_flags(df)

    assert flags['has_missing'] is True
    assert flags['missing_rate'] > 0
    assert flags['missing_by_column']['a'] > 0
    assert flags['missing_by_column']['b'] > 0


def test_quality_flags_with_duplicates():
    """Test quality flags detect duplicate rows."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 1, 2],  # Rows 0 and 3 are duplicates, 1 and 4 too
        'b': [10, 20, 30, 10, 20]
    })

    flags = core.compute_quality_flags(df)

    assert flags['has_duplicates'] is True
    assert flags['n_duplicates'] > 0


# NEW TEST 1: Constant columns
def test_quality_flags_constant_columns():
    """Test detection of constant columns."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 10, 10, 10, 10],  # Constant column
        'c': ['x', 'y', 'z', 'x', 'y'],
        'd': ['same', 'same', 'same', 'same', 'same']  # Another constant
    })

    flags = core.compute_quality_flags(df)

    assert flags['has_constant_columns'] is True
    assert flags['n_constant_columns'] == 2
    assert 'b' in flags['constant_columns']
    assert 'd' in flags['constant_columns']
    assert 'a' not in flags['constant_columns']
    assert 'c' not in flags['constant_columns']


# NEW TEST 2: High cardinality categoricals
def test_quality_flags_high_cardinality():
    """Test detection of high cardinality categorical columns."""
    # Create a dataset with high cardinality column
    n_rows = 100
    df = pd.DataFrame({
        'normal_cat': ['a', 'b', 'c'] * (n_rows // 3) + ['a'],
        'high_card': [f'val_{i}' for i in range(n_rows)],  # 100 unique values
        'numeric': range(n_rows)
    })

    flags = core.compute_quality_flags(df)

    assert flags['has_high_cardinality_categoricals'] is True
    assert len(flags['high_cardinality_categoricals']) > 0

    # Check that high_card is flagged
    high_card_cols = [item['column'] for item in flags['high_cardinality_categoricals']]
    assert 'high_card' in high_card_cols
    assert 'normal_cat' not in high_card_cols


# NEW TEST 3: Many zero values
def test_quality_flags_zero_heavy_columns():
    """Test detection of columns with many zero values."""
    df = pd.DataFrame({
        'normal': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'zero_heavy': [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],  # 50% zeros
        'very_zero': [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],  # 80% zeros
        'no_zeros': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })

    flags = core.compute_quality_flags(df)

    assert flags['has_many_zero_values'] is True
    assert len(flags['zero_heavy_columns']) >= 2

    # Check specific columns
    zero_cols = {item['column']: item['zero_rate'] for item in flags['zero_heavy_columns']}

    assert 'zero_heavy' in zero_cols
    assert 'very_zero' in zero_cols
    assert 'normal' not in zero_cols
    assert 'no_zeros' not in zero_cols

    # Check rates are correct
    assert zero_cols['zero_heavy'] >= 0.3
    assert zero_cols['very_zero'] >= 0.3


# NEW TEST 4: Outliers detection
def test_quality_flags_outliers():
    """Test detection of outliers."""
    # Create data with outliers
    normal_data = list(range(1, 101))  # Normal range: 1-100
    outlier_data = normal_data + [1000, -500]  # Add extreme outliers

    df = pd.DataFrame({
        'normal': normal_data + [101, 102],  # No outliers
        'with_outliers': outlier_data
    })

    flags = core.compute_quality_flags(df)

    # Should detect outliers
    assert flags['has_outliers'] is True
    assert len(flags['outlier_columns']) > 0

    # Check that with_outliers is flagged
    outlier_cols = [item['column'] for item in flags['outlier_columns']]
    assert 'with_outliers' in outlier_cols


def test_compute_column_stats_numeric():
    """Test column statistics for numeric columns."""
    df = pd.DataFrame({
        'values': [1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10]
    })

    stats = core.compute_column_stats(df, 'values')

    assert stats['name'] == 'values'
    assert stats['n_total'] == 10
    assert stats['n_missing'] == 1
    assert stats['n_unique'] == 9  # 9 unique non-null values
    assert 'mean' in stats
    assert 'median' in stats
    assert stats['mean'] is not None


def test_compute_column_stats_categorical():
    """Test column statistics for categorical columns."""
    df = pd.DataFrame({
        'category': ['a', 'b', 'c', 'a', 'b', 'a', np.nan, 'c', 'a', 'b']
    })

    stats = core.compute_column_stats(df, 'category')

    assert stats['name'] == 'category'
    assert stats['n_missing'] == 1
    assert stats['n_unique'] == 3
    assert 'top_values' in stats
    assert 'mode' in stats


def test_get_correlations():
    """Test correlation matrix computation."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],  # Perfect correlation with a
        'c': [5, 4, 3, 2, 1],   # Perfect negative correlation with a
        'cat': ['x', 'y', 'z', 'x', 'y']
    })

    corr = core.get_correlations(df)

    assert not corr.empty
    assert 'a' in corr.columns
    assert 'b' in corr.columns
    assert 'c' in corr.columns
    assert 'cat' not in corr.columns  # Should exclude categorical

    # Check perfect correlation
    assert abs(corr.loc['a', 'b'] - 1.0) < 0.01
    assert abs(corr.loc['a', 'c'] + 1.0) < 0.01  # Negative correlation


def test_identify_potential_targets():
    """Test identification of potential target columns."""
    df = pd.DataFrame({
        'id': range(10),
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # Binary target
        'feature1': range(10),
        'class_label': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'outcome': [True, False, True, False, True, False, True, False, True, False]
    })

    targets = core.identify_potential_targets(df)

    assert 'target' in targets  # Has 'target' in name
    assert 'class_label' in targets  # Has 'class' in name and is binary
    assert 'outcome' in targets  # Has 'outcome' in name and is binary
    assert 'id' not in targets  # Not binary
    assert 'feature1' not in targets  # Not binary


def test_generate_summary_text():
    """Test summary text generation."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [10, 20, 30, 40, 50],
        'c': ['x', 'y', 'z', 'x', 'y']
    })

    stats = core.compute_basic_stats(df)
    flags = core.compute_quality_flags(df)
    summary = core.generate_summary_text(df, stats, flags)

    assert 'DATASET OVERVIEW' in summary
    assert 'DATA QUALITY' in summary
    assert 'Rows: 5' in summary
    assert 'Columns: 3' in summary
    assert 'Quality Score' in summary


def test_quality_score_penalties():
    """Test that quality score is penalized for issues."""
    # Clean data - should have high score
    clean_df = pd.DataFrame({
        'a': range(100),
        'b': range(100, 200),
        'c': ['x', 'y'] * 50
    })
    clean_flags = core.compute_quality_flags(clean_df)
    clean_score = clean_flags['quality_score']

    # Data with issues - should have lower score
    problematic_df = pd.DataFrame({
        'a': [np.nan] * 50 + list(range(50)),  # 50% missing
        'b': [1] * 100,  # Constant column
        'c': [f'val_{i}' for i in range(100)],  # High cardinality
        'd': [0] * 80 + list(range(20))  # 80% zeros
    })
    prob_flags = core.compute_quality_flags(problematic_df)
    prob_score = prob_flags['quality_score']

    assert prob_score < clean_score
    assert clean_score >= 90
    assert prob_score < 70


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
