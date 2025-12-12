"""
Visualization functions for EDA reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import List, Optional

# Use non-interactive backend
matplotlib.use('Agg')


def plot_missing_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """
    Create a heatmap showing missing values pattern.

    Args:
        df: Input DataFrame
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create binary matrix: 1 for missing, 0 for present
    missing_matrix = df.isna().astype(int)

    if missing_matrix.sum().sum() == 0:
        # No missing values
        ax.text(0.5, 0.5, 'No Missing Values',
                ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    else:
        im = ax.imshow(missing_matrix.T, cmap='Reds', aspect='auto',
                      interpolation='nearest')

        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns, fontsize=8)
        ax.set_xlabel('Row Index')
        ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Missing (1) / Present (0)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame, output_dir: str,
                               max_cols: int = 6) -> List[str]:
    """
    Create histogram plots for numeric columns.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        max_cols: Maximum number of columns to plot

    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = numeric_cols[:max_cols]

    created_files = []

    if not numeric_cols:
        return created_files

    # Create subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        ax = axes[row, col_idx]

        data = df[col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # Add mean line
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_val:.2f}')
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(numeric_cols), n_rows * n_cols):
        row = idx // n_cols
        col_idx = idx % n_cols
        axes[row, col_idx].axis('off')

    plt.tight_layout()
    output_path = output_dir / 'numeric_distributions.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    created_files.append(str(output_path))
    return created_files


def plot_categorical_bars(df: pd.DataFrame, output_dir: str,
                         top_k: int = 10, max_cols: int = 4) -> List[str]:
    """
    Create bar plots for categorical columns.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        top_k: Number of top categories to show
        max_cols: Maximum number of columns to plot

    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = categorical_cols[:max_cols]

    created_files = []

    if not categorical_cols:
        return created_files

    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 6))

        value_counts = df[col].value_counts().head(top_k)

        ax.barh(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.8)
        ax.set_yticks(range(len(value_counts)))
        ax.set_yticklabels(value_counts.index)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_title(f'Top {min(top_k, len(value_counts))} Values in {col}',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(value_counts.values):
            ax.text(v, i, f' {v}', va='center', fontsize=10)

        plt.tight_layout()
        safe_col_name = col.replace('/', '_').replace(' ', '_')
        output_path = output_dir / f'categorical_{safe_col_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        created_files.append(str(output_path))

    return created_files


def plot_correlation_matrix(df: pd.DataFrame, output_path: str,
                            method: str = 'pearson') -> None:
    """
    Create correlation matrix heatmap for numeric columns.

    Args:
        df: Input DataFrame
        output_path: Path to save the plot
        method: Correlation method
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty or len(numeric_df.columns) < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Not enough numeric columns for correlation',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    corr_matrix = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)

    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title(f'Correlation Matrix ({method.capitalize()})',
                fontsize=14, fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax, label='Correlation Coefficient')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplots(df: pd.DataFrame, output_path: str, max_cols: int = 6) -> None:
    """
    Create boxplots for numeric columns.

    Args:
        df: Input DataFrame
        output_path: Path to save the plot
        max_cols: Maximum number of columns to plot
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = numeric_cols[:max_cols]

    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No numeric columns available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    data_to_plot = []
    labels = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            # Standardize for better visualization
            data_standardized = (data - data.mean()) / (data.std() + 1e-8)
            data_to_plot.append(data_standardized)
            labels.append(col)

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)

        # Color boxes
        colors = plt.cm.Set3(range(len(data_to_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Standardized Values', fontsize=12)
        ax.set_title('Boxplots of Numeric Features (Standardized)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plots(df: pd.DataFrame, output_dir: str,
                        max_hist_columns: int = 6,
                        top_k_categories: int = 10) -> Dict[str, List[str]]:
    """
    Create all summary plots for the dataset.

    Args:
        df: Input DataFrame
        output_dir: Directory to save plots
        max_hist_columns: Max number of histograms to create
        top_k_categories: Number of top categories to show in bar plots

    Returns:
        Dictionary mapping plot type to list of file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_plots = {}

    # Missing values heatmap
    missing_path = output_dir / 'missing_heatmap.png'
    plot_missing_heatmap(df, str(missing_path))
    created_plots['missing'] = [str(missing_path)]

    # Numeric distributions
    hist_paths = plot_numeric_distributions(df, str(output_dir), max_cols=max_hist_columns)
    created_plots['histograms'] = hist_paths

    # Categorical bars
    bar_paths = plot_categorical_bars(df, str(output_dir), top_k=top_k_categories)
    created_plots['categoricals'] = bar_paths

    # Correlation matrix
    corr_path = output_dir / 'correlation_matrix.png'
    plot_correlation_matrix(df, str(corr_path))
    created_plots['correlation'] = [str(corr_path)]

    # Boxplots
    box_path = output_dir / 'boxplots.png'
    plot_boxplots(df, str(box_path))
    created_plots['boxplots'] = [str(box_path)]

    return created_plots
