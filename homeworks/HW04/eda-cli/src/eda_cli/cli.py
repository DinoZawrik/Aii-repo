"""
Command-line interface for EDA CLI tool.
"""

import click
import sys
from pathlib import Path
from typing import Optional
import json

from . import core, viz


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """
    EDA CLI - Exploratory Data Analysis Command Line Tool

    A tool for quick exploratory data analysis of CSV datasets.
    """
    pass


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--sep', default=',', help='CSV separator (default: comma)')
@click.option('--encoding', default='utf-8', help='File encoding (default: utf-8)')
def overview(filepath: str, sep: str, encoding: str):
    """
    Display a quick overview of the dataset.

    Shows basic statistics, data quality metrics, and potential issues.

    Example:
        eda-cli overview data/example.csv
        eda-cli overview data/data.csv --sep ";" --encoding "latin-1"
    """
    try:
        click.echo(f"Loading dataset from: {filepath}")
        df = core.load_dataset(filepath, sep=sep, encoding=encoding)

        click.echo(f"✓ Loaded successfully: {len(df)} rows, {len(df.columns)} columns\n")

        # Compute statistics
        stats = core.compute_basic_stats(df)
        quality_flags = core.compute_quality_flags(df)

        # Generate and print summary
        summary = core.generate_summary_text(df, stats, quality_flags)
        click.echo(summary)

        # Column details
        click.echo("\nCOLUMN DETAILS:")
        click.echo("=" * 60)
        for col in df.columns[:10]:  # Show first 10 columns
            col_stats = core.compute_column_stats(df, col)
            click.echo(f"\n{col_stats['name']} ({col_stats['dtype']})")
            click.echo(f"  Missing: {col_stats['n_missing']} ({col_stats['missing_rate']*100:.1f}%)")
            click.echo(f"  Unique: {col_stats['n_unique']}")

            if 'mean' in col_stats and col_stats['mean'] is not None:
                click.echo(f"  Mean: {col_stats['mean']:.2f}, Median: {col_stats['median']:.2f}")

        if len(df.columns) > 10:
            click.echo(f"\n... and {len(df.columns) - 10} more columns")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--out-dir', default='reports', help='Output directory for report and plots')
@click.option('--sep', default=',', help='CSV separator (default: comma)')
@click.option('--encoding', default='utf-8', help='File encoding (default: utf-8)')
@click.option('--max-hist-columns', default=6, type=int,
              help='Maximum number of columns to include in histograms (default: 6)')
@click.option('--top-k-categories', default=10, type=int,
              help='Number of top categories to show in bar plots (default: 10)')
@click.option('--title', default='EDA Report',
              help='Title for the report (default: "EDA Report")')
@click.option('--min-missing-share', default=0.05, type=float,
              help='Minimum missing rate to flag column as problematic (default: 0.05)')
@click.option('--json-summary', is_flag=True,
              help='Also generate JSON summary file')
def report(filepath: str, out_dir: str, sep: str, encoding: str,
          max_hist_columns: int, top_k_categories: int, title: str,
          min_missing_share: float, json_summary: bool):
    """
    Generate a comprehensive EDA report with visualizations.

    Creates a Markdown report with statistics, quality checks, and plots.

    Examples:
        eda-cli report data/example.csv
        eda-cli report data/example.csv --out-dir my_reports
        eda-cli report data/example.csv --title "My Dataset Analysis" --json-summary
        eda-cli report data/example.csv --max-hist-columns 10 --top-k-categories 15
    """
    try:
        click.echo(f"📊 Generating EDA report for: {filepath}")
        click.echo(f"   Output directory: {out_dir}")

        # Load dataset
        df = core.load_dataset(filepath, sep=sep, encoding=encoding)
        click.echo(f"✓ Loaded: {len(df)} rows, {len(df.columns)} columns")

        # Create output directory
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Compute statistics and quality flags
        click.echo("   Computing statistics...")
        stats = core.compute_basic_stats(df)
        quality_flags = core.compute_quality_flags(df)

        # Generate visualizations
        click.echo("   Creating visualizations...")
        plots = viz.create_summary_plots(
            df,
            str(output_path),
            max_hist_columns=max_hist_columns,
            top_k_categories=top_k_categories
        )

        # Generate report
        click.echo("   Writing report...")
        report_path = output_path / 'report.md'
        _write_markdown_report(
            df, stats, quality_flags, plots, report_path,
            title=title,
            min_missing_share=min_missing_share,
            max_hist_columns=max_hist_columns,
            top_k_categories=top_k_categories
        )

        click.echo(f"✓ Report saved to: {report_path}")

        # Optional JSON summary
        if json_summary:
            click.echo("   Generating JSON summary...")
            json_path = output_path / 'summary.json'
            _write_json_summary(df, stats, quality_flags, json_path)
            click.echo(f"✓ JSON summary saved to: {json_path}")

        # Print plot summary
        total_plots = sum(len(paths) for paths in plots.values())
        click.echo(f"✓ Created {total_plots} visualization(s)")

        click.echo(f"\n🎉 Report generation complete!")
        click.echo(f"   Quality Score: {quality_flags['quality_score']:.1f}/100")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--n', default=5, type=int, help='Number of rows to display (default: 5)')
@click.option('--sep', default=',', help='CSV separator (default: comma)')
@click.option('--encoding', default='utf-8', help='File encoding (default: utf-8)')
def head(filepath: str, n: int, sep: str, encoding: str):
    """
    Display the first N rows of the dataset.

    Example:
        eda-cli head data/example.csv
        eda-cli head data/example.csv --n 10
    """
    try:
        df = core.load_dataset(filepath, sep=sep, encoding=encoding)
        click.echo(f"First {n} rows of {filepath}:\n")
        click.echo(df.head(n).to_string())
        click.echo(f"\nTotal: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--n', default=10, type=int, help='Number of rows to sample (default: 10)')
@click.option('--seed', default=None, type=int, help='Random seed for reproducibility')
@click.option('--sep', default=',', help='CSV separator (default: comma)')
@click.option('--encoding', default='utf-8', help='File encoding (default: utf-8)')
def sample(filepath: str, n: int, seed: Optional[int], sep: str, encoding: str):
    """
    Display a random sample of N rows from the dataset.

    Example:
        eda-cli sample data/example.csv
        eda-cli sample data/example.csv --n 20 --seed 42
    """
    try:
        df = core.load_dataset(filepath, sep=sep, encoding=encoding)

        if seed is not None:
            sample_df = df.sample(n=min(n, len(df)), random_state=seed)
            click.echo(f"Random sample ({n} rows, seed={seed}) from {filepath}:\n")
        else:
            sample_df = df.sample(n=min(n, len(df)))
            click.echo(f"Random sample ({n} rows) from {filepath}:\n")

        click.echo(sample_df.to_string())
        click.echo(f"\nTotal: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _write_markdown_report(df, stats, quality_flags, plots, output_path,
                          title='EDA Report', min_missing_share=0.05,
                          max_hist_columns=6, top_k_categories=10):
    """Write comprehensive Markdown report."""
    lines = []

    lines.append(f"# {title}\n")
    lines.append(f"**Generated by EDA CLI v0.1.0**\n")
    lines.append("---\n")

    # Dataset Overview
    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- **Rows:** {stats['n_rows']:,}")
    lines.append(f"- **Columns:** {stats['n_cols']}")
    lines.append(f"- **Memory Usage:** {stats['memory_usage_mb']:.2f} MB")
    lines.append(f"- **Numeric Columns:** {stats['n_numeric']}")
    lines.append(f"- **Categorical Columns:** {stats['n_categorical']}\n")

    # Data Quality
    lines.append("## 2. Data Quality\n")
    lines.append(f"**Overall Quality Score:** {quality_flags['quality_score']:.1f}/100\n")

    lines.append(f"**Report Configuration:**")
    lines.append(f"- Minimum missing share threshold: {min_missing_share*100:.1f}%")
    lines.append(f"- Max histogram columns: {max_hist_columns}")
    lines.append(f"- Top K categories shown: {top_k_categories}\n")

    # Missing Values
    lines.append("### 2.1 Missing Values\n")
    if quality_flags['has_missing']:
        lines.append(f"- Overall missing rate: {quality_flags['missing_rate']*100:.2f}%")
        lines.append(f"- Columns with missing values above {min_missing_share*100:.0f}% threshold:\n")

        problematic_cols = [(col, rate) for col, rate in quality_flags['missing_by_column'].items()
                           if rate > min_missing_share]
        if problematic_cols:
            for col, rate in sorted(problematic_cols, key=lambda x: x[1], reverse=True):
                lines.append(f"  - `{col}`: {rate*100:.2f}%")
        else:
            lines.append(f"  - No columns above threshold")
    else:
        lines.append("✓ No missing values detected\n")

    if 'missing' in plots and plots['missing']:
        lines.append(f"\n![Missing Values]({Path(plots['missing'][0]).name})\n")

    # Duplicates
    lines.append("### 2.2 Duplicate Rows\n")
    if quality_flags['has_duplicates']:
        lines.append(f"⚠ Found {quality_flags['n_duplicates']} duplicate rows ({quality_flags['duplicate_rate']*100:.2f}%)\n")
    else:
        lines.append("✓ No duplicate rows detected\n")

    # New Heuristics
    lines.append("### 2.3 Advanced Quality Checks\n")

    if quality_flags['has_constant_columns']:
        lines.append(f"**Constant Columns:** {quality_flags['n_constant_columns']} found")
        for col in quality_flags['constant_columns']:
            lines.append(f"  - `{col}`")
        lines.append("")

    if quality_flags['has_high_cardinality_categoricals']:
        lines.append(f"**High Cardinality Categoricals:**")
        for item in quality_flags['high_cardinality_categoricals']:
            lines.append(f"  - `{item['column']}`: {item['unique_count']} unique ({item['unique_rate']*100:.1f}%)")
        lines.append("")

    if quality_flags['has_many_zero_values']:
        lines.append(f"**Zero-Heavy Columns:**")
        for item in quality_flags['zero_heavy_columns']:
            lines.append(f"  - `{item['column']}`: {item['zero_rate']*100:.1f}% zeros")
        lines.append("")

    if quality_flags['has_outliers']:
        lines.append(f"**Columns with Outliers:**")
        for item in quality_flags['outlier_columns'][:5]:
            lines.append(f"  - `{item['column']}`: {item['n_outliers']} outliers ({item['outlier_rate']*100:.1f}%)")
        lines.append("")

    # Numeric Distributions
    lines.append("## 3. Numeric Features\n")
    if stats['n_numeric'] > 0:
        lines.append(f"Showing distributions for up to {max_hist_columns} numeric columns:\n")
        if 'histograms' in plots and plots['histograms']:
            for plot_path in plots['histograms']:
                lines.append(f"![Distributions]({Path(plot_path).name})\n")

        if 'boxplots' in plots and plots['boxplots']:
            lines.append(f"![Boxplots]({Path(plots['boxplots'][0]).name})\n")
    else:
        lines.append("No numeric columns found.\n")

    # Categorical Features
    lines.append("## 4. Categorical Features\n")
    if stats['n_categorical'] > 0:
        lines.append(f"Showing top {top_k_categories} values for each categorical column:\n")
        if 'categoricals' in plots and plots['categoricals']:
            for plot_path in plots['categoricals']:
                lines.append(f"![Categorical]({Path(plot_path).name})\n")
    else:
        lines.append("No categorical columns found.\n")

    # Correlations
    lines.append("## 5. Correlations\n")
    if stats['n_numeric'] >= 2:
        if 'correlation' in plots and plots['correlation']:
            lines.append(f"![Correlation Matrix]({Path(plots['correlation'][0]).name})\n")
    else:
        lines.append("Not enough numeric columns for correlation analysis.\n")

    # Column List
    lines.append("## 6. Column List\n")
    lines.append("| Column | Type | Missing | Unique |")
    lines.append("|--------|------|---------|--------|")
    for col in df.columns:
        col_stats = core.compute_column_stats(df, col)
        lines.append(f"| {col} | {col_stats['dtype']} | {col_stats['missing_rate']*100:.1f}% | {col_stats['n_unique']} |")
    lines.append("")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_json_summary(df, stats, quality_flags, output_path):
    """Write JSON summary."""
    summary = {
        'n_rows': stats['n_rows'],
        'n_cols': stats['n_cols'],
        'n_numeric': stats['n_numeric'],
        'n_categorical': stats['n_categorical'],
        'quality_score': quality_flags['quality_score'],
        'has_missing': quality_flags['has_missing'],
        'missing_rate': quality_flags['missing_rate'],
        'has_duplicates': quality_flags['has_duplicates'],
        'n_duplicates': quality_flags['n_duplicates'],
        'constant_columns': quality_flags['constant_columns'],
        'problematic_columns': []
    }

    # Add problematic columns
    for col, rate in quality_flags['missing_by_column'].items():
        if rate > 0.05:
            summary['problematic_columns'].append({
                'name': col,
                'issue': 'high_missing_rate',
                'value': rate
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    cli()
