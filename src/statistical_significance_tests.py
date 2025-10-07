#!/usr/bin/env python3
"""
Statistical Significance Testing for GNN Publication

This script performs comprehensive statistical analysis comparing the best-performing
model (RGGC with Explainer) against all baseline methods including:
- Paired t-tests for cross-validation fold comparisons
- Effect size calculations (Cohen's d)
- Wilcoxon signed-rank tests (non-parametric alternative)
- Publication-ready results tables

Usage:
    python statistical_significance_tests.py --results_dir enhanced_results_case1 --output_dir publication_stats

Examples:
    # Run full statistical analysis for Case 1
    python statistical_significance_tests.py --results_dirs enhanced_results_case1

    # Compare across all cases
    python statistical_significance_tests.py --results_dirs enhanced_results_case1 enhanced_results_case2 enhanced_results_case3

    # Specify different best model
    python statistical_significance_tests.py --results_dirs enhanced_results_case1 --best_model rggc
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')


class StatisticalSignificanceTester:
    """
    Performs statistical significance testing comparing GNN methods
    """

    def __init__(self, results_dirs: List[str], output_dir: str = 'publication_stats'):
        """
        Initialize the statistical tester

        Args:
            results_dirs: List of result directories to analyze
            output_dir: Directory to save statistical analysis results
        """
        self.results_dirs = [Path(d) for d in results_dirs]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Statistical Significance Tester Initialized")
        print(f"Results directories: {len(self.results_dirs)}")
        print(f"Output directory: {self.output_dir}")

    def load_metrics_from_csv(self, results_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all metrics CSV files from a results directory

        Args:
            results_dir: Path to results directory

        Returns:
            Dictionary mapping model_target_phase to DataFrame
        """
        metrics_dir = results_dir / 'case1_h2_hydrogenotrophic_only' / 'detailed_results'
        if not metrics_dir.exists():
            # Try alternate structure
            for subdir in results_dir.iterdir():
                if subdir.is_dir():
                    metrics_dir = subdir / 'detailed_results'
                    if metrics_dir.exists():
                        break

        if not metrics_dir.exists():
            print(f"⚠️  Metrics directory not found in {results_dir}")
            return {}

        metrics_files = list(metrics_dir.glob('*_metrics.csv'))
        print(f"Found {len(metrics_files)} metrics files in {metrics_dir.name}")

        all_metrics = {}
        for csv_file in metrics_files:
            # Parse filename: model_target_phase_metrics.csv
            parts = csv_file.stem.replace('_metrics', '').split('_')
            if len(parts) >= 3:
                model_type = parts[0]
                target = parts[1]
                phase = parts[2]
                key = f"{model_type}_{target}_{phase}"

                df = pd.read_csv(csv_file)
                all_metrics[key] = df

        return all_metrics

    def compute_cohen_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size

        Args:
            group1: First group of values
            group2: Second group of values

        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def paired_t_test(self, method1_scores: np.ndarray, method2_scores: np.ndarray) -> Dict:
        """
        Perform paired t-test and compute effect size

        Args:
            method1_scores: Scores from method 1 (across CV folds)
            method2_scores: Scores from method 2 (across CV folds)

        Returns:
            Dictionary with t-statistic, p-value, Cohen's d
        """
        # Remove NaN values
        mask = ~(np.isnan(method1_scores) | np.isnan(method2_scores))
        clean_scores1 = method1_scores[mask]
        clean_scores2 = method2_scores[mask]

        if len(clean_scores1) < 2:
            return {
                't_statistic': np.nan,
                'p_value': np.nan,
                'cohen_d': np.nan,
                'mean_diff': np.nan,
                'n_pairs': 0
            }

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(clean_scores1, clean_scores2)

        # Effect size
        cohen_d = self.compute_cohen_d(clean_scores1, clean_scores2)

        # Mean difference
        mean_diff = np.mean(clean_scores1) - np.mean(clean_scores2)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'mean_diff': mean_diff,
            'n_pairs': len(clean_scores1),
            'method1_mean': np.mean(clean_scores1),
            'method1_std': np.std(clean_scores1),
            'method2_mean': np.mean(clean_scores2),
            'method2_std': np.std(clean_scores2)
        }

    def wilcoxon_test(self, method1_scores: np.ndarray, method2_scores: np.ndarray) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)

        Args:
            method1_scores: Scores from method 1
            method2_scores: Scores from method 2

        Returns:
            Dictionary with test statistic and p-value
        """
        # Remove NaN values
        mask = ~(np.isnan(method1_scores) | np.isnan(method2_scores))
        clean_scores1 = method1_scores[mask]
        clean_scores2 = method2_scores[mask]

        if len(clean_scores1) < 2:
            return {'statistic': np.nan, 'p_value': np.nan}

        try:
            stat, p_value = stats.wilcoxon(clean_scores1, clean_scores2, alternative='greater')
            return {'statistic': stat, 'p_value': p_value}
        except Exception as e:
            print(f"⚠️  Wilcoxon test failed: {e}")
            return {'statistic': np.nan, 'p_value': np.nan}

    def compare_methods(self, all_metrics: Dict[str, pd.DataFrame],
                       baseline_key: str, treatment_key: str,
                       metric: str = 'r2') -> Dict:
        """
        Compare two methods using paired statistical tests

        Args:
            all_metrics: Dictionary of all loaded metrics
            baseline_key: Key for baseline method (e.g., 'gcn_H2-km_knn')
            treatment_key: Key for treatment method (e.g., 'kg_gt_H2-km_explainer')
            metric: Metric to compare (default: 'r2')

        Returns:
            Dictionary with statistical test results
        """
        if baseline_key not in all_metrics or treatment_key not in all_metrics:
            print(f"⚠️  Missing data for comparison: {baseline_key} vs {treatment_key}")
            return {}

        baseline_df = all_metrics[baseline_key]
        treatment_df = all_metrics[treatment_key]

        # Get fold-level scores (exclude 'overall' row)
        baseline_folds = baseline_df[baseline_df['fold'] != 'overall']
        treatment_folds = treatment_df[treatment_df['fold'] != 'overall']

        baseline_scores = baseline_folds[metric].values
        treatment_scores = treatment_folds[metric].values

        # Ensure same number of folds
        min_folds = min(len(baseline_scores), len(treatment_scores))
        baseline_scores = baseline_scores[:min_folds]
        treatment_scores = treatment_scores[:min_folds]

        # Paired t-test
        t_test_results = self.paired_t_test(treatment_scores, baseline_scores)

        # Wilcoxon test
        wilcoxon_results = self.wilcoxon_test(treatment_scores, baseline_scores)

        return {
            'baseline_method': baseline_key,
            'treatment_method': treatment_key,
            'metric': metric,
            'paired_t_test': t_test_results,
            'wilcoxon_test': wilcoxon_results,
            'baseline_mean': float(np.mean(baseline_scores)),
            'treatment_mean': float(np.mean(treatment_scores)),
            'improvement': float(np.mean(treatment_scores) - np.mean(baseline_scores))
        }

    def generate_comparison_table(self, all_metrics: Dict[str, pd.DataFrame],
                                  target: str = 'H2-km',
                                  metric: str = 'r2',
                                  best_model: str = 'rggc') -> pd.DataFrame:
        """
        Generate comprehensive comparison table for publication

        Args:
            all_metrics: All loaded metrics
            target: Target variable (e.g., 'H2-km', 'ACE-km')
            metric: Metric to compare
            best_model: Best performing model to compare against (default: 'rggc')

        Returns:
            DataFrame with comparison results
        """
        # Define the best model key (RGGC with explainer is the top performer)
        best_model_key = f'{best_model}_{target}_explainer'
        best_model_name = best_model.upper()

        baseline_methods = {
            'GCN (k-NN)': f'gcn_{target}_knn',
            'GCN (Explainer)': f'gcn_{target}_explainer',
            'GAT (k-NN)': f'gat_{target}_knn',
            'GAT (Explainer)': f'gat_{target}_explainer',
            'RGGC (k-NN)': f'rggc_{target}_knn',
            'KG-GT (k-NN)': f'kg_gt_{target}_knn',
            'KG-GT (Explainer)': f'kg_gt_{target}_explainer'
        }

        results_rows = []

        for baseline_name, baseline_key in baseline_methods.items():
            comparison = self.compare_methods(all_metrics, baseline_key, best_model_key, metric)

            if not comparison:
                continue

            t_test = comparison['paired_t_test']
            wilcoxon = comparison['wilcoxon_test']

            # Determine significance level
            p_val = t_test['p_value']
            if np.isnan(p_val):
                sig = 'N/A'
            elif p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            # Effect size interpretation
            cohen_d = t_test['cohen_d']
            if np.isnan(cohen_d):
                effect_interp = 'N/A'
            elif abs(cohen_d) < 0.2:
                effect_interp = 'negligible'
            elif abs(cohen_d) < 0.5:
                effect_interp = 'small'
            elif abs(cohen_d) < 0.8:
                effect_interp = 'medium'
            else:
                effect_interp = 'large'

            results_rows.append({
                'Baseline Method': baseline_name,
                f'Baseline {metric.upper()}': f"{comparison['baseline_mean']:.4f}",
                f'{best_model_name} {metric.upper()}': f"{comparison['treatment_mean']:.4f}",
                'Improvement': f"{comparison['improvement']:+.4f}",
                't-statistic': f"{t_test['t_statistic']:.3f}" if not np.isnan(t_test['t_statistic']) else 'N/A',
                'p-value': f"{p_val:.4f}" if not np.isnan(p_val) else 'N/A',
                'Significance': sig,
                "Cohen's d": f"{cohen_d:.3f}" if not np.isnan(cohen_d) else 'N/A',
                'Effect Size': effect_interp,
                'Wilcoxon p': f"{wilcoxon['p_value']:.4f}" if not np.isnan(wilcoxon['p_value']) else 'N/A'
            })

        return pd.DataFrame(results_rows)

    def run_full_analysis(self, targets: List[str] = None, metrics: List[str] = None,
                         best_model: str = 'rggc'):
        """
        Run complete statistical analysis for all targets and metrics

        Args:
            targets: List of target variables (default: ['H2-km', 'ACE-km'])
            metrics: List of metrics to analyze (default: ['r2', 'rmse', 'mae'])
            best_model: Best performing model to compare against (default: 'rggc')
        """
        if targets is None:
            targets = ['H2-km', 'ACE-km']

        if metrics is None:
            metrics = ['r2', 'rmse', 'mae']

        print(f"\n{'='*80}")
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print(f"{'='*80}\n")

        all_results = {}

        for results_dir in self.results_dirs:
            print(f"\n📁 Processing: {results_dir.name}")
            print(f"{'='*60}")

            # Load all metrics
            all_metrics = self.load_metrics_from_csv(results_dir)

            if not all_metrics:
                print(f"⚠️  No metrics found in {results_dir}")
                continue

            dir_results = {}

            for target in targets:
                print(f"\n🎯 Target: {target}")
                print(f"{'-'*40}")

                target_results = {}

                for metric in metrics:
                    print(f"\n  📊 Metric: {metric.upper()}")

                    # Generate comparison table
                    comparison_table = self.generate_comparison_table(
                        all_metrics, target=target, metric=metric, best_model=best_model
                    )

                    if comparison_table.empty:
                        print(f"    ⚠️  No comparisons available for {target} - {metric}")
                        continue

                    # Save table
                    output_file = self.output_dir / f"{results_dir.name}_{target}_{metric}_comparison.csv"
                    comparison_table.to_csv(output_file, index=False)
                    print(f"    ✅ Saved: {output_file.name}")

                    # Print summary
                    significant_comparisons = comparison_table[
                        comparison_table['Significance'].isin(['*', '**', '***'])
                    ]

                    print(f"    Total comparisons: {len(comparison_table)}")
                    print(f"    Significant improvements: {len(significant_comparisons)}")

                    if len(significant_comparisons) > 0:
                        print(f"\n    Significant Results:")
                        for _, row in significant_comparisons.iterrows():
                            print(f"      • {row['Baseline Method']}: "
                                  f"improvement={row['Improvement']}, "
                                  f"p={row['p-value']}, "
                                  f"effect={row['Effect Size']}")

                    target_results[metric] = {
                        'comparison_table': comparison_table.to_dict('records'),
                        'output_file': str(output_file),
                        'n_comparisons': len(comparison_table),
                        'n_significant': len(significant_comparisons)
                    }

                dir_results[target] = target_results

            all_results[results_dir.name] = dir_results

        # Save comprehensive JSON summary
        summary_file = self.output_dir / 'statistical_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📄 Summary JSON: {summary_file.name}")

        return all_results

    def create_publication_table(self, target: str = 'H2-km', metric: str = 'r2',
                                best_model: str = 'rggc') -> pd.DataFrame:
        """
        Create final publication-ready table with best results

        Args:
            target: Target variable
            metric: Metric to report
            best_model: Best performing model (default: 'rggc')

        Returns:
            Publication-ready DataFrame
        """
        print(f"\n📝 Creating publication table for {target} - {metric.upper()}")

        # Load all results
        all_comparisons = []

        for results_dir in self.results_dirs:
            all_metrics = self.load_metrics_from_csv(results_dir)
            if not all_metrics:
                continue

            comparison_table = self.generate_comparison_table(all_metrics, target, metric, best_model)
            if not comparison_table.empty:
                comparison_table['Case'] = results_dir.name
                all_comparisons.append(comparison_table)

        if not all_comparisons:
            print("⚠️  No data available for publication table")
            return pd.DataFrame()

        # Combine all cases
        combined = pd.concat(all_comparisons, ignore_index=True)

        # Get best model name for column header
        best_model_name = best_model.upper()

        # Create publication table
        pub_table = combined[[
            'Case', 'Baseline Method',
            f'Baseline {metric.upper()}', f'{best_model_name} {metric.upper()}',
            'Improvement', 'p-value', 'Significance', "Cohen's d", 'Effect Size'
        ]]

        # Save publication table
        pub_file = self.output_dir / f'publication_table_{target}_{metric}.csv'
        pub_table.to_csv(pub_file, index=False)

        print(f"✅ Publication table saved: {pub_file.name}")

        # Also create LaTeX version
        latex_file = self.output_dir / f'publication_table_{target}_{metric}.tex'
        with open(latex_file, 'w') as f:
            f.write(pub_table.to_latex(index=False, escape=False))

        print(f"✅ LaTeX table saved: {latex_file.name}")

        return pub_table


def main():
    parser = argparse.ArgumentParser(description='Statistical Significance Testing for GNN Publication')
    parser.add_argument('--results_dirs', nargs='+', default=['enhanced_results_case1'],
                       help='Directories containing results to analyze')
    parser.add_argument('--output_dir', default='publication_stats',
                       help='Output directory for statistical analysis results')
    parser.add_argument('--targets', nargs='+', default=['H2-km', 'ACE-km'],
                       help='Target variables to analyze')
    parser.add_argument('--metrics', nargs='+', default=['r2', 'rmse', 'mae'],
                       help='Metrics to analyze')
    parser.add_argument('--best_model', default='rggc',
                       help='Best performing model to compare against (default: rggc)')

    args = parser.parse_args()

    # Initialize tester
    tester = StatisticalSignificanceTester(
        results_dirs=args.results_dirs,
        output_dir=args.output_dir
    )

    # Run full analysis
    results = tester.run_full_analysis(
        targets=args.targets,
        metrics=args.metrics,
        best_model=args.best_model
    )

    # Create publication tables
    print(f"\n{'='*80}")
    print("CREATING PUBLICATION TABLES")
    print(f"{'='*80}")

    for target in args.targets:
        for metric in ['r2']:  # Focus on R² for main publication table
            pub_table = tester.create_publication_table(target=target, metric=metric, best_model=args.best_model)

            if not pub_table.empty:
                print(f"\n{target} - {metric.upper()} Summary:")
                print(pub_table.to_string(index=False))

    print(f"\n🎉 Statistical analysis complete!")
    print(f"📁 All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
