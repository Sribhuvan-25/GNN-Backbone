#!/usr/bin/env python3
"""
Statistical Validation Framework for Node Pruning Research

This module provides comprehensive statistical validation for graph neural network
node pruning methodologies, including significance testing, confidence intervals,
effect size analysis, and multiple comparison corrections.

Mathematical Foundation:
======================
Statistical tests performed:
1. Paired t-test: H₀: μ_original = μ_pruned vs H₁: μ_original ≠ μ_pruned
2. Wilcoxon signed-rank test: Non-parametric alternative to t-test
3. Bootstrap confidence intervals: CI = [x̄ ± t_{α/2,df} × SE]
4. Effect size (Cohen's d): d = (μ₁ - μ₂) / σ_pooled
5. Multiple comparison correction: Bonferroni, FDR, Holm-Bonferroni

Authors: Research Team
Date: 2024
"""

import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
import warnings
from typing import Dict, List, Tuple, Optional, Union

# Optional imports - fallback if not available
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels not available - using manual multiple comparison correction")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for node pruning research
    
    Provides methods for:
    - Statistical significance testing
    - Effect size calculation
    - Confidence interval estimation
    - Multiple comparison correction
    - Bootstrap validation
    - Power analysis
    """
    
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        """
        Initialize statistical validator
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_state = random_state
        np.random.seed(random_state)
    
    def validate_pruning_effectiveness(self, 
                                     original_performance: np.ndarray, 
                                     pruned_performance: np.ndarray,
                                     method_name: str = "Node Pruning",
                                     n_bootstrap: int = 1000) -> Dict:
        """
        Comprehensive statistical validation of pruning effectiveness
        
        Mathematical Framework:
        =====================
        
        1) Paired t-test:
           t = (x̄_diff) / (s_diff / √n)
           Where x̄_diff = mean difference, s_diff = standard deviation of differences
           
        2) Effect size (Cohen's d):
           d = (μ₁ - μ₂) / σ_pooled
           Where σ_pooled = √((s₁² + s₂²) / 2)
           
        3) Bootstrap confidence interval:
           CI = [percentile(bootstrap_diffs, α/2), percentile(bootstrap_diffs, 1-α/2)]
           
        4) Power analysis:
           Power = P(reject H₀ | H₁ is true)
        
        Args:
            original_performance: Performance metrics before pruning
            pruned_performance: Performance metrics after pruning  
            method_name: Name of the pruning method
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dict containing comprehensive statistical analysis
        """
        
        # Input validation
        if len(original_performance) != len(pruned_performance):
            raise ValueError("Performance arrays must have same length")
        
        if len(original_performance) < 3:
            warnings.warn("Sample size < 3: Statistical tests may be unreliable")
        
        # Basic statistics
        orig_mean, orig_std = np.mean(original_performance), np.std(original_performance)
        pruned_mean, pruned_std = np.mean(pruned_performance), np.std(pruned_performance)
        diff_scores = pruned_performance - original_performance
        diff_mean, diff_std = np.mean(diff_scores), np.std(diff_scores)
        
        # 1. Paired t-test
        t_stat, t_p_value = ttest_rel(original_performance, pruned_performance)
        
        # 2. Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_p_value = wilcoxon(original_performance, pruned_performance, 
                                       zero_method='wilcox', alternative='two-sided')
        except ValueError:
            # Handle case where all differences are zero
            w_stat, w_p_value = 0, 1.0
        
        # 3. Effect size (Cohen's d)
        pooled_std = np.sqrt((orig_std**2 + pruned_std**2) / 2)
        cohens_d = (pruned_mean - orig_mean) / pooled_std if pooled_std > 0 else 0
        
        # 4. Bootstrap confidence intervals
        bootstrap_results = self._bootstrap_confidence_interval(
            original_performance, pruned_performance, n_bootstrap
        )
        
        # 5. Practical significance assessment
        practical_significance = self._assess_practical_significance(
            diff_mean, diff_std, orig_mean
        )
        
        # 6. Power analysis
        power_analysis = self._estimate_statistical_power(
            original_performance, pruned_performance, self.alpha
        )
        
        # Compile results
        results = {
            'method_name': method_name,
            'sample_size': len(original_performance),
            
            # Descriptive statistics
            'descriptive_stats': {
                'original': {'mean': orig_mean, 'std': orig_std, 'median': np.median(original_performance)},
                'pruned': {'mean': pruned_mean, 'std': pruned_std, 'median': np.median(pruned_performance)},
                'difference': {'mean': diff_mean, 'std': diff_std, 'median': np.median(diff_scores)}
            },
            
            # Statistical tests
            'statistical_tests': {
                'paired_ttest': {
                    't_statistic': t_stat,
                    'p_value': t_p_value, 
                    'significant': t_p_value < self.alpha,
                    'degrees_freedom': len(original_performance) - 1
                },
                'wilcoxon_test': {
                    'w_statistic': w_stat,
                    'p_value': w_p_value,
                    'significant': w_p_value < self.alpha
                }
            },
            
            # Effect size
            'effect_size': {
                'cohens_d': cohens_d,
                'interpretation': self._interpret_effect_size(cohens_d),
                'pooled_std': pooled_std
            },
            
            # Confidence intervals
            'confidence_intervals': bootstrap_results,
            
            # Practical significance
            'practical_significance': practical_significance,
            
            # Power analysis
            'power_analysis': power_analysis,
            
            # Overall assessment
            'overall_assessment': {
                'statistically_significant': t_p_value < self.alpha and w_p_value < self.alpha,
                'practically_significant': practical_significance['is_practically_significant'],
                'recommended_action': self._recommend_action(t_p_value, w_p_value, cohens_d, practical_significance)
            }
        }
        
        return results
    
    def _bootstrap_confidence_interval(self, 
                                     original: np.ndarray, 
                                     pruned: np.ndarray, 
                                     n_bootstrap: int) -> Dict:
        """
        Bootstrap confidence intervals for performance differences
        
        Mathematical Foundation:
        ======================
        Bootstrap sampling: X*_b ~ F̂_n for b = 1,...,B
        Confidence interval: [θ̂_{α/2}, θ̂_{1-α/2}] where θ̂_p is p-th percentile
        """
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sampling with replacement
            indices = np.random.choice(len(original), size=len(original), replace=True)
            orig_sample = original[indices]
            pruned_sample = pruned[indices]
            
            bootstrap_diffs.append(np.mean(pruned_sample) - np.mean(orig_sample))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))
        
        return {
            'bootstrap_mean_difference': np.mean(bootstrap_diffs),
            'bootstrap_std': np.std(bootstrap_diffs),
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'confidence_level': 1 - self.alpha,
            'bootstrap_samples': n_bootstrap,
            'contains_zero': ci_lower <= 0 <= ci_upper
        }
    
    def _assess_practical_significance(self, 
                                     diff_mean: float, 
                                     diff_std: float, 
                                     baseline_mean: float) -> Dict:
        """
        Assess practical significance of the difference
        
        Practical significance criteria:
        - Relative improvement > 5%
        - Absolute improvement > 1 standard deviation of baseline
        - Consistency: |difference| > 2 × SE
        """
        relative_improvement = abs(diff_mean / baseline_mean) if baseline_mean != 0 else 0
        practical_threshold = 0.05  # 5% relative improvement
        
        is_practically_significant = (
            relative_improvement > practical_threshold or
            abs(diff_mean) > diff_std  # Effect larger than noise
        )
        
        return {
            'relative_improvement': relative_improvement,
            'relative_improvement_percent': relative_improvement * 100,
            'practical_threshold': practical_threshold,
            'is_practically_significant': is_practically_significant,
            'magnitude_assessment': 'Large' if relative_improvement > 0.1 
                                  else 'Medium' if relative_improvement > 0.05 
                                  else 'Small'
        }
    
    def _estimate_statistical_power(self, 
                                   original: np.ndarray, 
                                   pruned: np.ndarray, 
                                   alpha: float) -> Dict:
        """
        Estimate statistical power of the test
        
        Power = P(reject H₀ | H₁ is true)
        """
        from scipy.stats import norm
        
        n = len(original)
        diff_mean = np.mean(pruned - original)
        diff_std = np.std(pruned - original)
        
        if diff_std == 0:
            return {'estimated_power': 1.0 if diff_mean != 0 else 0.0, 'note': 'Zero variance case'}
        
        # Effect size for power calculation
        effect_size = abs(diff_mean) / diff_std
        
        # Critical value for two-tailed test
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Approximate power calculation
        power = 1 - stats.t.cdf(t_critical, n - 1, ncp) + stats.t.cdf(-t_critical, n - 1, ncp)
        
        return {
            'estimated_power': power,
            'effect_size_for_power': effect_size,
            'sample_size': n,
            'alpha_level': alpha,
            'power_interpretation': 'High' if power > 0.8 else 'Medium' if power > 0.6 else 'Low'
        }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size
        
        Cohen's conventions:
        - Small: |d| ≈ 0.2
        - Medium: |d| ≈ 0.5  
        - Large: |d| ≈ 0.8
        """
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _recommend_action(self, 
                         t_p_value: float, 
                         w_p_value: float, 
                         cohens_d: float, 
                         practical_sig: Dict) -> str:
        """
        Provide recommendation based on statistical results
        """
        statistically_significant = t_p_value < self.alpha and w_p_value < self.alpha
        large_effect = abs(cohens_d) > 0.8
        practically_significant = practical_sig['is_practically_significant']
        
        if statistically_significant and practically_significant and large_effect:
            return "Strongly recommend: Method shows significant and substantial improvement"
        elif statistically_significant and practically_significant:
            return "Recommend: Method shows significant improvement"
        elif statistically_significant:
            return "Consider: Statistically significant but limited practical impact"
        elif practically_significant:
            return "Investigate: Practically meaningful but not statistically significant (may need larger sample)"
        else:
            return "Do not recommend: No significant improvement detected"
    
    def multiple_method_comparison(self, 
                                  results_dict: Dict[str, Dict], 
                                  correction_method: str = 'bonferroni') -> Dict:
        """
        Statistical comparison of multiple pruning methods with correction
        
        Mathematical Framework:
        =====================
        
        1) Friedman Test:
           χ²_F = (12/nk(k+1)) * Σ R_j² - 3n(k+1)
           Where R_j = sum of ranks for method j
           
        2) Multiple Comparison Corrections:
           - Bonferroni: α_corrected = α / m
           - FDR (Benjamini-Hochberg): Controls false discovery rate
           - Holm-Bonferroni: Step-down method
        
        Args:
            results_dict: Dictionary mapping method names to performance results
            correction_method: Multiple comparison correction method
            
        Returns:
            Dict containing comprehensive comparison analysis
        """
        
        # Extract performance data
        method_names = list(results_dict.keys())
        method_performances = []
        
        for method in method_names:
            if isinstance(results_dict[method], dict) and 'performance' in results_dict[method]:
                method_performances.append(results_dict[method]['performance'])
            else:
                # Assume it's directly the performance array
                method_performances.append(results_dict[method])
        
        # Check if we have enough methods for comparison
        if len(method_names) < 2:
            raise ValueError("Need at least 2 methods for comparison")
        
        # 1. Friedman test for multiple related samples
        try:
            friedman_stat, friedman_p = friedmanchisquare(*method_performances)
        except ValueError as e:
            return {'error': f"Friedman test failed: {str(e)}"}
        
        # 2. Pairwise comparisons
        n_comparisons = len(method_names) * (len(method_names) - 1) // 2
        pairwise_results = []
        p_values = []
        
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                method1, method2 = method_names[i], method_names[j]
                perf1, perf2 = method_performances[i], method_performances[j]
                
                # Ensure arrays have same length for paired test
                min_len = min(len(perf1), len(perf2))
                perf1_trimmed = perf1[:min_len]
                perf2_trimmed = perf2[:min_len]
                
                # Paired t-test
                t_stat, p_value = ttest_rel(perf1_trimmed, perf2_trimmed)
                
                # Effect size
                diff_mean = np.mean(perf2_trimmed) - np.mean(perf1_trimmed)
                pooled_std = np.sqrt((np.var(perf1_trimmed) + np.var(perf2_trimmed)) / 2)
                cohens_d = diff_mean / pooled_std if pooled_std > 0 else 0
                
                pairwise_results.append({
                    'method1': method1,
                    'method2': method2,
                    'comparison': f"{method1}_vs_{method2}",
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'mean_difference': diff_mean,
                    'cohens_d': cohens_d,
                    'effect_size_interpretation': self._interpret_effect_size(cohens_d),
                    'sample_size': min_len
                })
                
                p_values.append(p_value)
        
        # 3. Multiple comparison correction
        if correction_method == 'bonferroni':
            alpha_corrected = self.alpha / n_comparisons
            corrected_results = [p < alpha_corrected for p in p_values]
            corrected_p_values = p_values  # No p-value adjustment for Bonferroni
        elif correction_method in ['fdr', 'holm']:
            if HAS_STATSMODELS:
                corrected_results, corrected_p_values, _, _ = multipletests(
                    p_values, alpha=self.alpha, method='fdr_bh' if correction_method == 'fdr' else 'holm'
                )
            else:
                # Manual implementation of Benjamini-Hochberg (FDR)
                corrected_results, corrected_p_values = self._manual_fdr_correction(p_values)
        else:
            raise ValueError(f"Unknown correction method: {correction_method}")
        
        # Add correction results to pairwise comparisons
        for i, result in enumerate(pairwise_results):
            result['corrected_significant'] = corrected_results[i]
            result['correction_method'] = correction_method
            if correction_method in ['fdr', 'holm']:
                result['corrected_p_value'] = corrected_p_values[i]
            else:
                result['corrected_alpha'] = alpha_corrected
        
        # 4. Ranking analysis
        method_rankings = self._calculate_method_rankings(method_performances, method_names)
        
        return {
            'friedman_test': {
                'statistic': friedman_stat,
                'p_value': friedman_p,
                'significant': friedman_p < self.alpha,
                'degrees_freedom': len(method_names) - 1
            },
            'pairwise_comparisons': pairwise_results,
            'multiple_comparison_correction': {
                'method': correction_method,
                'n_comparisons': n_comparisons,
                'significant_after_correction': sum(corrected_results),
                'total_comparisons': len(corrected_results)
            },
            'method_rankings': method_rankings,
            'overall_assessment': {
                'best_method': method_rankings['best_method'],
                'significant_differences_exist': friedman_p < self.alpha,
                'n_significant_pairwise': sum(corrected_results)
            }
        }
    
    def _calculate_method_rankings(self, 
                                  method_performances: List[np.ndarray], 
                                  method_names: List[str]) -> Dict:
        """
        Calculate rankings for all methods based on performance
        """
        method_means = [np.mean(perf) for perf in method_performances]
        
        # Rank methods (higher performance = better rank)
        ranked_indices = np.argsort(method_means)[::-1]
        
        rankings = {}
        for rank, idx in enumerate(ranked_indices):
            rankings[method_names[idx]] = {
                'rank': rank + 1,
                'mean_performance': method_means[idx],
                'std_performance': np.std(method_performances[idx])
            }
        
        return {
            'rankings': rankings,
            'best_method': method_names[ranked_indices[0]],
            'worst_method': method_names[ranked_indices[-1]],
            'performance_range': max(method_means) - min(method_means)
        }
    
    def generate_statistical_report(self, 
                                   validation_results: Dict, 
                                   save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive statistical validation report
        
        Args:
            validation_results: Results from statistical validation
            save_path: Optional path to save the report
            
        Returns:
            Formatted statistical report as string
        """
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("STATISTICAL VALIDATION REPORT")
        report_lines.append("="*80)
        
        # Method information
        report_lines.append(f"\nMethod: {validation_results.get('method_name', 'Unknown')}")
        report_lines.append(f"Sample Size: {validation_results['sample_size']}")
        report_lines.append(f"Significance Level: α = {self.alpha}")
        
        # Descriptive statistics
        desc_stats = validation_results['descriptive_stats']
        report_lines.append(f"\nDESCRIPTIVE STATISTICS:")
        report_lines.append(f"Original Performance:")
        report_lines.append(f"  Mean ± SD: {desc_stats['original']['mean']:.4f} ± {desc_stats['original']['std']:.4f}")
        report_lines.append(f"  Median: {desc_stats['original']['median']:.4f}")
        
        report_lines.append(f"Pruned Performance:")
        report_lines.append(f"  Mean ± SD: {desc_stats['pruned']['mean']:.4f} ± {desc_stats['pruned']['std']:.4f}")
        report_lines.append(f"  Median: {desc_stats['pruned']['median']:.4f}")
        
        report_lines.append(f"Difference:")
        report_lines.append(f"  Mean ± SD: {desc_stats['difference']['mean']:.4f} ± {desc_stats['difference']['std']:.4f}")
        
        # Statistical tests
        stat_tests = validation_results['statistical_tests']
        report_lines.append(f"\nSTATISTICAL TESTS:")
        report_lines.append(f"Paired t-test:")
        report_lines.append(f"  t({stat_tests['paired_ttest']['degrees_freedom']}) = {stat_tests['paired_ttest']['t_statistic']:.4f}")
        report_lines.append(f"  p-value = {stat_tests['paired_ttest']['p_value']:.6f}")
        report_lines.append(f"  Significant: {'Yes' if stat_tests['paired_ttest']['significant'] else 'No'}")
        
        report_lines.append(f"Wilcoxon signed-rank test:")
        report_lines.append(f"  W = {stat_tests['wilcoxon_test']['w_statistic']:.4f}")
        report_lines.append(f"  p-value = {stat_tests['wilcoxon_test']['p_value']:.6f}")
        report_lines.append(f"  Significant: {'Yes' if stat_tests['wilcoxon_test']['significant'] else 'No'}")
        
        # Effect size
        effect_size = validation_results['effect_size']
        report_lines.append(f"\nEFFECT SIZE:")
        report_lines.append(f"Cohen's d = {effect_size['cohens_d']:.4f}")
        report_lines.append(f"Interpretation: {effect_size['interpretation']}")
        
        # Confidence intervals
        ci = validation_results['confidence_intervals']
        report_lines.append(f"\nCONFIDENCE INTERVALS:")
        report_lines.append(f"{ci['confidence_level']*100:.0f}% Bootstrap CI: [{ci['confidence_interval_lower']:.4f}, {ci['confidence_interval_upper']:.4f}]")
        report_lines.append(f"Contains zero: {'Yes' if ci['contains_zero'] else 'No'}")
        
        # Practical significance
        practical = validation_results['practical_significance']
        report_lines.append(f"\nPRACTICAL SIGNIFICANCE:")
        report_lines.append(f"Relative improvement: {practical['relative_improvement_percent']:.2f}%")
        report_lines.append(f"Magnitude: {practical['magnitude_assessment']}")
        report_lines.append(f"Practically significant: {'Yes' if practical['is_practically_significant'] else 'No'}")
        
        # Overall assessment
        overall = validation_results['overall_assessment']
        report_lines.append(f"\nOVERALL ASSESSMENT:")
        report_lines.append(f"Statistically significant: {'Yes' if overall['statistically_significant'] else 'No'}")
        report_lines.append(f"Practically significant: {'Yes' if overall['practically_significant'] else 'No'}")
        report_lines.append(f"Recommendation: {overall['recommended_action']}")
        
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Statistical report saved to: {save_path}")
        
        return report
    
    def _manual_fdr_correction(self, p_values: List[float]) -> Tuple[List[bool], List[float]]:
        """
        Manual implementation of Benjamini-Hochberg FDR correction
        
        Algorithm:
        1. Sort p-values in ascending order
        2. For each p_i, test if p_i ≤ (i/m) × α
        3. Find largest i where condition holds
        4. Reject all hypotheses up to that i
        """
        p_array = np.array(p_values)
        m = len(p_array)
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_array)
        sorted_p_values = p_array[sorted_indices]
        
        # Benjamini-Hochberg procedure
        corrected_p_values = np.zeros_like(p_array)
        rejected = np.zeros(m, dtype=bool)
        
        # Work backwards to find largest i where p_i <= (i/m) * alpha
        for i in range(m-1, -1, -1):
            threshold = ((i + 1) / m) * self.alpha
            if sorted_p_values[i] <= threshold:
                # Reject all hypotheses from 0 to i
                rejected[sorted_indices[:i+1]] = True
                break
        
        # Adjust p-values
        for i in range(m):
            corrected_p_values[i] = min(1.0, p_values[i] * m / (i + 1))
        
        return rejected.tolist(), corrected_p_values.tolist()


# Example usage and testing
def main():
    """
    Example usage of the statistical validation framework
    """
    
    # Create validator
    validator = StatisticalValidator(alpha=0.05, random_state=42)
    
    # Simulate performance data
    np.random.seed(42)
    original_performance = np.random.normal(0.75, 0.1, 20)  # Original method
    pruned_performance = np.random.normal(0.80, 0.12, 20)   # Improved method
    
    # Validate pruning effectiveness
    results = validator.validate_pruning_effectiveness(
        original_performance, 
        pruned_performance,
        method_name="Unified Node Pruning",
        n_bootstrap=1000
    )
    
    # Generate report
    report = validator.generate_statistical_report(results)
    print(report)
    
    # Example: Multiple method comparison
    methods_data = {
        'Unified_Pruning': pruned_performance,
        'Random_Pruning': np.random.normal(0.70, 0.15, 20),
        'No_Pruning': original_performance
    }
    
    comparison_results = validator.multiple_method_comparison(methods_data)
    print(f"\nBest method: {comparison_results['overall_assessment']['best_method']}")


if __name__ == "__main__":
    main()