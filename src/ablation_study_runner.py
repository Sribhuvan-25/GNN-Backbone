#!/usr/bin/env python3
"""
Ablation Study Runner for GNN Models

This script performs comprehensive ablation studies to isolate the contribution of:
1. GNN architecture alone (without domain expert features)
2. Domain expert knowledge alone (anchored features only)
3. Edge-based sparsification (GNNExplainer pruning)
4. Full system (Best GNN model + domain knowledge + sparsification)

The ablation study systematically removes components to quantify their individual
and combined contributions to model performance. By default, it evaluates RGGC as
the best-performing model.

Usage:
    python ablation_study_runner.py --case case1 --epochs 100

Examples:
    # Run full ablation study for Case 1
    python ablation_study_runner.py --case case1

    # Quick test with minimal configuration
    python ablation_study_runner.py --case case1 --quick

    # Run specific ablation components only
    python ablation_study_runner.py --case case1 --components gnn_only domain_only full_system
"""

import argparse
import os
import sys
import time
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AblationStudyRunner:
    """
    Runs comprehensive ablation studies for Knowledge-Guided GNN
    """

    def __init__(self, case_type: str, data_path: str, output_dir: str,
                 num_epochs: int = 100, num_folds: int = 5, quick: bool = False):
        """
        Initialize ablation study runner

        Args:
            case_type: Domain expert case (case1, case2, etc.)
            data_path: Path to dataset
            output_dir: Directory to save ablation results
            num_epochs: Training epochs per configuration
            num_folds: Cross-validation folds
            quick: Quick test mode with minimal configuration
        """
        self.case_type = case_type
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_epochs = num_epochs
        self.num_folds = num_folds
        self.quick = quick

        # Adjust for quick mode
        if quick:
            self.num_epochs = 5
            self.num_folds = 2

        self.results = {}

        print(f"{'='*80}")
        print("ABLATION STUDY RUNNER")
        print(f"{'='*80}")
        print(f"Case: {case_type}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Folds: {self.num_folds}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")

    def run_baseline_gnn_only(self) -> Dict:
        """
        Run GNN without domain expert features (vanilla GNN on correlation graph)

        This isolates the contribution of the GNN architecture alone.

        Returns:
            Dictionary with baseline GNN results
        """
        print(f"\n{'='*60}")
        print("ABLATION 1: GNN Architecture Only (No Domain Knowledge)")
        print(f"{'='*60}")
        print("Configuration:")
        print("  ✓ GNN models: GCN, GAT, RGGC")
        print("  ✗ Anchored domain expert features")
        print("  ✓ k-NN graph (correlation-based)")
        print("  ✗ GNNExplainer sparsification")
        print(f"{'='*60}\n")

        try:
            from pipelines.embeddings_pipeline import MixedEmbeddingPipeline

            config = {
                'data_path': self.data_path,
                'num_epochs': self.num_epochs,
                'num_folds': self.num_folds,
                'use_nested_cv': not self.quick,
                'save_dir': str(self.output_dir / 'ablation_gnn_only'),
                'k_neighbors': 10,
                'hidden_dim': 64,
                'dropout_rate': 0.3,
                'batch_size': 8,
                'learning_rate': 0.001,
                'patience': 20 if not self.quick else 5,
                # NO domain expert features
                'use_domain_features': False,
                'use_explainer': False  # No explainer sparsification
            }

            print("Initializing baseline GNN pipeline...")
            start_time = time.time()

            pipeline = MixedEmbeddingPipeline(**config)
            results = pipeline.run_mixed_embedding_pipeline()

            runtime = time.time() - start_time

            print(f"\n✅ Baseline GNN completed in {runtime/60:.2f} minutes")

            # Extract key metrics
            summary = self._extract_summary(results, 'GNN Only')

            return {
                'config': config,
                'results': results,
                'summary': summary,
                'runtime_minutes': runtime / 60
            }

        except Exception as e:
            print(f"❌ Baseline GNN failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def run_domain_knowledge_only(self) -> Dict:
        """
        Run with domain expert features only (no GNN, just classical ML on anchored features)

        This isolates the contribution of domain expert knowledge.

        Returns:
            Dictionary with domain-knowledge-only results
        """
        print(f"\n{'='*60}")
        print("ABLATION 2: Domain Expert Knowledge Only")
        print(f"{'='*60}")
        print("Configuration:")
        print("  ✗ GNN models")
        print("  ✓ Anchored domain expert features")
        print("  ✓ Classical ML models (LinearSVR, ExtraTrees)")
        print(f"{'='*60}\n")

        try:
            from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline

            config = {
                'data_path': self.data_path,
                'case_type': self.case_type,
                'num_epochs': self.num_epochs,
                'num_folds': self.num_folds,
                'use_nested_cv': not self.quick,
                'save_dir': str(self.output_dir / 'ablation_domain_only'),
                'k_neighbors': 10,
                'hidden_dim': 64,
                'dropout_rate': 0.3,
                'batch_size': 8,
                'learning_rate': 0.001,
                'patience': 20 if not self.quick else 5,
                'importance_threshold': 0.2,
                'use_node_pruning': False,
                # ONLY domain features, NO GNN embeddings
                'skip_gnn_training': True  # Skip GNN, use only domain features
            }

            print("Initializing domain-knowledge-only pipeline...")
            start_time = time.time()

            pipeline = DomainExpertCasesPipeline(**config)

            # Run only classical ML on anchored features
            results = self._run_domain_features_only(pipeline)

            runtime = time.time() - start_time

            print(f"\n✅ Domain knowledge only completed in {runtime/60:.2f} minutes")

            summary = self._extract_summary(results, 'Domain Knowledge Only')

            return {
                'config': config,
                'results': results,
                'summary': summary,
                'runtime_minutes': runtime / 60
            }

        except Exception as e:
            print(f"❌ Domain knowledge only failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def run_gnn_without_sparsification(self) -> Dict:
        """
        Run GNN + domain knowledge WITHOUT explainer sparsification

        This isolates the contribution of edge sparsification.

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print("ABLATION 3: GNN + Domain Knowledge (No Sparsification)")
        print(f"{'='*60}")
        print("Configuration:")
        print("  ✓ GNN models (including KG-GT)")
        print("  ✓ Anchored domain expert features")
        print("  ✓ k-NN graph")
        print("  ✗ GNNExplainer sparsification")
        print(f"{'='*60}\n")

        try:
            from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline

            config = {
                'data_path': self.data_path,
                'case_type': self.case_type,
                'num_epochs': self.num_epochs,
                'num_folds': self.num_folds,
                'use_nested_cv': not self.quick,
                'save_dir': str(self.output_dir / 'ablation_no_sparsification'),
                'k_neighbors': 10,
                'hidden_dim': 64,
                'dropout_rate': 0.3,
                'batch_size': 8,
                'learning_rate': 0.001,
                'patience': 20 if not self.quick else 5,
                'use_node_pruning': False,
                # NO explainer sparsification
                'skip_explainer': True
            }

            print("Initializing GNN without sparsification...")
            start_time = time.time()

            pipeline = DomainExpertCasesPipeline(**config)

            # Run only k-NN training (no explainer)
            results = self._run_knn_only(pipeline)

            runtime = time.time() - start_time

            print(f"\n✅ GNN without sparsification completed in {runtime/60:.2f} minutes")

            summary = self._extract_summary(results, 'GNN + Domain (No Sparsification)')

            return {
                'config': config,
                'results': results,
                'summary': summary,
                'runtime_minutes': runtime / 60
            }

        except Exception as e:
            print(f"❌ GNN without sparsification failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def run_full_system(self) -> Dict:
        """
        Run complete system: GNN + domain knowledge + explainer sparsification

        This is the full KG-GT system with all components.

        Returns:
            Dictionary with full system results
        """
        print(f"\n{'='*60}")
        print("ABLATION 4: Full System (All Components)")
        print(f"{'='*60}")
        print("Configuration:")
        print("  ✓ All GNN models (GCN, GAT, RGGC, KG-GT)")
        print("  ✓ Anchored domain expert features")
        print("  ✓ k-NN graph")
        print("  ✓ GNNExplainer edge sparsification")
        print(f"{'='*60}\n")

        try:
            from pipelines.domain_expert_cases_pipeline_refactored import DomainExpertCasesPipeline

            config = {
                'data_path': self.data_path,
                'case_type': self.case_type,
                'num_epochs': self.num_epochs,
                'num_folds': self.num_folds,
                'use_nested_cv': not self.quick,
                'save_dir': str(self.output_dir / 'ablation_full_system'),
                'k_neighbors': 10,
                'hidden_dim': 64,
                'dropout_rate': 0.3,
                'batch_size': 8,
                'learning_rate': 0.001,
                'patience': 20 if not self.quick else 5,
                'importance_threshold': 0.2,
                'use_node_pruning': False
            }

            print("Initializing full system pipeline...")
            start_time = time.time()

            pipeline = DomainExpertCasesPipeline(**config)
            results = pipeline.run_case_specific_pipeline()

            runtime = time.time() - start_time

            print(f"\n✅ Full system completed in {runtime/60:.2f} minutes")

            summary = self._extract_summary(results, 'Full System')

            return {
                'config': config,
                'results': results,
                'summary': summary,
                'runtime_minutes': runtime / 60
            }

        except Exception as e:
            print(f"❌ Full system failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _run_domain_features_only(self, pipeline) -> Dict:
        """
        Run classical ML on domain expert features only (helper method)

        Args:
            pipeline: Initialized pipeline

        Returns:
            Results dictionary
        """
        from sklearn.model_selection import cross_validate
        from sklearn.svm import LinearSVR
        from sklearn.ensemble import ExtraTreesRegressor

        print("Training classical ML on anchored features...")

        results = {}

        # Get anchored features
        case_features = pipeline.get_case_specific_features()

        for target_idx, target_name in enumerate(pipeline.dataset.target_names):
            print(f"\nProcessing {target_name}...")

            target_results = {}

            # Get target values
            y = np.array([data.y[target_idx].item() for data in pipeline.dataset.data_list])

            # Get anchored features for this target
            anchored_families = case_features.get(target_name, [])

            if not anchored_families:
                print(f"  ⚠️  No anchored features for {target_name}")
                continue

            # Extract features
            X_anchored = []
            for data in pipeline.dataset.data_list:
                # Get node features for anchored families
                family_features = []
                for family in anchored_families:
                    if family in pipeline.dataset.node_feature_names:
                        idx = pipeline.dataset.node_feature_names.index(family)
                        family_features.append(data.x[idx, 0].item())

                X_anchored.append(family_features)

            X_anchored = np.array(X_anchored)

            print(f"  Anchored features shape: {X_anchored.shape}")

            # Train models
            models = {
                'LinearSVR': LinearSVR(max_iter=10000),
                'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42)
            }

            for model_name, model in models.items():
                print(f"  Training {model_name}...")

                cv_results = cross_validate(
                    model, X_anchored, y,
                    cv=self.num_folds,
                    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                    return_train_score=True
                )

                target_results[model_name] = {
                    'test_r2': cv_results['test_r2'].mean(),
                    'test_r2_std': cv_results['test_r2'].std(),
                    'test_mse': -cv_results['test_neg_mean_squared_error'].mean(),
                    'test_mae': -cv_results['test_neg_mean_absolute_error'].mean()
                }

                print(f"    R² = {target_results[model_name]['test_r2']:.4f} ± {target_results[model_name]['test_r2_std']:.4f}")

            results[target_name] = target_results

        return results

    def _run_knn_only(self, pipeline) -> Dict:
        """
        Run k-NN training only (no explainer) - helper method

        Args:
            pipeline: Initialized pipeline

        Returns:
            Results dictionary
        """
        print("Running k-NN training without explainer...")

        results = {}

        for target_idx, target_name in enumerate(pipeline.dataset.target_names):
            print(f"\nProcessing {target_name}...")

            # Train on k-NN graph only
            knn_results = pipeline.train_all_models_cv(target_idx, target_name)

            results[target_name] = {
                'knn_training': knn_results
            }

        return results

    def _extract_summary(self, results: Dict, component_name: str) -> Dict:
        """
        Extract summary metrics from results

        Args:
            results: Results dictionary
            component_name: Name of ablation component

        Returns:
            Summary dictionary
        """
        summary = {'component': component_name, 'targets': {}}

        if not results or 'error' in results:
            return summary

        for target_name, target_data in results.items():
            if not isinstance(target_data, dict):
                continue

            target_summary = {}

            # Extract best R² from different training phases
            best_r2 = -np.inf

            # Check k-NN training
            if 'knn_training' in target_data:
                for model_key, model_data in target_data['knn_training'].items():
                    if 'test_metrics' in model_data:
                        r2 = model_data['test_metrics'].get('r2_score', -np.inf)
                        best_r2 = max(best_r2, r2)

            # Check explainer training
            if 'explainer_training' in target_data:
                for model_key, model_data in target_data['explainer_training'].items():
                    if 'test_metrics' in model_data:
                        r2 = model_data['test_metrics'].get('r2_score', -np.inf)
                        best_r2 = max(best_r2, r2)

            # Check classical ML results
            if 'LinearSVR' in target_data or 'ExtraTrees' in target_data:
                for ml_model in ['LinearSVR', 'ExtraTrees']:
                    if ml_model in target_data:
                        r2 = target_data[ml_model].get('test_r2', -np.inf)
                        best_r2 = max(best_r2, r2)

            target_summary['best_r2'] = float(best_r2) if best_r2 != -np.inf else None

            summary['targets'][target_name] = target_summary

        return summary

    def run_complete_ablation_study(self, components: List[str] = None) -> Dict:
        """
        Run complete ablation study with all components

        Args:
            components: List of components to run (default: all)
                       Options: ['gnn_only', 'domain_only', 'no_sparsification', 'full_system']

        Returns:
            Complete ablation results
        """
        if components is None:
            components = ['gnn_only', 'domain_only', 'no_sparsification', 'full_system']

        print(f"\n{'='*80}")
        print("RUNNING COMPLETE ABLATION STUDY")
        print(f"{'='*80}")
        print(f"Components to evaluate: {len(components)}")
        for comp in components:
            print(f"  • {comp}")
        print(f"{'='*80}\n")

        total_start = time.time()

        ablation_results = {}

        # Component 1: GNN Only
        if 'gnn_only' in components:
            print("\n[1/4] Running GNN architecture only...")
            ablation_results['gnn_only'] = self.run_baseline_gnn_only()

        # Component 2: Domain Knowledge Only
        if 'domain_only' in components:
            print("\n[2/4] Running domain expert knowledge only...")
            ablation_results['domain_only'] = self.run_domain_knowledge_only()

        # Component 3: GNN + Domain (no sparsification)
        if 'no_sparsification' in components:
            print("\n[3/4] Running GNN + domain knowledge (no sparsification)...")
            ablation_results['no_sparsification'] = self.run_gnn_without_sparsification()

        # Component 4: Full System
        if 'full_system' in components:
            print("\n[4/4] Running full system...")
            ablation_results['full_system'] = self.run_full_system()

        total_runtime = time.time() - total_start

        # Generate comparison report
        self._generate_ablation_report(ablation_results, total_runtime)

        # Save results
        results_file = self.output_dir / 'ablation_study_results.json'
        with open(results_file, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for comp, data in ablation_results.items():
                serializable_results[comp] = {
                    'summary': data.get('summary', {}),
                    'runtime_minutes': data.get('runtime_minutes', 0)
                }
            json.dump(serializable_results, f, indent=2)

        print(f"\n✅ Ablation study results saved: {results_file}")

        return ablation_results

    def _generate_ablation_report(self, results: Dict, total_runtime: float):
        """
        Generate comprehensive ablation study report

        Args:
            results: Ablation results
            total_runtime: Total runtime in seconds
        """
        print(f"\n{'='*80}")
        print("ABLATION STUDY REPORT")
        print(f"{'='*80}\n")

        # Create comparison table
        comparison_data = []

        for component, data in results.items():
            if 'error' in data:
                print(f"⚠️  {component}: Failed - {data['error']}")
                continue

            summary = data.get('summary', {})

            for target_name, target_summary in summary.get('targets', {}).items():
                best_r2 = target_summary.get('best_r2')

                comparison_data.append({
                    'Component': component,
                    'Target': target_name,
                    'Best R²': f"{best_r2:.4f}" if best_r2 is not None else 'N/A',
                    'Runtime (min)': f"{data.get('runtime_minutes', 0):.2f}"
                })

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            print("Performance Comparison:")
            print(df.to_string(index=False))

            # Save comparison table
            comparison_file = self.output_dir / 'ablation_comparison.csv'
            df.to_csv(comparison_file, index=False)
            print(f"\n✅ Comparison table saved: {comparison_file}")

            # Calculate component contributions
            print(f"\n{'='*60}")
            print("COMPONENT CONTRIBUTION ANALYSIS")
            print(f"{'='*60}\n")

            # Get full system performance
            full_system_summary = results.get('full_system', {}).get('summary', {})

            for target_name in full_system_summary.get('targets', {}).keys():
                print(f"{target_name}:")

                full_r2 = full_system_summary['targets'][target_name].get('best_r2', 0)

                # Calculate contributions
                contributions = {}

                for comp in ['gnn_only', 'domain_only', 'no_sparsification']:
                    if comp in results and 'summary' in results[comp]:
                        comp_summary = results[comp]['summary']
                        if target_name in comp_summary.get('targets', {}):
                            comp_r2 = comp_summary['targets'][target_name].get('best_r2', 0)
                            if comp_r2 is not None and full_r2 is not None:
                                contribution = full_r2 - comp_r2
                                contributions[comp] = contribution
                                print(f"  {comp}: R² improvement = {contribution:+.4f}")

                print()

        print(f"Total runtime: {total_runtime/60:.2f} minutes")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Runner for Knowledge-Guided GNN')
    parser.add_argument('--case', default='case1', choices=['case1', 'case2', 'case3', 'case4', 'case5'],
                       help='Domain expert case to analyze')
    parser.add_argument('--data_path', default='../Data/New_Data.csv',
                       help='Path to dataset')
    parser.add_argument('--output_dir', default='ablation_study_results',
                       help='Output directory for ablation results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs per configuration')
    parser.add_argument('--folds', type=int, default=5,
                       help='Cross-validation folds')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (5 epochs, 2 folds)')
    parser.add_argument('--components', nargs='+',
                       choices=['gnn_only', 'domain_only', 'no_sparsification', 'full_system'],
                       help='Specific components to run (default: all)')

    args = parser.parse_args()

    # Initialize runner
    runner = AblationStudyRunner(
        case_type=args.case,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        num_folds=args.folds,
        quick=args.quick
    )

    # Run ablation study
    results = runner.run_complete_ablation_study(components=args.components)

    print(f"\n🎉 Ablation study complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
