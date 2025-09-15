#!/usr/bin/env python3
"""
Biological Validation Framework for Microbial Community GNN Node Pruning

This module provides biological validation of node pruning results by analyzing
the retained nodes against known microbial pathway associations, metabolic
functions, and domain expertise in anaerobic digestion systems.

Essential for publication as it demonstrates that the pruning method identifies
biologically meaningful nodes rather than statistical artifacts.

Mathematical Foundation:
========================
Biological validation uses several quantitative measures:

1. Pathway Enrichment Analysis:
   Hypergeometric test for overrepresentation of metabolic pathways
   P(X ≥ k) = Σ(i=k to min(K,n)) [(K choose i)(N-K choose n-i)] / (N choose n)
   
   Where:
   - N = total number of microbial families
   - K = families involved in pathway P
   - n = number of retained families after pruning
   - k = retained families involved in pathway P

2. Functional Coherence Score:
   FC = (1/|S|) Σ(i,j∈S) similarity(function_i, function_j)
   
   Where S is the set of retained families and similarity is based on
   shared metabolic pathways or functional annotations.

3. Domain Expert Validation:
   Quantitative assessment of retained families against expert knowledge
   using structured scoring rubrics for anaerobic digestion systems.

Biological Context:
==================
Anaerobic Digestion Pathways:
- Acetoclastic Methanogenesis: CH3COOH → CH4 + CO2
- Hydrogenotrophic Methanogenesis: 4H2 + CO2 → CH4 + 2H2O
- Acidogenesis: Complex organics → Organic acids + H2 + CO2
- Acetogenesis: Organic acids → CH3COOH + H2 + CO2

Key Microbial Groups:
- Acetogens: Acetobacterium, Clostridium, Moorella
- Methanogens: Methanosarcina, Methanosaeta, Methanobrevibacter
- Fermenters: Bacteroides, Prevotella, Ruminococcus

Authors: Research Team  
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import json
import warnings
from dataclasses import dataclass
from collections import defaultdict
import re

try:
    from scipy import stats
    from scipy.stats import hypergeom, fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available, using approximate methods")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("matplotlib/seaborn not available, skipping plots")


@dataclass
class PathwayEnrichmentResult:
    """Results of pathway enrichment analysis"""
    pathway_name: str
    total_families: int
    pathway_families: int
    retained_families: int
    retained_in_pathway: int
    p_value: float
    odds_ratio: float
    enrichment_score: float
    significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pathway_name': self.pathway_name,
            'total_families': self.total_families,
            'pathway_families': self.pathway_families,
            'retained_families': self.retained_families,
            'retained_in_pathway': self.retained_in_pathway,
            'p_value': self.p_value,
            'odds_ratio': self.odds_ratio,
            'enrichment_score': self.enrichment_score,
            'significant': self.significant,
            'expected_count': (self.retained_families * self.pathway_families) / self.total_families,
            'fold_enrichment': self.enrichment_score
        }


class BiologicalValidator:
    """
    Comprehensive biological validation framework for microbial GNN node pruning
    
    This class validates that pruned nodes represent biologically meaningful
    microbial communities rather than statistical artifacts.
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize biological validator
        
        Args:
            significance_threshold: P-value threshold for pathway enrichment
        """
        self.significance_threshold = significance_threshold
        
        # Load microbial pathway databases
        self._load_pathway_databases()
        self._load_functional_annotations()
        
        self.enrichment_results: List[PathwayEnrichmentResult] = []
        
    def _load_pathway_databases(self):
        """Load microbial pathway and functional group databases"""
        
        # Anaerobic Digestion Core Pathways
        # Based on KEGG, MetaCyc, and domain expertise
        
        self.pathway_database = {
            'Acetoclastic_Methanogenesis': {
                'description': 'Conversion of acetate to methane and CO2',
                'key_reaction': 'CH3COOH → CH4 + CO2',
                'families': [
                    'Methanosarcinaceae', 'Methanosaetaceae', 'Methanotrichaceae',
                    'Acetobacteraceae', 'Clostridium', 'Moorella'
                ],
                'essential_genera': ['Methanosarcina', 'Methanosaeta'],
                'pathway_id': 'map00680'
            },
            
            'Hydrogenotrophic_Methanogenesis': {
                'description': 'Conversion of H2/CO2 to methane',
                'key_reaction': '4H2 + CO2 → CH4 + 2H2O',
                'families': [
                    'Methanobacteriaceae', 'Methanococcaceae', 'Methanomicrobiaceae',
                    'Methanocellaceae', 'Methanobrevibacter', 'Methanothermus'
                ],
                'essential_genera': ['Methanobrevibacter', 'Methanobacterium'],
                'pathway_id': 'map00680'
            },
            
            'Acidogenesis': {
                'description': 'Breakdown of complex organics to organic acids',
                'key_reaction': 'Complex organics → VFA + H2 + CO2',
                'families': [
                    'Bacteroidaceae', 'Prevotellaceae', 'Ruminococcaceae',
                    'Lachnospiraceae', 'Clostridiaceae', 'Peptostreptococcaceae',
                    'Veillonellaceae', 'Acidaminococcaceae'
                ],
                'essential_genera': ['Bacteroides', 'Prevotella', 'Clostridium'],
                'pathway_id': 'map00010'
            },
            
            'Acetogenesis': {
                'description': 'Conversion of organic acids to acetate',
                'key_reaction': 'Propionate/Butyrate → Acetate + H2',
                'families': [
                    'Syntrophomonadaceae', 'Syntrophobacteraceae', 
                    'Pelotomaculaceae', 'Syntrophaceae',
                    'Clostridiaceae', 'Acetobacteraceae'
                ],
                'essential_genera': ['Syntrophomonas', 'Syntrophobacter', 'Pelotomaculum'],
                'pathway_id': 'map00640'
            },
            
            'Homoacetogenesis': {
                'description': 'CO2 fixation to acetate via Wood-Ljungdahl pathway',
                'key_reaction': '2CO2 + 4H2 → CH3COOH + 2H2O',
                'families': [
                    'Acetobacteraceae', 'Eubacteriaceae', 'Clostridiaceae',
                    'Thermoanaerobacteraceae', 'Moorella', 'Sporomusa'
                ],
                'essential_genera': ['Acetobacterium', 'Moorella', 'Sporomusa'],
                'pathway_id': 'map00720'
            },
            
            'Sulfate_Reduction': {
                'description': 'Anaerobic respiration using sulfate as electron acceptor',
                'key_reaction': 'SO4²⁻ + 8H⁺ + 8e⁻ → S²⁻ + 4H2O',
                'families': [
                    'Desulfovibrionaceae', 'Desulfobacteraceae', 'Desulfobulbaceae',
                    'Desulfotomaculum', 'Desulfohalobiaceae'
                ],
                'essential_genera': ['Desulfovibrio', 'Desulfobacter'],
                'pathway_id': 'map00920'
            },
            
            'Nitrate_Reduction': {
                'description': 'Anaerobic respiration using nitrate/nitrite',
                'key_reaction': 'NO3⁻ → NO2⁻ → N2O → N2',
                'families': [
                    'Pseudomonadaceae', 'Enterobacteriaceae', 'Paracoccaceae',
                    'Rhodobacteraceae', 'Dechloromonas'
                ],
                'essential_genera': ['Pseudomonas', 'Paracoccus'],
                'pathway_id': 'map00910'
            },
            
            'Iron_Reduction': {
                'description': 'Anaerobic respiration using iron as electron acceptor',
                'key_reaction': 'Fe³⁺ + e⁻ → Fe²⁺',
                'families': [
                    'Geobacteraceae', 'Shewanellaceae', 'Desulfuromonadaceae',
                    'Pelobacteraceae', 'Geothermobacter'
                ],
                'essential_genera': ['Geobacter', 'Shewanella'],
                'pathway_id': 'map00190'
            }
        }
        
        # Functional group classifications
        self.functional_groups = {
            'Primary_Methanogens': [
                'Methanosarcinaceae', 'Methanosaetaceae', 'Methanobacteriaceae',
                'Methanococcaceae', 'Methanomicrobiaceae'
            ],
            
            'Syntrophic_Bacteria': [
                'Syntrophomonadaceae', 'Syntrophobacteraceae', 
                'Pelotomaculaceae', 'Syntrophaceae'
            ],
            
            'Fermentative_Bacteria': [
                'Bacteroidaceae', 'Prevotellaceae', 'Ruminococcaceae',
                'Lachnospiraceae', 'Clostridiaceae'
            ],
            
            'Sulfate_Reducers': [
                'Desulfovibrionaceae', 'Desulfobacteraceae', 'Desulfobulbaceae'
            ],
            
            'Acetogens': [
                'Acetobacteraceae', 'Eubacteriaceae', 'Moorella'
            ]
        }
        
    def _load_functional_annotations(self):
        """Load functional annotations for microbial families"""
        
        # Metabolic capabilities matrix
        # Each family gets scores (0-1) for various metabolic functions
        
        self.functional_matrix = {
            'Methanosarcinaceae': {
                'methane_production': 1.0, 'acetate_utilization': 1.0,
                'h2_utilization': 0.8, 'co2_fixation': 0.6,
                'anaerobic_respiration': 0.0, 'fermentation': 0.0
            },
            'Methanosaetaceae': {
                'methane_production': 1.0, 'acetate_utilization': 1.0,
                'h2_utilization': 0.2, 'co2_fixation': 0.3,
                'anaerobic_respiration': 0.0, 'fermentation': 0.0
            },
            'Methanobacteriaceae': {
                'methane_production': 1.0, 'acetate_utilization': 0.1,
                'h2_utilization': 1.0, 'co2_fixation': 1.0,
                'anaerobic_respiration': 0.0, 'fermentation': 0.0
            },
            'Bacteroidaceae': {
                'methane_production': 0.0, 'acetate_utilization': 0.3,
                'h2_utilization': 0.2, 'co2_fixation': 0.1,
                'anaerobic_respiration': 0.4, 'fermentation': 1.0
            },
            'Prevotellaceae': {
                'methane_production': 0.0, 'acetate_utilization': 0.4,
                'h2_utilization': 0.3, 'co2_fixation': 0.2,
                'anaerobic_respiration': 0.3, 'fermentation': 1.0
            },
            'Ruminococcaceae': {
                'methane_production': 0.0, 'acetate_utilization': 0.5,
                'h2_utilization': 0.6, 'co2_fixation': 0.3,
                'anaerobic_respiration': 0.2, 'fermentation': 1.0
            },
            'Clostridiaceae': {
                'methane_production': 0.0, 'acetate_utilization': 0.7,
                'h2_utilization': 0.8, 'co2_fixation': 0.6,
                'anaerobic_respiration': 0.3, 'fermentation': 1.0
            },
            'Syntrophomonadaceae': {
                'methane_production': 0.0, 'acetate_utilization': 0.3,
                'h2_utilization': 0.2, 'co2_fixation': 0.2,
                'anaerobic_respiration': 0.8, 'fermentation': 0.6
            },
            'Desulfovibrionaceae': {
                'methane_production': 0.0, 'acetate_utilization': 0.6,
                'h2_utilization': 0.8, 'co2_fixation': 0.3,
                'anaerobic_respiration': 1.0, 'fermentation': 0.2
            }
        }
        
    def validate_pruning_results(self, 
                                retained_families: List[str],
                                all_families: List[str],
                                target_pathway: str = 'all') -> Dict[str, Any]:
        """
        Comprehensive biological validation of node pruning results
        
        Args:
            retained_families: List of microbial families retained after pruning
            all_families: List of all microbial families in original dataset  
            target_pathway: Specific pathway to focus on ('all' for comprehensive)
            
        Returns:
            Dictionary with comprehensive biological validation results
        """
        
        print("="*80)
        print("BIOLOGICAL VALIDATION OF NODE PRUNING RESULTS")
        print("="*80)
        
        # Clean and standardize family names
        retained_families = self._standardize_family_names(retained_families)
        all_families = self._standardize_family_names(all_families)
        
        print(f"\nDataset Overview:")
        print(f"  Total families in dataset: {len(all_families)}")
        print(f"  Families retained after pruning: {len(retained_families)}")
        print(f"  Retention rate: {len(retained_families)/len(all_families)*100:.1f}%")
        
        # 1. Pathway enrichment analysis
        print(f"\n1. Pathway Enrichment Analysis:")
        enrichment_results = self._perform_pathway_enrichment(retained_families, all_families, target_pathway)
        
        # 2. Functional coherence analysis
        print(f"\n2. Functional Coherence Analysis:")
        coherence_score = self._calculate_functional_coherence(retained_families)
        
        # 3. Domain expert validation
        print(f"\n3. Domain Expert Validation:")
        expert_score = self._domain_expert_validation(retained_families, target_pathway)
        
        # 4. Pathway completeness analysis
        print(f"\n4. Pathway Completeness Analysis:")
        completeness_analysis = self._analyze_pathway_completeness(retained_families, target_pathway)
        
        # 5. Network topology validation
        print(f"\n5. Network Topology Validation:")
        topology_validation = self._validate_network_topology(retained_families)
        
        # Generate comprehensive report
        validation_report = {
            'dataset_overview': {
                'total_families': len(all_families),
                'retained_families': len(retained_families),
                'retention_rate': len(retained_families)/len(all_families),
                'families_retained': retained_families,
                'families_removed': list(set(all_families) - set(retained_families))
            },
            'pathway_enrichment': {
                'results': [r.to_dict() for r in enrichment_results],
                'significant_pathways': len([r for r in enrichment_results if r.significant]),
                'total_pathways_tested': len(enrichment_results)
            },
            'functional_coherence': coherence_score,
            'domain_expert_validation': expert_score,
            'pathway_completeness': completeness_analysis,
            'network_topology': topology_validation,
            'overall_biological_validity': self._calculate_overall_validity_score(
                enrichment_results, coherence_score, expert_score, completeness_analysis
            )
        }
        
        # Print summary
        self._print_validation_summary(validation_report)
        
        return validation_report
    
    def _standardize_family_names(self, family_names: List[str]) -> List[str]:
        """Standardize microbial family names to match database"""
        standardized = []
        
        for name in family_names:
            # Remove common prefixes/suffixes and standardize
            name = str(name).strip()
            
            # Handle common naming variations
            if name.endswith('_family') or name.endswith('_Family'):
                name = name[:-7]
            
            if not name.endswith('aceae') and not name.endswith('aceae'):
                # Try to match against known families
                matches = [f for f in self.pathway_database.get('Acidogenesis', {}).get('families', []) 
                          if name.lower() in f.lower() or f.lower() in name.lower()]
                if matches:
                    name = matches[0]
                else:
                    # Add standard suffix if missing
                    name = name + 'aceae'
            
            standardized.append(name)
        
        return standardized
    
    def _perform_pathway_enrichment(self, 
                                  retained_families: List[str],
                                  all_families: List[str],
                                  target_pathway: str) -> List[PathwayEnrichmentResult]:
        """Perform hypergeometric test for pathway enrichment"""
        
        enrichment_results = []
        
        pathways_to_test = [target_pathway] if target_pathway != 'all' else list(self.pathway_database.keys())
        
        for pathway_name in pathways_to_test:
            if pathway_name not in self.pathway_database:
                continue
                
            pathway_info = self.pathway_database[pathway_name]
            pathway_families = set(pathway_info['families'])
            
            # Count overlaps
            total_families = len(all_families)
            families_in_pathway = len([f for f in all_families if f in pathway_families])
            retained_count = len(retained_families)
            retained_in_pathway = len([f for f in retained_families if f in pathway_families])
            
            # Skip if no families from this pathway in dataset
            if families_in_pathway == 0:
                continue
                
            print(f"   {pathway_name}:")
            print(f"     Pathway families in dataset: {families_in_pathway}/{len(pathway_families)}")
            print(f"     Retained families in pathway: {retained_in_pathway}/{families_in_pathway}")
            
            # Hypergeometric test
            if HAS_SCIPY:
                # P(X >= k) where X ~ Hypergeom(N, K, n)
                p_value = hypergeom.sf(retained_in_pathway - 1, total_families, 
                                     families_in_pathway, retained_count)
                
                # Fisher's exact test for odds ratio
                contingency_table = [
                    [retained_in_pathway, retained_count - retained_in_pathway],
                    [families_in_pathway - retained_in_pathway, 
                     total_families - families_in_pathway - (retained_count - retained_in_pathway)]
                ]
                odds_ratio, fisher_p = fisher_exact(contingency_table)
                
            else:
                # Approximate p-value using normal approximation
                expected = (retained_count * families_in_pathway) / total_families
                variance = expected * (1 - families_in_pathway/total_families) * \
                          (total_families - retained_count) / (total_families - 1)
                
                if variance > 0:
                    z_score = (retained_in_pathway - expected) / np.sqrt(variance)
                    p_value = 1 - stats.norm.cdf(z_score) if z_score > 0 else stats.norm.cdf(z_score)
                else:
                    p_value = 1.0
                
                # Approximate odds ratio
                odds_ratio = (retained_in_pathway * (total_families - families_in_pathway)) / \
                           max(1, (families_in_pathway - retained_in_pathway) * (retained_count - retained_in_pathway))
            
            # Enrichment score (fold change)
            expected_count = (retained_count * families_in_pathway) / total_families
            enrichment_score = retained_in_pathway / max(0.1, expected_count)
            
            significant = p_value < self.significance_threshold
            
            result = PathwayEnrichmentResult(
                pathway_name=pathway_name,
                total_families=total_families,
                pathway_families=families_in_pathway,
                retained_families=retained_count,
                retained_in_pathway=retained_in_pathway,
                p_value=p_value,
                odds_ratio=odds_ratio,
                enrichment_score=enrichment_score,
                significant=significant
            )
            
            enrichment_results.append(result)
            
            # Print result
            significance_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            enrichment_desc = "Enriched" if enrichment_score > 1 else "Depleted"
            
            print(f"     {enrichment_desc}: {enrichment_score:.2f}x (p = {p_value:.4f}){significance_marker}")
            
        return enrichment_results
    
    def _calculate_functional_coherence(self, retained_families: List[str]) -> Dict[str, float]:
        """Calculate functional coherence of retained families"""
        
        coherence_scores = {}
        
        # Get functional profiles for retained families
        retained_profiles = {}
        for family in retained_families:
            if family in self.functional_matrix:
                retained_profiles[family] = self.functional_matrix[family]
        
        if len(retained_profiles) < 2:
            return {'overall_coherence': 0.0, 'functional_profiles': retained_profiles}
        
        # Calculate pairwise functional similarity
        function_names = list(list(retained_profiles.values())[0].keys())
        similarities = []
        
        for func in function_names:
            func_values = [profile[func] for profile in retained_profiles.values()]
            
            # Calculate coefficient of variation (lower = more coherent)
            if np.mean(func_values) > 0:
                cv = np.std(func_values) / np.mean(func_values)
                coherence = max(0, 1 - cv)  # Higher coherence = lower variation
            else:
                coherence = 1.0 if all(v == 0 for v in func_values) else 0.0
            
            coherence_scores[func] = coherence
            similarities.append(coherence)
        
        overall_coherence = np.mean(similarities) if similarities else 0.0
        
        print(f"   Overall functional coherence: {overall_coherence:.3f}")
        print(f"   Function-specific coherence:")
        for func, score in coherence_scores.items():
            print(f"     {func}: {score:.3f}")
        
        return {
            'overall_coherence': overall_coherence,
            'function_specific': coherence_scores,
            'retained_profiles': retained_profiles
        }
    
    def _domain_expert_validation(self, retained_families: List[str], target_pathway: str) -> Dict[str, Any]:
        """Domain expert validation based on anaerobic digestion knowledge"""
        
        expert_scores = {}
        
        # Essential families for different pathways
        essential_families = {
            'Acetoclastic_Methanogenesis': ['Methanosarcinaceae', 'Methanosaetaceae'],
            'Hydrogenotrophic_Methanogenesis': ['Methanobacteriaceae', 'Methanococcaceae'],
            'Acidogenesis': ['Bacteroidaceae', 'Prevotellaceae', 'Ruminococcaceae'],
            'Acetogenesis': ['Syntrophomonadaceae', 'Syntrophobacteraceae'],
            'all': ['Methanosarcinaceae', 'Methanosaetaceae', 'Methanobacteriaceae', 
                   'Bacteroidaceae', 'Syntrophomonadaceae']
        }
        
        pathways_to_check = [target_pathway] if target_pathway != 'all' else ['all']
        
        for pathway in pathways_to_check:
            if pathway not in essential_families:
                continue
                
            essential = essential_families[pathway]
            retained_essential = [f for f in retained_families if f in essential]
            
            # Expert scoring rubric
            completeness = len(retained_essential) / len(essential)
            
            # Bonus for including key syntrophic pairs
            syntrophic_bonus = 0.0
            if any('Syntropho' in f for f in retained_families) and \
               any('Methan' in f for f in retained_families):
                syntrophic_bonus = 0.2
            
            # Penalty for missing critical methanogens
            methanogen_penalty = 0.0
            methanogens_retained = [f for f in retained_families if 'Methan' in f]
            if len(methanogens_retained) == 0:
                methanogen_penalty = -0.5
            
            expert_score = min(1.0, max(0.0, completeness + syntrophic_bonus + methanogen_penalty))
            
            expert_scores[pathway] = {
                'completeness_score': completeness,
                'syntrophic_bonus': syntrophic_bonus,
                'methanogen_penalty': methanogen_penalty,
                'final_score': expert_score,
                'essential_retained': retained_essential,
                'essential_missing': [f for f in essential if f not in retained_families]
            }
            
            print(f"   {pathway} Expert Score: {expert_score:.3f}")
            print(f"     Essential families retained: {len(retained_essential)}/{len(essential)}")
            if retained_essential:
                print(f"     Key families present: {', '.join(retained_essential)}")
            if expert_scores[pathway]['essential_missing']:
                print(f"     ⚠️  Missing essential: {', '.join(expert_scores[pathway]['essential_missing'])}")
        
        return expert_scores
    
    def _analyze_pathway_completeness(self, retained_families: List[str], target_pathway: str) -> Dict[str, Any]:
        """Analyze completeness of metabolic pathways"""
        
        completeness_analysis = {}
        
        pathways_to_analyze = [target_pathway] if target_pathway != 'all' else list(self.pathway_database.keys())
        
        for pathway_name in pathways_to_analyze:
            if pathway_name not in self.pathway_database:
                continue
                
            pathway_info = self.pathway_database[pathway_name]
            pathway_families = set(pathway_info['families'])
            essential_genera = set(pathway_info.get('essential_genera', []))
            
            # Count coverage
            retained_in_pathway = [f for f in retained_families if f in pathway_families]
            
            # Check for essential genera (approximate matching)
            essential_coverage = 0
            for genus in essential_genera:
                if any(genus.lower() in f.lower() for f in retained_families):
                    essential_coverage += 1
            
            pathway_completeness = len(retained_in_pathway) / len(pathway_families) if pathway_families else 0
            essential_completeness = essential_coverage / len(essential_genera) if essential_genera else 1
            
            # Overall pathway viability score
            viability_score = 0.3 * pathway_completeness + 0.7 * essential_completeness
            
            completeness_analysis[pathway_name] = {
                'pathway_completeness': pathway_completeness,
                'essential_completeness': essential_completeness,
                'viability_score': viability_score,
                'retained_families': retained_in_pathway,
                'missing_families': list(pathway_families - set(retained_families)),
                'pathway_viable': viability_score > 0.5
            }
        
        print(f"   Pathway viability analysis:")
        for pathway, analysis in completeness_analysis.items():
            viable = "✅" if analysis['pathway_viable'] else "❌"
            print(f"     {pathway}: {analysis['viability_score']:.3f} {viable}")
        
        return completeness_analysis
    
    def _validate_network_topology(self, retained_families: List[str]) -> Dict[str, Any]:
        """Validate network topology of retained families"""
        
        # Analyze functional group representation
        group_representation = {}
        
        for group_name, group_families in self.functional_groups.items():
            retained_in_group = [f for f in retained_families if f in group_families]
            representation = len(retained_in_group) / len(group_families) if group_families else 0
            
            group_representation[group_name] = {
                'representation': representation,
                'retained_count': len(retained_in_group),
                'total_count': len(group_families),
                'retained_families': retained_in_group
            }
        
        # Network balance score (all functional groups should be represented)
        non_zero_groups = sum(1 for g in group_representation.values() if g['representation'] > 0)
        balance_score = non_zero_groups / len(self.functional_groups)
        
        topology_validation = {
            'functional_group_representation': group_representation,
            'network_balance_score': balance_score,
            'well_balanced': balance_score > 0.6,
            'dominant_groups': [name for name, info in group_representation.items() 
                              if info['representation'] > 0.5],
            'missing_groups': [name for name, info in group_representation.items() 
                             if info['representation'] == 0]
        }
        
        print(f"   Network balance score: {balance_score:.3f}")
        if topology_validation['missing_groups']:
            print(f"   ⚠️  Missing functional groups: {', '.join(topology_validation['missing_groups'])}")
        
        return topology_validation
    
    def _calculate_overall_validity_score(self, 
                                        enrichment_results: List[PathwayEnrichmentResult],
                                        coherence_score: Dict[str, float],
                                        expert_score: Dict[str, Any],
                                        completeness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall biological validity score"""
        
        # Component scores (0-1 scale)
        enrichment_score = np.mean([1.0 if r.significant and r.enrichment_score > 1 else 0.5 
                                   for r in enrichment_results]) if enrichment_results else 0.5
        
        coherence_component = coherence_score.get('overall_coherence', 0.0)
        
        expert_component = np.mean([score['final_score'] for score in expert_score.values()]) if expert_score else 0.5
        
        completeness_component = np.mean([analysis['viability_score'] 
                                        for analysis in completeness_analysis.values()]) if completeness_analysis else 0.5
        
        # Weighted overall score
        weights = {'enrichment': 0.3, 'coherence': 0.2, 'expert': 0.3, 'completeness': 0.2}
        
        overall_score = (
            weights['enrichment'] * enrichment_score +
            weights['coherence'] * coherence_component +
            weights['expert'] * expert_component +
            weights['completeness'] * completeness_component
        )
        
        # Qualitative assessment
        if overall_score >= 0.8:
            validity_level = "Excellent"
            recommendation = "Highly biologically valid - suitable for publication"
        elif overall_score >= 0.6:
            validity_level = "Good"
            recommendation = "Biologically valid with minor concerns"
        elif overall_score >= 0.4:
            validity_level = "Moderate"
            recommendation = "Some biological validity concerns - review methodology"
        else:
            validity_level = "Poor"
            recommendation = "Significant biological validity issues - major revision needed"
        
        return {
            'overall_score': overall_score,
            'validity_level': validity_level,
            'recommendation': recommendation,
            'component_scores': {
                'enrichment': enrichment_score,
                'coherence': coherence_component,
                'expert': expert_component,
                'completeness': completeness_component
            },
            'weights': weights
        }
    
    def _print_validation_summary(self, validation_report: Dict[str, Any]):
        """Print comprehensive validation summary"""
        
        print(f"\n{'='*80}")
        print("BIOLOGICAL VALIDATION SUMMARY")
        print(f"{'='*80}")
        
        overall = validation_report['overall_biological_validity']
        
        print(f"\nOverall Biological Validity: {overall['overall_score']:.3f} ({overall['validity_level']})")
        print(f"Recommendation: {overall['recommendation']}")
        
        print(f"\nComponent Scores:")
        for component, score in overall['component_scores'].items():
            print(f"  {component.capitalize()}: {score:.3f}")
        
        enrichment = validation_report['pathway_enrichment']
        if enrichment['significant_pathways'] > 0:
            print(f"\n✅ Significant pathway enrichments: {enrichment['significant_pathways']}")
        
        completeness = validation_report['pathway_completeness']
        viable_pathways = sum(1 for p in completeness.values() if p['pathway_viable'])
        if viable_pathways > 0:
            print(f"✅ Viable metabolic pathways: {viable_pathways}")
        
        topology = validation_report['network_topology']
        if topology['well_balanced']:
            print(f"✅ Well-balanced functional network")
        else:
            print(f"⚠️  Network imbalance detected")
        
        print(f"\n🔬 Biological validation complete!")
    
    def save_results(self, validation_report: Dict[str, Any], output_path: str):
        """Save validation results to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        print(f"\n✅ Biological validation results saved to {output_path}")


def main():
    """Test biological validation framework"""
    print("Testing Biological Validation Framework")
    print("="*50)
    
    # Example usage with sample data
    validator = BiologicalValidator()
    
    # Sample retained families (after node pruning)
    retained_families = [
        'Methanosarcinaceae', 'Methanosaetaceae', 'Bacteroidaceae',
        'Ruminococcaceae', 'Syntrophomonadaceae', 'Clostridiaceae',
        'Prevotellaceae'
    ]
    
    # Sample full dataset
    all_families = retained_families + [
        'Lachnospiraceae', 'Veillonellaceae', 'Desulfovibrionaceae',
        'Methanobacteriaceae', 'Acetobacteraceae', 'Peptostreptococcaceae'
    ]
    
    print("Sample validation with acetoclastic methanogenesis focus:")
    results = validator.validate_pruning_results(
        retained_families=retained_families,
        all_families=all_families,
        target_pathway='Acetoclastic_Methanogenesis'
    )
    
    print(f"\n✅ Biological validation framework complete!")
    print(f"Overall validity score: {results['overall_biological_validity']['overall_score']:.3f}")


if __name__ == "__main__":
    main()