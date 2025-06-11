"""
Adaptive Microbial GNN Package
==============================

Non-temporal dynamic GNN for microbial interaction network learning.

Main components:
- AdaptiveMicrobialGNN: Core model with adaptive graph structure learning
- AdaptiveMicrobialPipeline: Complete training and analysis pipeline
- BiologicalConstraints: Helper for biological interpretability

Usage:
    from adaptive_gnn import AdaptiveMicrobialPipeline
    
    pipeline = AdaptiveMicrobialPipeline()
    pipeline.load_data("data.csv")
    results = pipeline.run_complete_pipeline()
"""

from .models.adaptive_microbial_gnn import (
    AdaptiveMicrobialGNN,
    AdaptiveGraphLearner, 
    DynamicEdgeAttention,
    BiologicalConstraints
)

from .adaptive_microbial_pipeline import AdaptiveMicrobialPipeline

__version__ = "1.0.0"
__author__ = "Adaptive GNN Research Team"

__all__ = [
    'AdaptiveMicrobialGNN',
    'AdaptiveGraphLearner',
    'DynamicEdgeAttention', 
    'BiologicalConstraints',
    'AdaptiveMicrobialPipeline'
] 