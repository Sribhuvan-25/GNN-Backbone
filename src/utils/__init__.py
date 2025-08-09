"""
Utilities package for GNN pipeline components.
"""

from .taxonomy_utils import (
    extract_family_from_taxonomy,
    extract_family_from_column_name,
    aggregate_otus_to_families,
    convert_to_relative_abundance,
    apply_family_filtering
)

from .visualization_utils import (
    adjust_color_brightness,
    get_vibrant_node_colors,
    get_functional_group_colors,
    create_networkx_graph_from_edge_data,
    create_legend_for_functional_groups,
    save_graph_visualization,
    format_statistics_with_std,
    create_performance_comparison_plot
)

from .result_management import (
    create_results_directory_structure,
    save_fold_results,
    convert_for_json_serialization,
    save_embeddings,
    save_model_checkpoint,
    aggregate_fold_results,
    save_hyperparameter_tracking,
    save_combined_results_summary,
    load_fold_results,
    save_graph_metadata,
    create_experiment_log
)

__all__ = [
    # Taxonomy utilities
    'extract_family_from_taxonomy',
    'extract_family_from_column_name',
    'aggregate_otus_to_families',
    'convert_to_relative_abundance',
    'apply_family_filtering',
    
    # Visualization utilities
    'adjust_color_brightness',
    'get_vibrant_node_colors',
    'get_functional_group_colors',
    'create_networkx_graph_from_edge_data',
    'create_legend_for_functional_groups',
    'save_graph_visualization',
    'format_statistics_with_std',
    'create_performance_comparison_plot',
    
    # Result management utilities
    'create_results_directory_structure',
    'save_fold_results',
    'convert_for_json_serialization',
    'save_embeddings',
    'save_model_checkpoint',
    'aggregate_fold_results',
    'save_hyperparameter_tracking',
    'save_combined_results_summary',
    'load_fold_results',
    'save_graph_metadata',
    'create_experiment_log',
]