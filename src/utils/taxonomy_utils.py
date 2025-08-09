"""
Taxonomy processing utilities for microbial GNN analysis.
"""
import pandas as pd


def extract_family_from_taxonomy(taxonomy_string):
    """
    Extract family name from full taxonomy string.
    
    Args:
        taxonomy_string: Full taxonomy string with taxonomic levels
        
    Returns:
        str: Family name or None if not found
    """
    if 'f__' in taxonomy_string:
        family_part = taxonomy_string.split('f__')[1].split(';')[0].split('g__')[0]
        return family_part.strip()
    return None


def extract_family_from_column_name(colname):
    """
    Extract family from OTU column name.
    
    Args:
        colname: Column name containing taxonomic information
        
    Returns:
        str: Family name or "UnclassifiedFamily" if not found
    """
    for part in colname.split(';'):
        part = part.strip()
        if part.startswith('f__'):
            return part[3:] or "UnclassifiedFamily"
    return "UnclassifiedFamily"


def aggregate_otus_to_families(df, otu_cols):
    """
    Aggregate OTU columns to family level.
    
    Args:
        df: DataFrame containing OTU abundance data
        otu_cols: List of OTU column names
        
    Returns:
        tuple: (family_aggregated_df, family_to_columns_mapping)
    """
    # Map OTUs to families
    col_to_family = {c: extract_family_from_column_name(c) for c in otu_cols}
    family_to_cols = {}
    for c, fam in col_to_family.items():
        if fam not in family_to_cols:
            family_to_cols[fam] = []
        family_to_cols[fam].append(c)
    
    # Aggregate OTUs at family level
    df_fam = pd.DataFrame({
        fam: df[cols].sum(axis=1)
        for fam, cols in family_to_cols.items()
    }, index=df.index)
    
    return df_fam, family_to_cols


def convert_to_relative_abundance(df_fam):
    """
    Convert absolute abundance to relative abundance.
    
    Args:
        df_fam: DataFrame with family abundance data
        
    Returns:
        pd.DataFrame: DataFrame with relative abundances
    """
    return df_fam.div(df_fam.sum(axis=1), axis=0)


def apply_family_filtering(df_fam_rel, filter_mode='relaxed'):
    """
    Apply prevalence and abundance filtering to family data.
    
    Args:
        df_fam_rel: DataFrame with relative family abundances
        filter_mode: Filtering mode ('strict', 'relaxed', or 'permissive')
        
    Returns:
        tuple: (filtered_df, selected_families_index)
    """
    presence_count = (df_fam_rel > 0).sum(axis=0)
    prevalence = presence_count / df_fam_rel.shape[0]
    mean_abund = df_fam_rel.mean(axis=0)
    
    # Set thresholds based on filter mode
    if filter_mode == 'strict':
        prevalence_threshold = 0.05
        abundance_threshold = 0.01
        use_intersection = True
    elif filter_mode == 'relaxed':
        prevalence_threshold = 0.02
        abundance_threshold = 0.001
        use_intersection = False
    else:  # permissive
        prevalence_threshold = 0.018
        abundance_threshold = 0.0005
        use_intersection = False
    
    high_prev = prevalence[prevalence >= prevalence_threshold].index
    high_abund = mean_abund[mean_abund >= abundance_threshold].index
    
    # Apply filtering logic
    if use_intersection:
        selected_families = high_prev.intersection(high_abund)
    else:
        selected_families = high_prev.union(high_abund)
    
    # Ensure we don't include completely absent families
    non_zero_families = df_fam_rel.columns[df_fam_rel.sum(axis=0) > 0]
    selected_families = selected_families.intersection(non_zero_families)
    
    df_fam_rel_filtered = df_fam_rel[selected_families].copy()
    
    return df_fam_rel_filtered, selected_families