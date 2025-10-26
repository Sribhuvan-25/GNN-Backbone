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


# ============================================================================
# GENUS-LEVEL PROCESSING FUNCTIONS
# ============================================================================

def extract_genus_from_taxonomy(taxonomy_string):
    """
    Extract genus name from full taxonomy string.

    Args:
        taxonomy_string: Full taxonomy string with taxonomic levels

    Returns:
        str: Genus name or None if not found
    """
    if 'g__' in taxonomy_string:
        genus_part = taxonomy_string.split('g__')[1].split(';')[0].split('s__')[0]
        return genus_part.strip()
    return None


def extract_genus_from_column_name(colname):
    """
    Extract genus from OTU column name.

    Args:
        colname: Column name containing taxonomic information

    Returns:
        str: Genus name or "UnclassifiedGenus" if not found
    """
    for part in colname.split(';'):
        part = part.strip()
        if part.startswith('g__'):
            return part[3:] or "UnclassifiedGenus"
    return "UnclassifiedGenus"


def aggregate_otus_to_genera(df, otu_cols):
    """
    Aggregate OTU columns to genus level.

    Args:
        df: DataFrame containing OTU abundance data
        otu_cols: List of OTU column names

    Returns:
        tuple: (genus_aggregated_df, genus_to_columns_mapping)
    """
    # Map OTUs to genera
    col_to_genus = {c: extract_genus_from_column_name(c) for c in otu_cols}
    genus_to_cols = {}
    for c, gen in col_to_genus.items():
        if gen not in genus_to_cols:
            genus_to_cols[gen] = []
        genus_to_cols[gen].append(c)

    # Aggregate OTUs at genus level
    df_genus = pd.DataFrame({
        gen: df[cols].sum(axis=1)
        for gen, cols in genus_to_cols.items()
    }, index=df.index)

    return df_genus, genus_to_cols


def convert_to_relative_abundance_genus(df_genus):
    """
    Convert absolute abundance to relative abundance for genus data.

    Args:
        df_genus: DataFrame with genus abundance data

    Returns:
        pd.DataFrame: DataFrame with relative abundances
    """
    return df_genus.div(df_genus.sum(axis=1), axis=0)


def apply_genus_filtering(df_genus_rel, filter_mode='relaxed'):
    """
    Apply prevalence and abundance filtering to genus data.
    Uses properly calibrated thresholds to select ~100-200 genera.

    Args:
        df_genus_rel: DataFrame with relative genus abundances
        filter_mode: Filtering mode ('strict', 'relaxed', or 'permissive')

    Returns:
        tuple: (filtered_df, selected_genera_index)
    """
    presence_count = (df_genus_rel > 0).sum(axis=0)
    prevalence = presence_count / df_genus_rel.shape[0]
    mean_abund = df_genus_rel.mean(axis=0)

    # Set ULTRA-FOCUSED thresholds for genus level
    # Based on empirical testing with 833 total genera
    if filter_mode == 'strict':
        # Target: ~70 genera (highly focused on most informative)
        prevalence_threshold = 0.60  # 60% of samples
        abundance_threshold = 0.10   # 10% mean abundance
        use_intersection = False     # UNION (either criterion)
    elif filter_mode == 'relaxed':
        # Target: ~85-100 genera (balanced focus)
        prevalence_threshold = 0.55  # 55% of samples
        abundance_threshold = 0.08   # 8% mean abundance
        use_intersection = False     # UNION (either criterion)
    elif filter_mode == 'permissive':
        # Target: ~100-120 genera (moderate focus)
        prevalence_threshold = 0.50  # 50% of samples
        abundance_threshold = 0.07   # 7% mean abundance
        use_intersection = False     # UNION (either criterion)
    else:
        raise ValueError(f"Invalid filter_mode: {filter_mode}")

    high_prev = prevalence[prevalence >= prevalence_threshold].index
    high_abund = mean_abund[mean_abund >= abundance_threshold].index

    # Apply filtering logic (always UNION for genus level)
    selected_genera = high_prev.union(high_abund)

    # Ensure we don't include completely absent genera
    non_zero_genera = df_genus_rel.columns[df_genus_rel.sum(axis=0) > 0]
    selected_genera = selected_genera.intersection(non_zero_genera)

    df_genus_rel_filtered = df_genus_rel[selected_genera].copy()

    return df_genus_rel_filtered, selected_genera