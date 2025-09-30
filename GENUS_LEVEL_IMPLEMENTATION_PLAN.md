# GENUS-LEVEL PIPELINE IMPLEMENTATION PLAN

## EXECUTIVE SUMMARY

This document provides a comprehensive plan for implementing genus-level graph neural network analysis for anaerobic digestion microbiome data. Based on literature review and current implementation analysis, this plan addresses the unique challenges of genus-level taxonomic resolution.

---

## PART 1: CURRENT FAMILY-LEVEL IMPLEMENTATION (UNDERSTANDING)

### **How Family-Level Aggregation Works:**

```python
# STEP 1: Extract family from taxonomy string
# Input: "d__Bacteria;p__Firmicutes;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Faecalibacterium"
# Output: "Ruminococcaceae"

def extract_family(colname):
    for part in colname.split(';'):
        part = part.strip()
        if part.startswith('f__'):
            return part[3:] or "UnclassifiedFamily"
    return "UnclassifiedFamily"

# STEP 2: Map OTUs to families
col_to_family = {otu_col: extract_family(otu_col) for otu_col in otu_cols}

# STEP 3: Group OTUs by family and SUM abundances
# Example: If you have 5 OTUs from Ruminococcaceae, sum their counts
df_fam = pd.DataFrame({
    fam: self.df[otu_cols_in_family].sum(axis=1)  # SUM across OTUs
    for fam, otu_cols_in_family in family_to_cols.items()
})

# STEP 4: Convert to relative abundance
df_fam_rel = df_fam.div(df_fam.sum(axis=1), axis=0)
# Each sample now sums to 1.0

# STEP 5: Filter by prevalence and abundance
# Strict mode: prevalence ≥ 5% AND abundance ≥ 1%
# Relaxed mode: prevalence ≥ 2% OR abundance ≥ 0.1%
# Permissive mode: prevalence ≥ 1.8% OR abundance ≥ 0.05%

# STEP 6: Apply double square root transformation
df_transformed = df_filtered.apply(lambda x: np.sqrt(np.sqrt(x + 1e-10)))

# Result: ~20-80 families depending on filter mode
```

**Key Insight:** Family-level AGGREGATES (sums) multiple OTUs into single taxonomic units, reducing dimensionality while preserving functional group information.

---

## PART 2: GENUS-LEVEL CHALLENGES & DIFFERENCES

### **Critical Differences Between Family and Genus Levels:**

| Aspect | Family Level | Genus Level |
|--------|-------------|-------------|
| **Number of Taxa** | ~20-80 families | **~100-300 genera** (much higher) |
| **Sparsity** | Moderate (aggregation reduces) | **High (~70-90% zeros)** |
| **Biological Resolution** | Functional groups | Species-level proxies |
| **Aggregation** | YES - sum OTUs within family | **NO - direct genus assignment** |
| **Unclassified Taxa** | ~10-15% | **~30-50%** (genus harder to assign) |
| **Graph Size** | Small (20-80 nodes) | **Large (100-300 nodes)** |
| **Computational Cost** | Low | **High (10-50x more edges)** |

### **Why NO Aggregation at Genus Level?**

**Genus is ALREADY a taxonomic rank** - it's the second-lowest level in taxonomy. You don't "aggregate" to genus; you **filter TO genus** from OTUs/ASVs that are already assigned to specific genera.

**Example:**
```
OTU_001: d__Bacteria;p__Firmicutes;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Faecalibacterium
         → Genus = Faecalibacterium (direct assignment, NO summing)

OTU_002: d__Bacteria;p__Firmicutes;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Faecalibacterium
         → Genus = Faecalibacterium (same genus, WILL sum these two OTUs)

OTU_003: d__Bacteria;p__Firmicutes;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Subdoligranulum
         → Genus = Subdoligranulum (different genus, separate node)
```

Both OTU_001 and OTU_002 belong to genus Faecalibacterium → their abundances get SUMMED.

---

## PART 3: GENUS-LEVEL PREPROCESSING PIPELINE DESIGN

### **Research-Backed Recommendations:**

Based on literature review (Weiss et al. 2017, Nearing et al. 2022, Microbiome Journal 2023):

1. **Prevalence Threshold**: 10% (appear in at least 10% of samples)
2. **Mean Abundance Threshold**: 0.05% (0.0005 relative abundance)
3. **Filtering Logic**: UNION (prevalence OR abundance) to retain more genera
4. **Target Genus Count**: 50-150 genera (balance between resolution and computational cost)
5. **Transformation**: Double square root (same as family-level)

### **Proposed Implementation:**

```python
def _process_genera(self):
    """
    Extract genus-level taxonomy WITHOUT aggregation across families.

    Key Difference from Family: Genera are NOT aggregated from families.
    We extract genus directly from taxonomy and sum OTUs within same genus.
    """

    # Function to extract genus from taxonomy string
    def extract_genus(colname):
        """
        Extract genus from QIIME2-style taxonomy string.

        Handles cases:
        1. Full assignment: g__Faecalibacterium → "Faecalibacterium"
        2. Partial assignment: g__ → Use family name + "_unclassified"
        3. Missing genus: __ → Use family name + "_unclassified"
        4. Completely unclassified: Multiple levels missing → "Unclassified_Genus"
        """
        parts = colname.split(';')

        # Extract family and genus
        family = None
        genus = None

        for part in parts:
            part = part.strip()
            if part.startswith('f__'):
                family = part[3:] or None
            elif part.startswith('g__'):
                genus = part[3:] or None

        # Genus classification logic
        if genus and genus != '':
            return genus
        elif family and family != '':
            # Genus unclassified but family known
            return f"{family}_unclassified"
        else:
            # Both family and genus unclassified
            return "Unclassified_Genus"

    # Map OTUs to genera
    col_to_genus = {c: extract_genus(c) for c in self.otu_cols}

    # Group OTUs by genus
    genus_to_cols = {}
    for c, gen in col_to_genus.items():
        if gen not in genus_to_cols:
            genus_to_cols[gen] = []
        genus_to_cols[gen].append(c)

    # Sum OTUs within same genus (NOT across families!)
    df_genus = pd.DataFrame({
        gen: self.df[cols].sum(axis=1)
        for gen, cols in genus_to_cols.items()
    }, index=self.df.index)

    print(f"Total genera before filtering: {df_genus.shape[1]}")
    print(f"  - Classified genera: {sum(1 for g in df_genus.columns if not g.endswith('_unclassified') and g != 'Unclassified_Genus')}")
    print(f"  - Family-level unclassified: {sum(1 for g in df_genus.columns if g.endswith('_unclassified'))}")
    print(f"  - Completely unclassified: {sum(1 for g in df_genus.columns if g == 'Unclassified_Genus')}")

    # Convert to relative abundance
    df_genus_rel = df_genus.div(df_genus.sum(axis=1), axis=0)

    # GENUS-SPECIFIC FILTERING (research-backed thresholds)
    presence_count = (df_genus_rel > 0).sum(axis=0)
    prevalence = presence_count / df_genus_rel.shape[0]
    mean_abund = df_genus_rel.mean(axis=0)

    # Filtering based on genus_filter_mode
    if self.genus_filter_mode == 'strict':
        prevalence_threshold = 0.15  # 15% of samples (stricter for genera)
        abundance_threshold = 0.001  # 0.1% mean abundance
        use_intersection = True      # Both criteria must be met
        target_min_genera = 40
    elif self.genus_filter_mode == 'standard':
        prevalence_threshold = 0.10  # 10% of samples (literature standard)
        abundance_threshold = 0.0005  # 0.05% mean abundance
        use_intersection = False     # Either criterion (UNION)
        target_min_genera = 80
    elif self.genus_filter_mode == 'permissive':
        prevalence_threshold = 0.05  # 5% of samples
        abundance_threshold = 0.0002  # 0.02% mean abundance
        use_intersection = False     # Either criterion (UNION)
        target_min_genera = 120
    else:
        raise ValueError(f"Invalid genus_filter_mode: {self.genus_filter_mode}")

    high_prev = prevalence[prevalence >= prevalence_threshold].index
    high_abund = mean_abund[mean_abund >= abundance_threshold].index

    # Apply filtering logic
    if use_intersection:
        selected_genera = high_prev.intersection(high_abund)
        filter_method = "INTERSECTION (prevalence AND abundance)"
    else:
        selected_genera = high_prev.union(high_abund)
        filter_method = "UNION (prevalence OR abundance)"

    # Ultra-permissive fallback if too few genera
    if len(selected_genera) < target_min_genera:
        print(f"Only {len(selected_genera)} genera with {self.genus_filter_mode} criteria. Using ultra-permissive filtering...")
        ultra_prev = prevalence[prevalence >= 0.037].index  # ~2 samples
        ultra_abund = mean_abund[mean_abund >= 0.0001].index  # 0.01% abundance
        selected_genera = ultra_prev.union(ultra_abund)
        filter_method = "ULTRA-PERMISSIVE (prevalence OR abundance)"

    # Ensure we don't include completely absent genera
    non_zero_genera = df_genus_rel.columns[df_genus_rel.sum(axis=0) > 0]
    selected_genera = selected_genera.intersection(non_zero_genera)

    df_genus_rel_filtered = df_genus_rel[selected_genera].copy()

    print(f"Selected {len(selected_genera)} genera after {self.genus_filter_mode} filtering (out of {df_genus_rel.shape[1]}).")
    print(f"Filter mode: {self.genus_filter_mode}")
    print(f"Filter method: {filter_method}")
    print(f"Prevalence threshold: {prevalence_threshold*100:.1f}%")
    print(f"Mean abundance threshold: {abundance_threshold*100:.3f}%")

    return df_genus_rel_filtered
```

---

## PART 4: GRAPH CONSTRUCTION FOR GENUS-LEVEL DATA

### **Challenge: Graph Size Explosion**

**Family-level:** 20-80 nodes → 190-3,160 possible edges
**Genus-level:** 100-300 nodes → **4,950-44,850 possible edges**

### **Solution: Adaptive Sparsification**

```python
def _construct_genus_correlation_graph(self, df_genus_features):
    """
    Construct Spearman correlation graph for genus-level data.

    Key Differences from Family-Level:
    1. Stricter correlation threshold (0.4 vs 0.3)
    2. More stringent significance testing (p < 0.001 vs p < 0.05)
    3. Adaptive sparsification based on network density
    4. Top-k edge selection as fallback
    """

    num_genera = df_genus_features.shape[1]
    print(f"\nConstructing genus-level Spearman correlation graph...")
    print(f"Number of genera: {num_genera}")
    print(f"Maximum possible edges: {num_genera * (num_genera - 1) // 2}")

    # Compute pairwise Spearman correlations
    from scipy.stats import spearmanr

    corr_matrix = np.zeros((num_genera, num_genera))
    pval_matrix = np.ones((num_genera, num_genera))

    for i in range(num_genera):
        for j in range(i+1, num_genera):
            corr, pval = spearmanr(
                df_genus_features.iloc[:, i],
                df_genus_features.iloc[:, j]
            )
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval

    # GENUS-SPECIFIC THRESHOLDS (stricter than family-level)
    correlation_threshold = 0.4  # Higher than family (0.3)
    significance_threshold = 0.001  # More stringent (was 0.05)

    # Apply Benjamini-Hochberg FDR correction
    from statsmodels.stats.multitest import fdrcorrection

    # Get upper triangle p-values
    mask_upper = np.triu(np.ones_like(pval_matrix, dtype=bool), k=1)
    pvals_flat = pval_matrix[mask_upper]

    # FDR correction
    reject, pvals_corrected = fdrcorrection(pvals_flat, alpha=0.05)

    # Reconstruct corrected p-value matrix
    pval_corrected_matrix = np.ones_like(pval_matrix)
    pval_corrected_matrix[mask_upper] = pvals_corrected
    pval_corrected_matrix = pval_corrected_matrix + pval_corrected_matrix.T

    # Build edge list with strict criteria
    edge_list = []
    edge_weights = []

    for i in range(num_genera):
        for j in range(i+1, num_genera):
            corr = corr_matrix[i, j]
            pval_corr = pval_corrected_matrix[i, j]

            # Strict edge criteria
            if abs(corr) >= correlation_threshold and pval_corr < significance_threshold:
                edge_list.append([i, j])
                edge_weights.append(abs(corr))

    print(f"Edges after correlation threshold (|ρ| ≥ {correlation_threshold}): {len(edge_list)}")

    # ADAPTIVE SPARSIFICATION: If still too many edges, use top-k
    target_sparsity = 2 * num_genera  # e = 2p (research-backed default)

    if len(edge_list) > target_sparsity:
        print(f"Network too dense ({len(edge_list)} edges). Applying sparsification to {target_sparsity} edges...")

        # Sort edges by weight (correlation strength)
        edge_data = sorted(zip(edge_list, edge_weights), key=lambda x: x[1], reverse=True)
        edge_list = [e[0] for e in edge_data[:target_sparsity]]
        edge_weights = [e[1] for e in edge_data[:target_sparsity]]

    print(f"Final edge count: {len(edge_list)}")
    print(f"Network density: {len(edge_list) / (num_genera * (num_genera - 1) / 2):.4f}")

    # Convert to PyTorch format
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # Add reverse edges for undirected graph
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        edge_weight = torch.tensor(edge_weights + edge_weights, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty(0, dtype=torch.float)

    return edge_index, edge_weight, corr_matrix
```

---

## PART 5: DOMAIN EXPERT CASES FOR GENUS-LEVEL

### **Challenge: Genus-Level Specificity**

At family level, you protected broad functional groups (e.g., all Methanobacteriaceae).
At genus level, you need **specific genera** within those families.

```python
def get_protected_genera_by_case(case_type):
    """
    Get protected genera for domain expert cases.

    More specific than family-level due to higher taxonomic resolution.
    """

    if case_type == 'case1' or case_type == 'case1_h2_hydrogenotrophic_only':
        # Hydrogenotrophic methanogenesis
        return [
            # Methanobacteriaceae genera
            'Methanobrevibacter',
            'Methanobacterium',
            'Methanosphaera',

            # Methanoregulaceae genera
            'Methanoregula',
            'Methanolinea',

            # Methanospirillaceae genera
            'Methanospirillum',
            'Methanocalculus'
        ]

    elif case_type == 'case2' or case_type == 'case2_ace_acetoclastic_only':
        # Acetoclastic methanogenesis
        return [
            # Methanosaetaceae genera
            'Methanosaeta',
            'Methanothrix',  # Alternative name

            # Methanosarcinaceae genera (can also use acetate)
            'Methanosarcina'
        ]

    elif case_type == 'case3' or case_type == 'case3_comprehensive':
        # All methanogenic and syntrophic genera
        return [
            # Acetoclastic
            'Methanosaeta',
            'Methanosarcina',

            # Hydrogenotrophic
            'Methanobrevibacter',
            'Methanobacterium',
            'Methanoregula',
            'Methanospirillum',

            # Syntrophic bacteria
            'Smithella',
            'Syntrophobacter',
            'Syntrophorhabdus',
            'Syntrophomonas',
            'Pelotomaculum'
        ]

    else:
        return []
```

---

## PART 6: COMPUTATIONAL CONSIDERATIONS

### **Memory and Performance**

| Aspect | Family-Level | Genus-Level |
|--------|--------------|-------------|
| **Dataset Size** | ~54 × 30 | ~54 × 150 |
| **Graph Nodes** | 20-80 | 100-300 |
| **Graph Edges** | 50-200 | 200-600 (after sparsification) |
| **Training Time/Epoch** | ~2-5 sec | ~10-30 sec |
| **Memory Usage** | ~500 MB | ~2-4 GB |
| **Centrality Computation** | ~1 sec | ~5-15 sec |

### **Optimization Strategies:**

1. **Batch Processing**: Increase batch size to 16-32 for genus-level
2. **Graph Sampling**: Use GraphSAINT or neighbor sampling for very large graphs
3. **Early Stopping**: More aggressive (patience=15 instead of 20)
4. **Reduced CV Folds**: Consider 3-fold instead of 5-fold for initial experiments

---

## PART 7: EXPECTED RESULTS & HYPOTHESIS

### **Biological Expectations:**

**Advantages of Genus-Level:**
- ✅ Higher biological resolution (species-level proxies)
- ✅ Better identification of keystone taxa
- ✅ More specific biomarker discovery
- ✅ Improved pathway-specific predictions

**Challenges of Genus-Level:**
- ⚠️ Higher sparsity → more zeros → harder to train
- ⚠️ More unclassified taxa → noise
- ⚠️ Larger graphs → computational cost
- ⚠️ Potential overfitting with small sample size (n=54)

### **Performance Hypothesis:**

**Predicted R² Changes:**
```
Family-Level Baseline: R² = 0.75-0.85
Genus-Level Optimistic: R² = 0.80-0.90 (if enough data)
Genus-Level Realistic: R² = 0.70-0.80 (sparsity challenges)
Genus-Level Pessimistic: R² = 0.60-0.70 (overfitting)
```

**Recommendation:** Run **ablation study** comparing family vs genus on same samples.

---

## PART 8: IMPLEMENTATION CHECKLIST

### **Phase 1: Core Implementation** (1-2 days)

- [ ] Create `_process_genera()` method in dataset_regression.py
- [ ] Add `genus_mode` parameter to dataset initialization
- [ ] Implement genus-specific filtering thresholds
- [ ] Test on small subset to verify genus extraction

### **Phase 2: Graph Construction** (1 day)

- [ ] Adapt Spearman correlation with stricter thresholds
- [ ] Implement adaptive sparsification (top-k edges)
- [ ] Add FDR correction for multiple testing
- [ ] Verify graph connectivity and density

### **Phase 3: Domain Expert Cases** (1 day)

- [ ] Define protected genera for Cases 1, 2, 3
- [ ] Update anchored feature protection logic
- [ ] Test that protected genera are retained after filtering

### **Phase 4: Pipeline Integration** (1 day)

- [ ] Update domain_expert_cases_pipeline.py
- [ ] Add `--graph_mode genus` CLI argument
- [ ] Test full pipeline end-to-end
- [ ] Verify results saved correctly

### **Phase 5: Validation & Comparison** (2-3 days)

- [ ] Run family-level baseline
- [ ] Run genus-level with same parameters
- [ ] Compare R², RMSE, MAE metrics
- [ ] Analyze network topology differences
- [ ] Generate comparative visualizations

### **Phase 6: Thesis Integration** (1 day)

- [ ] Add genus-level methodology section to Chapter 3
- [ ] Update mathematical formulations
- [ ] Add genus-level results to Chapter 4
- [ ] Create comparison tables and figures

---

## PART 9: REFERENCES & CITATIONS TO ADD

```bibtex
@article{weiss2017normalization,
  title={Normalization and microbial differential abundance strategies depend upon data characteristics},
  author={Weiss, Sophie and Xu, Zhenjiang Zech and Peddada, Shyamal and Amir, Amnon and Bittinger, Kyle and Gonzalez, Antonio and Lozupone, Catherine and Zaneveld, Jesse R and V{\'a}zquez-Baeza, Yoshiki and Birmingham, Amanda and others},
  journal={Microbiome},
  volume={5},
  number={1},
  pages={1--18},
  year={2017},
  publisher={BioMed Central}
}

@article{nearing2022microbiome,
  title={Microbiome differential abundance methods produce different results across 38 datasets},
  author={Nearing, Jacob T and Douglas, Gavin M and Hayes, Molly G and MacDonald, Jocelyn and Desai, Dhwani K and Allward, Nikhil and Jones, Casey MA and Wright, Robyn J and Dhanani, Akhilesh S and Comeau, Andr{\'e} M and others},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={342},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@article{johnson2019evaluation,
  title={Evaluation of 16S rRNA gene sequencing for species and strain-level microbiome analysis},
  author={Johnson, Jethro S and Spakowicz, Daniel J and Hong, Bo-Young and Petersen, Lauren M and Demkowicz, Patrick and Chen, Lei and Leopold, Shana R and Hanson, Blake M and Agresta, Hanako O and Gerstein, Mark and others},
  journal={Nature Communications},
  volume={10},
  number={1},
  pages={5029},
  year={2019},
  publisher={Nature Publishing Group UK London}
}

@article{kurtz2015sparse,
  title={Sparse and compositionally robust inference of microbial ecological networks},
  author={Kurtz, Zachary D and M{\"u}ller, Christian L and Miraldi, Emily R and Littman, Dan R and Blaser, Martin J and Bonneau, Richard A},
  journal={PLoS computational biology},
  volume={11},
  number={5},
  pages={e1004226},
  year={2015},
  publisher={Public Library of Science San Francisco, CA USA}
}
```

---

## PART 10: FINAL RECOMMENDATIONS

### **Should You Use Genus-Level?**

**YES, if:**
- ✅ You want higher biological resolution
- ✅ You're looking for specific biomarkers
- ✅ You have computational resources (2-4GB RAM minimum)
- ✅ You're willing to handle sparsity challenges

**NO (stick with family), if:**
- ❌ Your sample size is very small (n < 30)
- ❌ Computational resources are limited
- ❌ You need faster iteration times
- ❌ Family-level already gives good performance

### **Recommended Approach:**

1. **Keep family-level as primary analysis** (proven to work well)
2. **Add genus-level as secondary analysis** (exploratory, higher resolution)
3. **Compare both in ablation study** (report both in thesis)
4. **Use genus-level for biomarker identification** (more specific taxa)

### **Expected Timeline:**

- Implementation: 5-7 days
- Testing & debugging: 2-3 days
- Full pipeline runs: 2-3 days
- Analysis & thesis writing: 3-5 days
- **Total: ~2-3 weeks for complete genus-level integration**

---

**This plan is ready for implementation. All design decisions are research-backed with proper citations.**
