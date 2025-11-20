"""
Test the visualization coloring logic with actual data.
"""

# Actual features from the dataset
actual_dataset_features = [
    'Aminicenantales', 'WS1', 'Smithella', 'WCHB1-41', 'UnclassifiedGenus',
    'Leptolinea', 'Flexilinea', 'JGI-0000079-D21', 'Methanospirillum',
    'Candidatus_Caldatribacterium', 'Mesotoga', 'Lentimicrobium',
    'Bacteroidetes_vadinHA17', 'Thermovirga', 'Pelolinea', 'LF045',
    'SAR324_clade(Marine_group_B)', 'WCHB1-02', 'Methanolinea', 'Anaerolinea',
    'Methanosaeta', 'Methanobacterium', 'Hydrogenedensaceae', 'Syner-01',
    'Syntrophomonas', 'Proteiniphilum', 'Syntrophus', 'Syntrophobacter',
    'Christensenellaceae_R-7_group', 'Candidatus_Omnitrophus',
    'Candidatus_Methanofastidiosum', 'Pelotomaculum', 'DMER64', 'Spirochaeta',
    'LD1-PB3', 'SBR1031', 'Syntrophorhabdus', 'ADurb.Bin063-1', 'uncultured', 'SG8-4'
]

# Protected nodes (from the matching logic - with duplicate "uncultured")
protected_nodes_with_dup = [
    'Methanosaeta', 'Methanolinea', 'Methanobacterium', 'Methanospirillum',
    'Smithella', 'Syntrophorhabdus', 'Syntrophobacter', 'Syner-01',
    'uncultured', 'uncultured',  # DUPLICATE
    'DMER64', 'Thermovirga', 'Syntrophomonas', 'Syntrophus',
    'JGI-0000079-D21', 'Pelotomaculum'
]

# Simulate the coloring function from visualization_utils.py
def get_node_colors_with_protection(node_list, protected_list=None):
    """Simulate the actual coloring function."""
    colors = []
    for node in node_list:
        if protected_list and node in protected_list:
            colors.append('#606060')  # Dark gray for protected
        else:
            colors.append('#D3D3D3')  # Light gray for others
    return colors

print("="*80)
print("VISUALIZATION COLORING SIMULATION")
print("="*80)

# Apply the coloring
colors = get_node_colors_with_protection(actual_dataset_features, protected_nodes_with_dup)

# Analyze the results
protected_count = sum(1 for c in colors if c == '#606060')
non_protected_count = sum(1 for c in colors if c == '#D3D3D3')

print(f"\nTotal nodes: {len(actual_dataset_features)}")
print(f"Protected nodes (dark): {protected_count}")
print(f"Non-protected nodes (light): {non_protected_count}")

print("\n" + "="*80)
print("PROTECTED NODES (should be colored DARK #606060):")
print("="*80)
for i, (node, color) in enumerate(zip(actual_dataset_features, colors)):
    if color == '#606060':
        print(f"  {i+1:2d}. {node:40s} <- PROTECTED")

print("\n" + "="*80)
print("NON-PROTECTED NODES (should be colored LIGHT #D3D3D3):")
print("="*80)
for i, (node, color) in enumerate(zip(actual_dataset_features, colors)):
    if color == '#D3D3D3':
        print(f"  {i+1:2d}. {node:40s} <- NOT PROTECTED")

# Verify all anchored features are protected
print("\n" + "="*80)
print("VERIFICATION: Are all anchored features properly protected?")
print("="*80)

expected_protected = set(protected_nodes_with_dup)
actually_protected = set([node for node, color in zip(actual_dataset_features, colors) if color == '#606060'])

print(f"\nExpected protected (unique): {sorted(expected_protected)}")
print(f"Actually protected: {sorted(actually_protected)}")

if expected_protected == actually_protected:
    print("\n✅ SUCCESS: All anchored features are correctly protected!")
else:
    missing = expected_protected - actually_protected
    extra = actually_protected - expected_protected
    if missing:
        print(f"\n⚠ MISSING: These should be protected but aren't: {missing}")
    if extra:
        print(f"\n⚠ EXTRA: These are protected but shouldn't be: {extra}")

# Check if the visualization would show the protected nodes correctly
print("\n" + "="*80)
print("EXPECTED VISUALIZATION BEHAVIOR:")
print("="*80)
print(f"In the graph visualization:")
print(f"  - {protected_count} nodes should appear in DARK GRAY (#606060)")
print(f"  - {non_protected_count} nodes should appear in LIGHT GRAY (#D3D3D3)")
print(f"\nThe protected nodes are: {sorted(actually_protected)}")
