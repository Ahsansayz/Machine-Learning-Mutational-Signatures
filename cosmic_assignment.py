#!/usr/bin/env python3
"""
COSMIC v3.4 Signature Assignment
=========================================
Maps each sample's mutation profile to known COSMIC (v3.4) reference signatures.
Tries SigProfilerAssignment first, falls back to manual cosine similarity fitting.

Input:  output/sbs96_matrix.csv (from Step 1)
Output: output/cosmic_assignment/
        - cosmic_activities.csv (samples × COSMIC signatures)
        - cosmic_cosine_similarities.csv
        - top_signatures_per_sample.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "output/sbs96_matrix.csv"
OUTPUT_DIR = "output/cosmic_assignment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# COSMIC SBS96 Reference Signatures (v3.4)
# Key signatures for gastric cancer
# ============================================================
# We'll download the full COSMIC reference or use the built-in one

def try_sigprofiler_assignment(input_file):
    """Try SigProfilerAssignment for COSMIC fitting."""
    try:
        from SigProfilerAssignment import Analyzer as Analyze
        print("✅ SigProfilerAssignment found! Using it for COSMIC fitting.")
        
        Analyze.cosmic_fit(
            samples=input_file,
            output=OUTPUT_DIR,
            input_type="matrix",
            context_type="96",
            genome_build="GRCh38",
            exome=True,
            cosmic_version=3.4,
            make_plots=True,
            sample_reconstruction_plots=False
        )
        
        # Find and rename the activities file
        activities_path = None
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for f in files:
                if 'Activities' in f and f.endswith('.txt'):
                    activities_path = os.path.join(root, f)
                    break
        
        if activities_path:
            activities = pd.read_csv(activities_path, sep='\t', index_col=0)
            activities.to_csv(os.path.join(OUTPUT_DIR, "cosmic_activities.csv"))
            print(f"  Saved: cosmic_activities.csv ({activities.shape})")
        
        return True
    except ImportError:
        print("⚠️ SigProfilerAssignment not installed. Using manual COSMIC fitting.")
        return False
    except Exception as e:
        print(f"⚠️ SigProfilerAssignment failed: {e}. Using manual fitting.")
        return False


def download_cosmic_reference():
    """Download COSMIC SBS96 reference signatures or use hardcoded key signatures."""
    cosmic_url = "https://cancer.sanger.ac.uk/signatures/documents/2123/COSMIC_v3.4_SBS_GRCh38.txt"
    
    cosmic_file = os.path.join(OUTPUT_DIR, "COSMIC_v3.4_SBS_GRCh38.txt")
    
    if os.path.exists(cosmic_file):
        print(f"  Using cached COSMIC reference: {cosmic_file}")
        return pd.read_csv(cosmic_file, sep='\t', index_col=0)
    
    try:
        import urllib.request
        print(f"  Downloading COSMIC v3.4 reference signatures...")
        urllib.request.urlretrieve(cosmic_url, cosmic_file)
        cosmic = pd.read_csv(cosmic_file, sep='\t', index_col=0)
        print(f"  Downloaded: {cosmic.shape[1]} COSMIC signatures")
        return cosmic
    except Exception as e:
        print(f"  ⚠️ Could not download COSMIC reference: {e}")
        print(f"  Trying alternative URL...")
        
        # Try alternative
        alt_url = "https://raw.githubusercontent.com/AlexandrovLab/SigProfilerExtractor/master/SigProfilerExtractor/data/COSMIC_v3.4_SBS_GRCh38.txt"
        try:
            urllib.request.urlretrieve(alt_url, cosmic_file)
            cosmic = pd.read_csv(cosmic_file, sep='\t', index_col=0)
            print(f"  Downloaded from alt: {cosmic.shape[1]} COSMIC signatures")
            return cosmic
        except:
            print(f"  ❌ Could not download. Will use de novo signatures only.")
            return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def nnls_fit(sample_profile, cosmic_signatures):
    """
    Non-negative least squares fitting of COSMIC signatures to a sample.
    This determines how much each COSMIC signature contributes to the sample.
    """
    from scipy.optimize import nnls
    
    # cosmic_signatures: matrix where each column is a COSMIC signature
    # sample_profile: the sample's 96-channel mutation profile
    
    # Solve: min ||sample - cosmic @ weights||^2 subject to weights >= 0
    weights, residual = nnls(cosmic_signatures, sample_profile)
    
    return weights


def manual_cosmic_fitting(input_file):
    """Manual COSMIC signature fitting using NNLS."""
    print("\n🔬 Running manual COSMIC v3.4 signature fitting...")
    
    # Load our mutation matrix
    matrix = pd.read_csv(input_file, index_col=0)
    print(f"  Loaded matrix: {matrix.shape[0]} samples × {matrix.shape[1]} channels")
    
    # Download COSMIC reference
    cosmic_ref = download_cosmic_reference()
    
    if cosmic_ref is None:
        print("  ❌ No COSMIC reference available. Skipping COSMIC assignment.")
        print("  💡 You can manually download from: https://cancer.sanger.ac.uk/signatures/")
        return None, None
    
    # Align channels between our matrix and COSMIC reference
    common_channels = [ch for ch in matrix.columns if ch in cosmic_ref.index]
    
    if len(common_channels) < 90:
        # Try mapping — channel labels might differ slightly
        print(f"  ⚠️ Only {len(common_channels)} channels match directly.")
        print(f"  Attempting label harmonization...")
        
        # Map our labels to COSMIC format or vice versa
        our_channels = set(matrix.columns)
        cosmic_channels = set(cosmic_ref.index)
        
        print(f"  Our channels (sample): {list(our_channels)[:5]}")
        print(f"  COSMIC channels (sample): {list(cosmic_channels)[:5]}")
        
        # If no match, try without brackets format
        if len(common_channels) < 10:
            print("  ❌ Channel format mismatch. Creating mapping...")
            # Our format: A[C>A]A  COSMIC might be: A[C>A]A or ACA etc
            # They should match if we used standard SBS96 labels
            print("  Please check channel label format manually.")
            return None, None
    
    print(f"  Matching channels: {len(common_channels)}/96")
    
    # Align
    matrix_aligned = matrix[common_channels]
    cosmic_aligned = cosmic_ref.loc[common_channels]
    
    sig_names = cosmic_aligned.columns.tolist()
    print(f"  COSMIC signatures: {len(sig_names)}")
    
    # Fit each sample
    activities = pd.DataFrame(0.0, index=matrix.index, columns=sig_names)
    cos_sims = pd.DataFrame(0.0, index=matrix.index, columns=['cosine_similarity'])
    
    from tqdm import tqdm
    
    for sample in tqdm(matrix_aligned.index, desc="Fitting COSMIC signatures"):
        sample_profile = matrix_aligned.loc[sample].values.astype(float)
        
        if sample_profile.sum() == 0:
            continue
        
        # NNLS fitting
        cosmic_matrix = cosmic_aligned.values.astype(float)  # channels × signatures
        weights = nnls_fit(sample_profile, cosmic_matrix)
        activities.loc[sample] = weights
        
        # Calculate reconstruction cosine similarity
        reconstruction = cosmic_matrix @ weights
        cos_sim = cosine_similarity(sample_profile, reconstruction)
        cos_sims.loc[sample, 'cosine_similarity'] = cos_sim
    
    # Filter out signatures with zero activity across all samples
    active_sigs = activities.columns[activities.sum(axis=0) > 0]
    activities_filtered = activities[active_sigs]
    
    # Save results
    activities_filtered.to_csv(os.path.join(OUTPUT_DIR, "cosmic_activities.csv"))
    cos_sims.to_csv(os.path.join(OUTPUT_DIR, "cosmic_cosine_similarities.csv"))
    
    # Top signatures per sample
    top_sigs = []
    for sample in activities_filtered.index:
        row = activities_filtered.loc[sample]
        total = row.sum()
        if total > 0:
            props = (row / total).sort_values(ascending=False)
            top3 = props.head(3)
            top_sigs.append({
                'Sample': sample,
                'Top1': f"{top3.index[0]} ({top3.values[0]:.1%})",
                'Top2': f"{top3.index[1]} ({top3.values[1]:.1%})" if len(top3) > 1 else "",
                'Top3': f"{top3.index[2]} ({top3.values[2]:.1%})" if len(top3) > 2 else "",
                'Cosine_Similarity': cos_sims.loc[sample, 'cosine_similarity']
            })
    
    top_df = pd.DataFrame(top_sigs)
    top_df.to_csv(os.path.join(OUTPUT_DIR, "top_signatures_per_sample.csv"), index=False)
    
    # Generate plots
    _plot_cosmic_landscape(activities_filtered)
    _plot_cosine_similarity_distribution(cos_sims)
    
    return activities_filtered, cos_sims


def _plot_cosmic_landscape(activities):
    """Plot the COSMIC signature landscape across all samples."""
    # Normalize to proportions
    props = activities.div(activities.sum(axis=1) + 1e-10, axis=0)
    
    # Get top 15 most active signatures
    top_sigs = props.mean().sort_values(ascending=False).head(15).index
    props_top = props[top_sigs]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Stacked bar plot
    props_top_sorted = props_top.sort_values(by=list(top_sigs), ascending=False)
    props_top_sorted.plot(kind='bar', stacked=True, ax=ax, width=1.0,
                           colormap='tab20', edgecolor='none')
    
    ax.set_title('COSMIC Signature Landscape — TCGA-STAD Gastric Cancer', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Samples (n={len(props)})')
    ax.set_ylabel('Relative Signature Contribution')
    ax.set_xticklabels([])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cosmic_signature_landscape.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {OUTPUT_DIR}/cosmic_signature_landscape.png")


def _plot_cosine_similarity_distribution(cos_sims):
    """Plot distribution of reconstruction cosine similarities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    values = cos_sims['cosine_similarity'].values
    ax.hist(values, bins=50, color='#0f3460', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(values), color='#e94560', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(values):.3f}')
    ax.axvline(x=0.90, color='gray', linestyle=':', linewidth=1,
               label='Quality threshold (0.90)')
    
    ax.set_xlabel('Cosine Similarity (Reconstruction)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('COSMIC Signature Fitting Quality', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cosine_similarity_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {OUTPUT_DIR}/cosine_similarity_distribution.png")


def main():
    print("=" * 60)
    print("COSMIC v3.4 Signature Assignment")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Run build_sbs96_matrix.py first!")
        sys.exit(1)
    
    # Try SigProfilerAssignment first
    success = try_sigprofiler_assignment(INPUT_FILE)
    
    if not success:
        activities, cos_sims = manual_cosmic_fitting(INPUT_FILE)
        
        if activities is not None:
            print(f"\n{'=' * 60}")
            print(f"📊 COSMIC Assignment Summary")
            print(f"{'=' * 60}")
            print(f"  Samples fitted: {len(activities)}")
            print(f"  Active COSMIC signatures: {activities.shape[1]}")
            print(f"  Mean cosine similarity: {cos_sims['cosine_similarity'].mean():.3f}")
            print(f"  Samples with cos_sim > 0.90: {(cos_sims['cosine_similarity'] > 0.90).sum()}")
    
    print(f"\n✅ COSMIC assignment complete!")


if __name__ == "__main__":
    main()
