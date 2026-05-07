#!/usr/bin/env python3
"""
De Novo Mutational Signature Extraction
================================================
Extracts mutational signatures from the 96-channel SBS matrix using NMF.
Tries SigProfilerExtractor first, falls back to custom NMF with stability analysis.

Input:  output/sbs96_matrix.csv (from Step 1)
Output: output/signatures/ (extracted signatures, activities, plots)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

OUTPUT_DIR = "output/signatures"
INPUT_FILE = "output/sbs96_matrix.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def try_sigprofiler_extraction(input_file):
    """Try using SigProfilerExtractor (the gold standard)."""
    try:
        from SigProfilerExtractor import sigpro as sig
        print("✅ SigProfilerExtractor found! Using it for extraction.")
        
        sig.sigProfilerExtractor(
            input_type="matrix",
            output=OUTPUT_DIR,
            input_data=input_file,
            reference_genome="GRCh38",
            minimum_signatures=2,
            maximum_signatures=15,
            nmf_replicates=100,
            cpu=-1,
            gpu=False,
            batch_size=1,
            seeds="random",
            opportunity_genome="GRCh38",
            context_type="default",
            exome=True,
            make_decomposition_plots=True,
            collapse_to_SBS96=True
        )
        return True
    except ImportError:
        print("⚠️ SigProfilerExtractor not installed. Using custom NMF.")
        return False
    except Exception as e:
        print(f"⚠️ SigProfilerExtractor failed: {e}. Using custom NMF.")
        return False


def custom_nmf_extraction(input_file, min_rank=2, max_rank=12, n_runs=50):
    """
    Custom NMF-based signature extraction with stability analysis.
    Implements the approach from Alexandrov et al. (2013).
    """
    from sklearn.decomposition import NMF
    from sklearn.metrics import silhouette_score
    
    print("\n🔬 Running custom NMF signature extraction...")
    print(f"   Testing ranks {min_rank} to {max_rank}, {n_runs} runs each\n")
    
    # Load matrix
    matrix = pd.read_csv(input_file, index_col=0)
    X = matrix.values.astype(float)
    
    # Ensure non-negative
    X = np.maximum(X, 0)
    
    # Store metrics for each rank
    rank_metrics = {}
    
    for rank in range(min_rank, max_rank + 1):
        print(f"  Rank {rank}: ", end="", flush=True)
        
        reconstruction_errors = []
        all_W = []
        all_H = []
        
        for run in range(n_runs):
            model = NMF(
                n_components=rank,
                init='nndsvda',
                random_state=run * 42,
                max_iter=1000,
                tol=1e-4,
                l1_ratio=0.0,
                alpha_W=0.0
            )
            
            W = model.fit_transform(X)  # samples × signatures
            H = model.components_       # signatures × mutation types
            
            # Reconstruction error
            reconstruction = np.linalg.norm(X - W @ H, 'fro') / np.linalg.norm(X, 'fro')
            reconstruction_errors.append(reconstruction)
            all_W.append(W)
            all_H.append(H)
        
        # Calculate stability (average cosine similarity between runs)
        stability = _calculate_stability(all_H, rank)
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        
        rank_metrics[rank] = {
            'mean_error': mean_error,
            'std_error': std_error,
            'stability': stability,
            'best_W': all_W[np.argmin(reconstruction_errors)],
            'best_H': all_H[np.argmin(reconstruction_errors)],
        }
        
        print(f"error={mean_error:.4f} ± {std_error:.4f}, stability={stability:.4f}")
    
    # Select optimal rank: maximize stability while minimizing error
    # Use the "elbow" approach + stability threshold
    optimal_rank = _select_optimal_rank(rank_metrics)
    print(f"\n🏆 Optimal number of signatures: {optimal_rank}")
    
    # Get best decomposition for optimal rank
    best = rank_metrics[optimal_rank]
    W = best['best_W']
    H = best['best_H']
    
    # Normalize signatures (columns of H sum to 1)
    H_norm = H / H.sum(axis=1, keepdims=True)
    
    # Save results
    sig_names = [f"Sig_{i+1}" for i in range(optimal_rank)]
    
    # Signature profiles (what each signature looks like)
    sig_profiles = pd.DataFrame(H_norm, index=sig_names, columns=matrix.columns)
    sig_profiles.to_csv(os.path.join(OUTPUT_DIR, "signature_profiles.csv"))
    
    # Signature activities/exposures (how much each signature contributes to each sample)
    sig_activities = pd.DataFrame(W, index=matrix.index, columns=sig_names)
    sig_activities.to_csv(os.path.join(OUTPUT_DIR, "signature_activities.csv"))
    
    # Rank selection metrics
    metrics_df = pd.DataFrame({
        'rank': list(rank_metrics.keys()),
        'mean_error': [m['mean_error'] for m in rank_metrics.values()],
        'std_error': [m['std_error'] for m in rank_metrics.values()],
        'stability': [m['stability'] for m in rank_metrics.values()],
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "rank_selection_metrics.csv"), index=False)
    
    # Generate plots
    _plot_rank_selection(metrics_df, optimal_rank)
    _plot_signature_profiles(sig_profiles, matrix.columns)
    _plot_signature_activities_heatmap(sig_activities)
    
    return sig_profiles, sig_activities, optimal_rank


def _calculate_stability(all_H, rank):
    """
    Calculate stability of NMF decomposition across runs.
    Uses average pairwise cosine similarity of matched signatures.
    """
    from itertools import combinations
    
    n_runs = len(all_H)
    if n_runs < 2:
        return 1.0
    
    # Sample a subset for efficiency
    max_pairs = min(100, n_runs * (n_runs - 1) // 2)
    pairs = list(combinations(range(n_runs), 2))
    if len(pairs) > max_pairs:
        np.random.seed(42)
        pairs = [pairs[i] for i in np.random.choice(len(pairs), max_pairs, replace=False)]
    
    similarities = []
    for i, j in pairs:
        H_i = all_H[i]
        H_j = all_H[j]
        
        # Compute cosine similarity matrix between signatures
        cos_sim = _cosine_similarity_matrix(H_i, H_j)
        
        # Hungarian matching (greedy approximation)
        matched_sims = []
        used = set()
        for _ in range(rank):
            best_val = -1
            best_pos = (0, 0)
            for r in range(rank):
                for c in range(rank):
                    if c not in used and cos_sim[r, c] > best_val:
                        best_val = cos_sim[r, c]
                        best_pos = (r, c)
            matched_sims.append(best_val)
            used.add(best_pos[1])
        
        similarities.append(np.mean(matched_sims))
    
    return np.mean(similarities)


def _cosine_similarity_matrix(A, B):
    """Compute pairwise cosine similarity between rows of A and B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def _select_optimal_rank(rank_metrics):
    """Select optimal rank using stability plateau + error elbow."""
    ranks = sorted(rank_metrics.keys())
    
    # Find ranks with high stability (> 0.8)
    stable_ranks = [r for r in ranks if rank_metrics[r]['stability'] > 0.8]
    
    if stable_ranks:
        # Among stable ranks, find the elbow in reconstruction error
        errors = [rank_metrics[r]['mean_error'] for r in stable_ranks]
        
        # Use the highest stable rank before stability drops significantly
        best = stable_ranks[0]
        for i, r in enumerate(stable_ranks[1:], 1):
            # Check if adding more signatures significantly reduces error
            error_reduction = errors[i-1] - errors[i]
            if error_reduction > 0.005:  # Meaningful reduction
                best = r
            else:
                break
        return best
    else:
        # Fall back: pick rank with best stability
        return max(ranks, key=lambda r: rank_metrics[r]['stability'])


def _plot_rank_selection(metrics_df, optimal_rank):
    """Plot rank selection metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reconstruction error
    ax1.errorbar(metrics_df['rank'], metrics_df['mean_error'], 
                  yerr=metrics_df['std_error'], marker='o', capsize=3,
                  color='#e94560', linewidth=2, markersize=6)
    ax1.axvline(x=optimal_rank, color='#0f3460', linestyle='--', alpha=0.7,
                label=f'Optimal rank = {optimal_rank}')
    ax1.set_xlabel('Number of Signatures', fontsize=12)
    ax1.set_ylabel('Reconstruction Error (Frobenius)', fontsize=12)
    ax1.set_title('Reconstruction Error vs Rank', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stability
    ax2.plot(metrics_df['rank'], metrics_df['stability'], marker='s',
             color='#16213e', linewidth=2, markersize=6)
    ax2.axvline(x=optimal_rank, color='#e94560', linestyle='--', alpha=0.7,
                label=f'Optimal rank = {optimal_rank}')
    ax2.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Stability threshold')
    ax2.set_xlabel('Number of Signatures', fontsize=12)
    ax2.set_ylabel('Average Stability (Cosine Similarity)', fontsize=12)
    ax2.set_title('Signature Stability vs Rank', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rank_selection.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {OUTPUT_DIR}/rank_selection.png")


def _plot_signature_profiles(sig_profiles, channel_labels):
    """Plot the 96-channel SBS profile for each extracted signature."""
    n_sigs = len(sig_profiles)
    
    # Color scheme for 6 substitution types
    sub_colors = {
        'C>A': '#03BCEE', 'C>G': '#010101', 'C>T': '#E32926',
        'T>A': '#CAC9C9', 'T>C': '#A1CE63', 'T>G': '#EBC5C3'
    }
    
    fig, axes = plt.subplots(n_sigs, 1, figsize=(20, 3.5 * n_sigs))
    if n_sigs == 1:
        axes = [axes]
    
    for idx, (sig_name, profile) in enumerate(sig_profiles.iterrows()):
        ax = axes[idx]
        colors = []
        for ch in channel_labels:
            # Extract substitution type from channel label (e.g., 'A[C>A]G' -> 'C>A')
            sub = ch[2:5]
            colors.append(sub_colors.get(sub, '#888888'))
        
        bars = ax.bar(range(96), profile.values, color=colors, width=0.8, edgecolor='none')
        ax.set_title(sig_name, fontsize=14, fontweight='bold', loc='left')
        ax.set_xlim(-1, 96)
        ax.set_ylabel('Probability')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == n_sigs - 1:
            # Add substitution type labels
            for i, sub in enumerate(['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']):
                ax.text(i * 16 + 8, -0.02, sub, ha='center', fontsize=10,
                        fontweight='bold', transform=ax.get_xaxis_transform(),
                        color=sub_colors[sub])
        
        ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "signature_profiles_96ch.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {OUTPUT_DIR}/signature_profiles_96ch.png")


def _plot_signature_activities_heatmap(sig_activities):
    """Plot heatmap of signature activities across samples."""
    # Normalize to proportions
    props = sig_activities.div(sig_activities.sum(axis=1) + 1e-10, axis=0)
    
    # Sort samples by dominant signature
    dominant = props.idxmax(axis=1)
    props['dominant'] = dominant
    props = props.sort_values('dominant')
    props = props.drop('dominant', axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(props, cmap='YlOrRd', xticklabels=True, yticklabels=False,
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Relative Contribution'})
    ax.set_title('Mutational Signature Activities Across Samples', fontsize=14, fontweight='bold')
    ax.set_xlabel('Signatures')
    ax.set_ylabel(f'Samples (n={len(props)})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "signature_activities_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: {OUTPUT_DIR}/signature_activities_heatmap.png")


def main():
    print("=" * 60)
    print("De Novo Mutational Signature Extraction")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Run build_sbs96_matrix.py first!")
        sys.exit(1)
    
    # Try SigProfilerExtractor first
    success = try_sigprofiler_extraction(INPUT_FILE)
    
    if not success:
        # Fall back to custom NMF
        sig_profiles, sig_activities, optimal_rank = custom_nmf_extraction(
            INPUT_FILE, min_rank=2, max_rank=12, n_runs=50
        )
        
        print(f"\n{'=' * 60}")
        print(f"📊 Extraction Summary")
        print(f"{'=' * 60}")
        print(f"  Optimal signatures: {optimal_rank}")
        print(f"  Signature profiles: {OUTPUT_DIR}/signature_profiles.csv")
        print(f"  Signature activities: {OUTPUT_DIR}/signature_activities.csv")
        print(f"  Rank selection plot: {OUTPUT_DIR}/rank_selection.png")
    
    print(f"\n✅ Signature extraction complete!")


if __name__ == "__main__":
    main()
