#!/usr/bin/env python3
"""
Visualization & Interpretation
=======================================
Generates publication-ready figures including:
1. SHAP explainability analysis
2. t-SNE/UMAP dimensionality reduction
3. Signature heatmap by subtype
4. Kaplan-Meier survival curves
5. Comprehensive summary plots

Input:  output/ml_results/, output/cosmic_assignment/, data/clinical_data.csv
Output: output/figures/
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = "output/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_shap_analysis():
    """Generate SHAP summary and force plots for model explainability."""
    try:
        import shap
    except ImportError:
        print("  ⚠️ SHAP not installed. Skipping SHAP analysis.")
        return
    
    print("\n🔍 Generating SHAP Explainability Plots...")
    
    # Load model and data
    model_path = "output/ml_results/best_model.pkl"
    if not os.path.exists(model_path):
        print("  ⚠️ No saved model found. Skipping SHAP.")
        return
    
    with open(model_path, 'rb') as f:
        saved = pickle.load(f)
    
    model = saved['model']
    le = saved['label_encoder']
    
    X = pd.read_csv("output/ml_features.csv", index_col=0)
    y_df = pd.read_csv("output/ml_labels.csv", index_col=0)
    common = X.index.intersection(y_df.index)
    X = X.loc[common]
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    # Scale features
    scaler = model.named_steps['scaler']
    classifier = model.named_steps['classifier']
    
    X_scaled = pd.DataFrame(
        scaler.transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    # Fit classifier for SHAP
    y_encoded = le.transform(y_df.loc[common, 'molecular_subtype'])
    classifier.fit(X_scaled, y_encoded)
    
    # SHAP explainer
    try:
        if hasattr(classifier, 'feature_importances_'):
            explainer = shap.TreeExplainer(classifier)
        else:
            explainer = shap.KernelExplainer(classifier.predict_proba, 
                                              shap.sample(X_scaled, 50))
        
        shap_values = explainer.shap_values(X_scaled)
        
        # Summary plot (bar)
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_scaled, class_names=le.classes_,
                          show=False, max_display=20)
        plt.title('SHAP Feature Importance — Gastric Cancer Subtype Classification',
                  fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: shap_summary.png")
        
        # SHAP bar plot
        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_scaled, class_names=le.classes_,
                          plot_type='bar', show=False, max_display=20)
        plt.title('Mean |SHAP| Feature Importance by Subtype', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📊 Saved: shap_bar.png")
        
    except Exception as e:
        print(f"  ⚠️ SHAP analysis failed: {e}")


def plot_dimensionality_reduction():
    """Generate t-SNE and UMAP plots colored by subtype."""
    print("\n🗺️ Generating Dimensionality Reduction Plots...")
    
    X = pd.read_csv("output/ml_features.csv", index_col=0)
    y_df = pd.read_csv("output/ml_labels.csv", index_col=0)
    common = X.index.intersection(y_df.index)
    X = X.loc[common]
    y = y_df.loc[common, 'molecular_subtype']
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    
    X_scaled = StandardScaler().fit_transform(X)
    
    # t-SNE
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
    X_tsne = tsne.fit_transform(X_scaled)
    
    palette = {'EBV': '#e94560', 'MSI': '#0f3460', 'GS': '#16213e', 'CIN': '#533483'}
    # Add fallback colors for any label
    unique_labels = y.unique()
    default_colors = ['#e94560', '#0f3460', '#16213e', '#533483', '#2a9d8f', '#e76f51']
    for i, label in enumerate(unique_labels):
        if label not in palette:
            palette[label] = default_colors[i % len(default_colors)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # t-SNE
    for subtype in unique_labels:
        mask = y == subtype
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                        c=palette[subtype], label=subtype, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
    axes[0].set_title('t-SNE — Signature-based Sample Clustering', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    axes[0].legend(fontsize=10)
    
    # UMAP (if available)
    try:
        import umap
        print("  Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        X_umap = reducer.fit_transform(X_scaled)
        
        for subtype in unique_labels:
            mask = y == subtype
            axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1],
                            c=palette[subtype], label=subtype, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        axes[1].set_title('UMAP — Signature-based Sample Clustering', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        axes[1].legend(fontsize=10)
    except ImportError:
        print("  ⚠️ UMAP not available. Showing only t-SNE.")
        axes[1].text(0.5, 0.5, 'UMAP not installed\npip install umap-learn', 
                     ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title('UMAP (not available)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: tsne_umap.png")


def plot_signature_heatmap_by_subtype():
    """Plot mean signature contribution heatmap grouped by molecular subtype."""
    print("\n🔥 Generating Signature-Subtype Heatmap...")
    
    # Load signature activities
    cosmic_file = "output/cosmic_assignment/cosmic_activities.csv"
    denovo_file = "output/signatures/signature_activities.csv"
    
    if os.path.exists(cosmic_file):
        activities = pd.read_csv(cosmic_file, index_col=0)
    elif os.path.exists(denovo_file):
        activities = pd.read_csv(denovo_file, index_col=0)
    else:
        print("  ⚠️ No signature activities found. Skipping.")
        return
    
    # Load labels
    y_df = pd.read_csv("output/ml_labels.csv", index_col=0)
    common = activities.index.intersection(y_df.index)
    
    if len(common) == 0:
        print("  ⚠️ No matching samples between signatures and labels. Skipping.")
        return
    
    activities_aligned = activities.loc[common]
    subtypes = y_df.loc[common, 'molecular_subtype']
    
    # Normalize to proportions
    props = activities_aligned.div(activities_aligned.sum(axis=1) + 1e-10, axis=0)
    
    # Filter to signatures with meaningful activity (mean > 1%)
    active_sigs = props.columns[props.mean() > 0.01]
    if len(active_sigs) > 25:
        active_sigs = props.mean().sort_values(ascending=False).head(25).index
    
    props_filtered = props[active_sigs]
    props_filtered['Subtype'] = subtypes
    
    # Mean by subtype
    mean_by_subtype = props_filtered.groupby('Subtype').mean()
    
    fig, ax = plt.subplots(figsize=(max(14, len(active_sigs)*0.6), 6))
    sns.heatmap(mean_by_subtype, cmap='YlOrRd', annot=True, fmt='.3f',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Mean Relative Contribution'})
    ax.set_title('Mean Mutational Signature Contribution by Molecular Subtype', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Molecular Subtype', fontsize=12)
    ax.set_xlabel('COSMIC Signature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "signature_subtype_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: signature_subtype_heatmap.png")


def plot_survival_analysis():
    """Generate Kaplan-Meier survival curves by predicted subtype."""
    print("\n📈 Generating Survival Analysis...")
    
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        print("  ⚠️ lifelines not installed. Skipping survival analysis.")
        return
    
    # Load clinical data
    clinical_file = "data/clinical_data.csv"
    if not os.path.exists(clinical_file):
        print("  ⚠️ No clinical data. Skipping survival analysis.")
        return
    
    clinical = pd.read_csv(clinical_file)
    
    # Check for survival columns
    has_survival = ('vital_status' in clinical.columns and 
                   ('days_to_death' in clinical.columns or 'days_to_last_follow_up' in clinical.columns))
    
    if not has_survival:
        print("  ⚠️ No survival data in clinical file. Skipping.")
        return
    
    # Prepare survival data
    clinical['OS_MONTHS'] = pd.to_numeric(
        clinical['days_to_death'].fillna(clinical.get('days_to_last_follow_up', 0)), 
        errors='coerce'
    ) / 30.44
    
    clinical['OS_STATUS'] = (clinical['vital_status'].str.lower() == 'dead').astype(int)
    
    # Get subtypes
    if 'molecular_subtype' not in clinical.columns:
        print("  ⚠️ No subtype data. Skipping survival analysis.")
        return
    
    survival_data = clinical.dropna(subset=['OS_MONTHS', 'molecular_subtype'])
    survival_data = survival_data[survival_data['OS_MONTHS'] > 0]
    
    if len(survival_data) < 20:
        print(f"  ⚠️ Only {len(survival_data)} samples with survival data. Skipping.")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    kmf = KaplanMeierFitter()
    
    palette = {'EBV': '#e94560', 'MSI': '#0f3460', 'GS': '#16213e', 'CIN': '#533483'}
    default_colors = ['#e94560', '#0f3460', '#16213e', '#533483', '#2a9d8f', '#e76f51']
    
    for i, subtype in enumerate(survival_data['molecular_subtype'].unique()):
        mask = survival_data['molecular_subtype'] == subtype
        color = palette.get(subtype, default_colors[i % len(default_colors)])
        
        kmf.fit(
            survival_data.loc[mask, 'OS_MONTHS'],
            event_observed=survival_data.loc[mask, 'OS_STATUS'],
            label=f'{subtype} (n={mask.sum()})'
        )
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True, linewidth=2)
    
    # Log-rank test
    try:
        result = multivariate_logrank_test(
            survival_data['OS_MONTHS'],
            survival_data['molecular_subtype'],
            survival_data['OS_STATUS']
        )
        p_val = result.p_value
        ax.text(0.95, 0.05, f'Log-rank p = {p_val:.2e}',
                transform=ax.transAxes, ha='right', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except:
        pass
    
    ax.set_title('Overall Survival by Molecular Subtype', fontsize=14, fontweight='bold')
    ax.set_xlabel('Months', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "survival_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: survival_curves.png")


def plot_tmb_distribution():
    """Plot TMB distribution by subtype."""
    print("\n📊 Generating TMB Distribution Plot...")
    
    clinical = pd.read_csv("data/clinical_data.csv") if os.path.exists("data/clinical_data.csv") else None
    
    if clinical is None or 'TMB' not in clinical.columns or 'molecular_subtype' not in clinical.columns:
        print("  ⚠️ No TMB/subtype data. Skipping.")
        return
    
    data = clinical.dropna(subset=['TMB', 'molecular_subtype'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    palette = {'EBV': '#e94560', 'MSI': '#0f3460', 'GS': '#16213e', 'CIN': '#533483', 'POLE': '#2a9d8f'}
    # Only use colors for subtypes present in data
    present_subtypes = sorted(data['molecular_subtype'].unique())
    palette_filtered = {k: v for k, v in palette.items() if k in present_subtypes}
    
    sns.boxplot(data=data, x='molecular_subtype', y='TMB', palette=palette_filtered, ax=ax, 
                order=present_subtypes)
    sns.stripplot(data=data, x='molecular_subtype', y='TMB', color='black', alpha=0.3,
                  size=3, ax=ax, order=sorted(data['molecular_subtype'].unique()))
    
    ax.set_title('Tumor Mutation Burden by Molecular Subtype', fontsize=14, fontweight='bold')
    ax.set_xlabel('Molecular Subtype', fontsize=12)
    ax.set_ylabel('TMB (mutations/Mb)', fontsize=12)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tmb_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: tmb_distribution.png")


def main():
    print("=" * 60)
    print("Visualization & Interpretation")
    print("=" * 60)
    
    # 1. SHAP Analysis
    plot_shap_analysis()
    
    # 2. t-SNE / UMAP
    plot_dimensionality_reduction()
    
    # 3. Signature-Subtype Heatmap
    plot_signature_heatmap_by_subtype()
    
    # 4. Survival Curves
    plot_survival_analysis()
    
    # 5. TMB Distribution
    plot_tmb_distribution()
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 All figures saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
