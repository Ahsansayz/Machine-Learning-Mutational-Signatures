#!/usr/bin/env python3
"""
Generate Comprehensive HTML Report
===========================================
Creates a professional HTML report summarizing all results from the pipeline.

Input:  All outputs from Steps 1-7
Output: output/report.html
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import base64

OUTPUT_FILE = "output/report.html"


def img_to_base64(path):
    """Embed image as base64 in HTML."""
    if not os.path.exists(path):
        return ""
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = path.split('.')[-1]
    return f'<img src="data:image/{ext};base64,{data}" style="max-width:100%; border-radius:8px; margin:10px 0;">'


def generate_report():
    """Generate the comprehensive HTML report."""
    
    # Collect data from all steps
    n_samples = 0
    n_signatures = 0
    best_model = "N/A"
    best_f1 = "N/A"
    best_acc = "N/A"
    
    # Step 1 data
    sbs96_info = ""
    if os.path.exists("output/sbs96_matrix.csv"):
        matrix = pd.read_csv("output/sbs96_matrix.csv", index_col=0)
        n_samples = matrix.shape[0]
        total_muts = int(matrix.values.sum())
        sbs96_info = f"""
        <div class="stat-card">
            <h3>96-Channel SBS Matrix</h3>
            <p><strong>Samples:</strong> {n_samples}</p>
            <p><strong>Channels:</strong> {matrix.shape[1]}</p>
            <p><strong>Total mutations:</strong> {total_muts:,}</p>
            <p><strong>Median per sample:</strong> {int(matrix.sum(axis=1).median()):,}</p>
        </div>
        """
    
    # Step 2 data
    sig_info = ""
    if os.path.exists("output/signatures/signature_profiles.csv"):
        profiles = pd.read_csv("output/signatures/signature_profiles.csv", index_col=0)
        n_signatures = profiles.shape[0]
        sig_info = f"""
        <div class="stat-card">
            <h3>De Novo Signatures</h3>
            <p><strong>Signatures extracted:</strong> {n_signatures}</p>
        </div>
        """
    
    # Step 3 data
    cosmic_info = ""
    if os.path.exists("output/cosmic_assignment/cosmic_activities.csv"):
        cosmic = pd.read_csv("output/cosmic_assignment/cosmic_activities.csv", index_col=0)
        active_sigs = cosmic.columns[cosmic.sum() > 0].tolist()
        cosmic_info = f"""
        <div class="stat-card">
            <h3>COSMIC Assignment</h3>
            <p><strong>Active COSMIC signatures:</strong> {len(active_sigs)}</p>
            <p><strong>Top signatures:</strong> {', '.join(active_sigs[:10])}</p>
        </div>
        """
    
    # Step 6 data
    ml_info = ""
    model_table = ""
    if os.path.exists("output/ml_results/model_comparison.csv"):
        comp = pd.read_csv("output/ml_results/model_comparison.csv")
        best_row = comp[comp['Is_Best'] == '✅'].iloc[0] if '✅' in comp['Is_Best'].values else comp.iloc[0]
        best_model = best_row['Model']
        best_f1 = best_row['F1_Weighted']
        best_acc = best_row['Accuracy']
        
        model_table = comp.to_html(index=False, classes='results-table', border=0)
        ml_info = f"""
        <div class="stat-card highlight">
            <h3>🏆 Best Model: {best_model}</h3>
            <p><strong>Accuracy:</strong> {best_acc}</p>
            <p><strong>F1 (Weighted):</strong> {best_f1}</p>
            <p><strong>MCC:</strong> {best_row.get('MCC', 'N/A')}</p>
            <p><strong>AUC-ROC:</strong> {best_row.get('AUC_ROC', 'N/A')}</p>
        </div>
        """
    
    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gastric Cancer Mutational Signature Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
            color: #e0e0e0;
            line-height: 1.6;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #e94560, #533483);
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(233, 69, 96, 0.3);
        }}
        header h1 {{
            font-size: 2.2em;
            color: white;
            margin-bottom: 10px;
        }}
        header p {{
            color: rgba(255,255,255,0.8);
            font-size: 1.1em;
        }}
        .section {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        .section h2 {{
            color: #e94560;
            font-size: 1.6em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(233, 69, 96, 0.3);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card.highlight {{
            border-color: #e94560;
            background: rgba(233, 69, 96, 0.1);
        }}
        .stat-card h3 {{
            color: #4ade80;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        .stat-card.highlight h3 {{
            color: #e94560;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .results-table th {{
            background: rgba(233, 69, 96, 0.2);
            color: #e94560;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }}
        .results-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .results-table tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }}
        .figure-box {{
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        .figure-box p {{
            color: #aaa;
            font-size: 0.9em;
            margin-top: 8px;
        }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}
        img {{ max-width: 100%; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧬 AI-Driven Mutational Signature Analysis</h1>
            <p>Classification of Gastric Cancer Molecular Subtypes Using NGS Data</p>
            <p style="margin-top:10px; font-size:0.9em;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </header>

        <!-- Summary Stats -->
        <div class="section">
            <h2>📊 Pipeline Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Dataset</h3>
                    <p><strong>Source:</strong> TCGA-STAD (GDC)</p>
                    <p><strong>Samples:</strong> {n_samples}</p>
                    <p><strong>Data type:</strong> WES (Masked Somatic MAF)</p>
                </div>
                {sbs96_info}
                {sig_info}
                {cosmic_info}
                {ml_info}
            </div>
        </div>

        <!-- Signature Extraction -->
        <div class="section">
            <h2>🔬 Mutational Signature Analysis</h2>
            <div class="figure-grid">
                <div class="figure-box">
                    {img_to_base64("output/signatures/rank_selection.png")}
                    <p>Optimal rank selection — reconstruction error and stability</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/signatures/signature_profiles_96ch.png")}
                    <p>96-channel SBS profiles of extracted signatures</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/signatures/signature_activities_heatmap.png")}
                    <p>Signature activity heatmap across all samples</p>
                </div>
            </div>
        </div>

        <!-- COSMIC Assignment -->
        <div class="section">
            <h2>🌐 COSMIC Signature Assignment</h2>
            <div class="figure-grid">
                <div class="figure-box">
                    {img_to_base64("output/cosmic_assignment/cosmic_signature_landscape.png")}
                    <p>COSMIC signature landscape — relative contributions per sample</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/cosmic_assignment/cosine_similarity_distribution.png")}
                    <p>Reconstruction quality — cosine similarity distribution</p>
                </div>
            </div>
        </div>

        <!-- ML Classification -->
        <div class="section">
            <h2>🤖 AI/ML Classification Results</h2>
            <h3 style="color:#4ade80; margin-bottom:15px;">Model Performance Comparison</h3>
            {model_table}
            <div class="figure-grid" style="margin-top:20px;">
                <div class="figure-box">
                    {img_to_base64("output/ml_results/best_confusion_matrix.png")}
                    <p>Best model confusion matrix</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/ml_results/roc_curves.png")}
                    <p>Multi-class ROC curves (One-vs-Rest)</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/ml_results/model_comparison_chart.png")}
                    <p>Performance comparison across all models</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/ml_results/feature_importance.png")}
                    <p>Top features by importance</p>
                </div>
            </div>
        </div>

        <!-- Explainability -->
        <div class="section">
            <h2>🔍 Model Interpretation & Explainability</h2>
            <div class="figure-grid">
                <div class="figure-box">
                    {img_to_base64("output/figures/shap_summary.png")}
                    <p>SHAP feature importance — which signatures drive classification</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/figures/shap_bar.png")}
                    <p>Mean |SHAP| values by subtype</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/figures/tsne_umap.png")}
                    <p>t-SNE and UMAP dimensionality reduction</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/figures/signature_subtype_heatmap.png")}
                    <p>Signature contribution heatmap by molecular subtype</p>
                </div>
            </div>
        </div>

        <!-- Survival Analysis -->
        <div class="section">
            <h2>📈 Clinical Validation</h2>
            <div class="figure-grid">
                <div class="figure-box">
                    {img_to_base64("output/figures/survival_curves.png")}
                    <p>Kaplan-Meier overall survival by molecular subtype</p>
                </div>
                <div class="figure-box">
                    {img_to_base64("output/figures/tmb_distribution.png")}
                    <p>Tumor Mutation Burden distribution by subtype</p>
                </div>
            </div>
        </div>

        <footer>
            <p>Generated by Gastric Cancer Mutational Signature Pipeline</p>
            <p>TCGA-STAD | COSMIC v3.4 | SigProfiler | scikit-learn | XGBoost</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html


def main():
    print("=" * 60)
    print("Generate Comprehensive HTML Report")
    print("=" * 60)
    
    html = generate_report()
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(html)
    
    print(f"\n💾 Report saved: {OUTPUT_FILE}")
    print(f"   Open in browser: file://{os.path.abspath(OUTPUT_FILE)}")
    print(f"\n✅ Report generation complete!")


if __name__ == "__main__":
    main()
