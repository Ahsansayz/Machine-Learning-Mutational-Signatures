#!/usr/bin/env python3
"""
Feature Engineering
===========================
Builds the ML-ready feature matrix by combining:
1. COSMIC signature activity weights
2. TMB (Tumor Mutation Burden)
3. Engineered signature-based features
4. Clinical features

Input:  output/cosmic_assignment/cosmic_activities.csv (from Step 3)
        output/signatures/signature_activities.csv (from Step 2)
        data/clinical_data.csv (from Step 4)
Output: output/ml_features.csv, output/ml_labels.csv
"""

import os
import sys
import pandas as pd
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_signature_data():
    """Load signature activities — prefer COSMIC assignment, fall back to de novo."""
    
    # Try COSMIC activities first (preferred for ML)
    cosmic_file = "output/cosmic_assignment/cosmic_activities.csv"
    denovo_file = "output/signatures/signature_activities.csv"
    
    if os.path.exists(cosmic_file):
        print("  📊 Loading COSMIC signature activities...")
        activities = pd.read_csv(cosmic_file, index_col=0)
        source = "COSMIC"
    elif os.path.exists(denovo_file):
        print("  📊 Loading de novo signature activities...")
        activities = pd.read_csv(denovo_file, index_col=0)
        source = "de_novo"
    else:
        print("  ❌ No signature activities found!")
        print("  Run extract_signatures.py or cosmic_assignment.py first.")
        sys.exit(1)
    
    print(f"  Loaded {source} activities: {activities.shape[0]} samples × {activities.shape[1]} signatures")
    
    # Remove signatures with zero variance (uninformative)
    nonzero_sigs = activities.columns[activities.var() > 0]
    removed = activities.shape[1] - len(nonzero_sigs)
    if removed > 0:
        print(f"  Removed {removed} zero-variance signatures")
    activities = activities[nonzero_sigs]
    
    return activities, source


def load_clinical_data():
    """Load clinical data."""
    clinical_file = "data/clinical_data.csv"
    
    if os.path.exists(clinical_file):
        print("  📋 Loading clinical data...")
        clinical = pd.read_csv(clinical_file)
        print(f"  Loaded clinical data: {len(clinical)} rows")
        return clinical
    else:
        print("  ⚠️ No clinical data found. ML will use signature features only.")
        return None


def engineer_features(activities, clinical=None, source="COSMIC"):
    """Build the feature matrix with engineered features."""
    
    features = activities.copy()
    
    # ============================================================
    # 1. Normalize to relative proportions
    # ============================================================
    total_activity = features.sum(axis=1)
    proportions = features.div(total_activity + 1e-10, axis=0)
    proportions.columns = [f"{c}_prop" for c in proportions.columns]
    
    # ============================================================
    # 2. Engineered features (COSMIC-specific)
    # ============================================================
    if source == "COSMIC":
        eng = pd.DataFrame(index=features.index)
        
        # MSI signature burden (key for MSI subtype detection)
        msi_sigs = [s for s in ['SBS6', 'SBS14', 'SBS15', 'SBS20', 'SBS21', 'SBS26', 'SBS44'] 
                   if s in features.columns]
        if msi_sigs:
            eng['MSI_sig_burden'] = features[msi_sigs].sum(axis=1)
            eng['MSI_sig_burden_prop'] = proportions[[f"{s}_prop" for s in msi_sigs if f"{s}_prop" in proportions.columns]].sum(axis=1)
        
        # APOBEC burden (key for EBV subtype detection)
        apobec_sigs = [s for s in ['SBS2', 'SBS13'] if s in features.columns]
        if apobec_sigs:
            eng['APOBEC_burden'] = features[apobec_sigs].sum(axis=1)
            eng['APOBEC_burden_prop'] = proportions[[f"{s}_prop" for s in apobec_sigs if f"{s}_prop" in proportions.columns]].sum(axis=1)
        
        # Clock-like signatures (SBS1 + SBS5)
        clock_sigs = [s for s in ['SBS1', 'SBS5'] if s in features.columns]
        if clock_sigs:
            eng['clock_burden'] = features[clock_sigs].sum(axis=1)
        
        # HRD burden (SBS3 — homologous recombination deficiency)
        if 'SBS3' in features.columns:
            eng['HRD_burden'] = features['SBS3']
        
        # SBS17 burden (CIN-associated, common in GC)
        sbs17_sigs = [s for s in ['SBS17a', 'SBS17b'] if s in features.columns]
        if sbs17_sigs:
            eng['SBS17_burden'] = features[sbs17_sigs].sum(axis=1)
        
        # Ratio features
        if 'SBS1' in features.columns and 'SBS5' in features.columns:
            eng['SBS1_SBS5_ratio'] = features['SBS1'] / (features['SBS5'] + 1e-10)
        
        # Dominant signature
        eng['dominant_sig_prop'] = proportions.max(axis=1)
        eng['n_active_sigs'] = (proportions > 0.05).sum(axis=1)
        
        # Log-transform total mutation burden
        eng['log_total_activity'] = np.log1p(total_activity)
    else:
        eng = pd.DataFrame(index=features.index)
        eng['dominant_sig_prop'] = proportions.max(axis=1)
        eng['n_active_sigs'] = (proportions > 0.05).sum(axis=1)
        eng['log_total_activity'] = np.log1p(total_activity)
    
    # ============================================================
    # 3. Combine: raw activities + proportions + engineered
    # ============================================================
    feature_matrix = pd.concat([features, proportions, eng], axis=1)
    
    # ============================================================
    # 4. Add clinical features (if available)
    # ============================================================
    if clinical is not None and len(clinical) > 0:
        # Map clinical data to signature samples
        # Need to match TCGA barcodes
        if 'full_barcode' in clinical.columns:
            clinical_indexed = clinical.set_index('full_barcode')
        elif 'patient_id' in clinical.columns:
            clinical_indexed = clinical.drop_duplicates('patient_id').set_index('patient_id')
        else:
            clinical_indexed = pd.DataFrame()
        
        # Try to match
        matched = 0
        if not clinical_indexed.empty:
            # Add TMB
            if 'TMB' in clinical_indexed.columns:
                tmb_map = {}
                for sample in feature_matrix.index:
                    # Try full barcode match
                    if sample in clinical_indexed.index:
                        tmb_map[sample] = clinical_indexed.loc[sample, 'TMB']
                    else:
                        # Try patient ID match (first 12 chars)
                        pid = sample[:12]
                        if pid in clinical_indexed.index:
                            tmb_map[sample] = clinical_indexed.loc[pid, 'TMB']
                
                if tmb_map:
                    feature_matrix['TMB'] = feature_matrix.index.map(tmb_map)
                    feature_matrix['log_TMB'] = np.log1p(feature_matrix['TMB'].fillna(0))
                    matched = sum(1 for v in tmb_map.values() if pd.notna(v))
                    print(f"  Added TMB for {matched} samples")
            
            # Add age
            if 'age_years' in clinical_indexed.columns:
                age_map = {}
                for sample in feature_matrix.index:
                    pid = sample[:12]
                    if pid in clinical_indexed.index:
                        age_map[sample] = clinical_indexed.loc[pid, 'age_years']
                
                if age_map:
                    feature_matrix['age_years'] = feature_matrix.index.map(age_map)
            
            # Add gender (encoded)
            if 'gender' in clinical_indexed.columns:
                gender_map = {}
                for sample in feature_matrix.index:
                    pid = sample[:12]
                    if pid in clinical_indexed.index:
                        g = clinical_indexed.loc[pid, 'gender']
                        gender_map[sample] = 1 if str(g).lower() == 'male' else 0
                
                if gender_map:
                    feature_matrix['gender_male'] = feature_matrix.index.map(gender_map)
    
    # Fill NaN with 0 (for missing clinical data)
    feature_matrix = feature_matrix.fillna(0)
    
    return feature_matrix


def extract_labels(clinical):
    """Extract molecular subtype labels for supervised learning."""
    
    if clinical is None:
        return None
    
    if 'molecular_subtype' not in clinical.columns:
        print("  ⚠️ No molecular_subtype column in clinical data")
        return None
    
    labels = clinical.dropna(subset=['molecular_subtype'])[['full_barcode', 'molecular_subtype']]
    labels = labels.drop_duplicates('full_barcode').set_index('full_barcode')
    
    if len(labels) == 0:
        return None
    
    print(f"\n  📋 Label Distribution:")
    for subtype, count in labels['molecular_subtype'].value_counts().items():
        print(f"    {subtype}: {count}")
    
    return labels


def main():
    print("=" * 60)
    print("Feature Engineering")
    print("=" * 60)
    
    # Load data
    activities, source = load_signature_data()
    clinical = load_clinical_data()
    
    # Engineer features
    feature_matrix = engineer_features(activities, clinical, source)
    
    # Extract labels
    labels = extract_labels(clinical) if clinical is not None else None
    
    # Align features and labels
    if labels is not None:
        common = feature_matrix.index.intersection(labels.index)
        if len(common) == 0:
            # Try matching by patient ID
            print("  Trying patient ID matching for labels...")
            feature_pids = {idx: idx[:12] for idx in feature_matrix.index}
            label_pids = {idx: idx[:12] for idx in labels.index}
            
            # Build reverse map
            pid_to_label = {v: labels.loc[k, 'molecular_subtype'] for k, v in label_pids.items()}
            
            matched_labels = {}
            for feat_idx, pid in feature_pids.items():
                if pid in pid_to_label:
                    matched_labels[feat_idx] = pid_to_label[pid]
            
            if matched_labels:
                labels = pd.DataFrame.from_dict(matched_labels, orient='index', columns=['molecular_subtype'])
                common = feature_matrix.index.intersection(labels.index)
        
        if len(common) > 0:
            feature_matrix_labeled = feature_matrix.loc[common]
            labels_aligned = labels.loc[common]
            
            print(f"\n  Labeled samples: {len(common)} / {len(feature_matrix)}")
            
            # Save labeled dataset
            feature_matrix_labeled.to_csv(os.path.join(OUTPUT_DIR, "ml_features.csv"))
            labels_aligned.to_csv(os.path.join(OUTPUT_DIR, "ml_labels.csv"))
            
            print(f"  💾 Saved: {OUTPUT_DIR}/ml_features.csv")
            print(f"  💾 Saved: {OUTPUT_DIR}/ml_labels.csv")
        else:
            print("  ⚠️ No labels could be matched to features")
            feature_matrix.to_csv(os.path.join(OUTPUT_DIR, "ml_features.csv"))
            print(f"  💾 Saved unlabeled features: {OUTPUT_DIR}/ml_features.csv")
    else:
        feature_matrix.to_csv(os.path.join(OUTPUT_DIR, "ml_features.csv"))
        print(f"\n  💾 Saved features (no labels): {OUTPUT_DIR}/ml_features.csv")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"📊 Feature Matrix Summary")
    print(f"{'=' * 60}")
    print(f"  Samples:  {feature_matrix.shape[0]}")
    print(f"  Features: {feature_matrix.shape[1]}")
    print(f"\n  Feature categories:")
    print(f"    Raw signature activities: {activities.shape[1]}")
    print(f"    Signature proportions:    {activities.shape[1]}")
    print(f"    Engineered features:      {feature_matrix.shape[1] - 2*activities.shape[1]}")
    
    print(f"\n✅ Feature engineering complete!")


if __name__ == "__main__":
    main()
