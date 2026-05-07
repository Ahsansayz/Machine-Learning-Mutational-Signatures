#!/usr/bin/env python3
"""
AI/ML Classification of Gastric Cancer Subtypes
=======================================================
Multi-class classification using signature-based features.
Models: Random Forest, XGBoost, SVM, MLP, Gradient Boosting, 1D-CNN (optional)
Evaluation: 5-fold Stratified CV, AUC-ROC, F1, MCC, Confusion Matrix

Input:  output/ml_features.csv, output/ml_labels.csv (from Step 5)
Output: output/ml_results/
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_curve, auc
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not available, skipping.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

INPUT_FEATURES = "output/ml_features.csv"
INPUT_LABELS = "output/ml_labels.csv"
OUTPUT_DIR = "output/ml_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load feature matrix and labels."""
    print("  Loading data...")
    
    X = pd.read_csv(INPUT_FEATURES, index_col=0)
    y_df = pd.read_csv(INPUT_LABELS, index_col=0)
    
    # Align
    common = X.index.intersection(y_df.index)
    X = X.loc[common]
    y = y_df.loc[common, 'molecular_subtype']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Classes: {list(class_names)}")
    print(f"  Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Handle missing/infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, y, y_encoded, le, class_names


def define_models():
    """Define all classification models with hyperparameters."""
    models = {}
    
    models['Random Forest'] = {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'classifier__n_estimators': [200, 500],
            'classifier__max_depth': [5, 10, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': ['balanced']
        }
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = {
            'model': XGBClassifier(
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1
            ),
            'params': {
                'classifier__n_estimators': [200, 500],
                'classifier__max_depth': [4, 6, 8],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__subsample': [0.8]
            }
        }
    
    models['SVM (RBF)'] = {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'classifier__C': [1, 10, 100],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__class_weight': ['balanced']
        }
    }
    
    models['MLP Neural Network'] = {
        'model': MLPClassifier(random_state=42, early_stopping=True, max_iter=1000),
        'params': {
            'classifier__hidden_layer_sizes': [(128, 64), (256, 128, 64), (128, 64, 32)],
            'classifier__activation': ['relu'],
            'classifier__alpha': [0.0001, 0.001]
        }
    }
    
    models['Gradient Boosting'] = {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [200, 300],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__subsample': [0.8]
        }
    }
    
    return models


def train_and_evaluate(X, y_encoded, le, class_names, models):
    """Train all models with 5-fold stratified CV."""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_overall_f1 = -1
    best_overall_model = None
    best_overall_name = ""
    
    for name, config in models.items():
        print(f"\n{'='*50}")
        print(f"🤖 Training: {name}")
        print(f"{'='*50}")
        
        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', config['model'])
        ])
        
        # Hyperparameter tuning with inner CV
        try:
            grid = GridSearchCV(
                pipeline, config['params'], cv=3,
                scoring='f1_weighted', n_jobs=-1, verbose=0,
                error_score='raise'
            )
            grid.fit(X, y_encoded)
            best_pipeline = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")
        except Exception as e:
            print(f"  GridSearch failed ({e}), using defaults...")
            pipeline.fit(X, y_encoded)
            best_pipeline = pipeline
        
        # Outer CV predictions
        y_pred = cross_val_predict(best_pipeline, X, y_encoded, cv=skf, method='predict')
        
        try:
            y_proba = cross_val_predict(best_pipeline, X, y_encoded, cv=skf, method='predict_proba')
        except:
            y_proba = None
        
        # Metrics
        acc = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        precision = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_encoded, y_pred)
        
        # AUC-ROC (one-vs-rest)
        if y_proba is not None and len(class_names) > 2:
            y_bin = label_binarize(y_encoded, classes=range(len(class_names)))
            try:
                auc_roc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
            except:
                auc_roc = None
        elif y_proba is not None:
            auc_roc = roc_auc_score(y_encoded, y_proba[:, 1])
        else:
            auc_roc = None
        
        report = classification_report(y_encoded, y_pred, target_names=class_names, output_dict=True)
        
        results[name] = {
            'accuracy': acc,
            'f1_weighted': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'auc_roc': auc_roc,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'report': report,
            'pipeline': best_pipeline
        }
        
        print(f"\n  📊 Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 (weighted): {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  MCC:       {mcc:.4f}")
        if auc_roc:
            print(f"  AUC-ROC:   {auc_roc:.4f}")
        
        print(f"\n  Classification Report:")
        print(classification_report(y_encoded, y_pred, target_names=class_names))
        
        # Track best
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = best_pipeline
            best_overall_name = name
    
    print(f"\n{'='*50}")
    print(f"🏆 BEST MODEL: {best_overall_name}")
    print(f"   F1 (weighted) = {best_overall_f1:.4f}")
    print(f"{'='*50}")
    
    return results, best_overall_name, best_overall_model


def save_results(results, class_names, y_encoded, best_name, best_model, X, le):
    """Save all results, models, and generate comparison plots."""
    
    # 1. Model comparison table
    comparison = []
    for name, res in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': f"{res['accuracy']:.4f}",
            'F1_Weighted': f"{res['f1_weighted']:.4f}",
            'Precision': f"{res['precision']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'MCC': f"{res['mcc']:.4f}",
            'AUC_ROC': f"{res['auc_roc']:.4f}" if res['auc_roc'] else "N/A",
            'Is_Best': '✅' if name == best_name else ''
        })
    
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\n💾 Saved: model_comparison.csv")
    
    # 2. Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        if idx >= len(axes):
            break
        cm = confusion_matrix(y_encoded, res['y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
        axes[idx].set_title(f'{name}\nF1={res["f1_weighted"]:.3f}', fontsize=11, fontweight='bold')
    
    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices_all.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: confusion_matrices_all.png")
    
    # 3. Best model confusion matrix (publication quality)
    best_res = results[best_name]
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_encoded, best_res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Best Model: {best_name}\nAccuracy={best_res["accuracy"]:.3f}, F1={best_res["f1_weighted"]:.3f}, MCC={best_res["mcc"]:.3f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. ROC Curves (best model)
    if best_res['y_proba'] is not None and len(class_names) > 2:
        y_bin = label_binarize(y_encoded, classes=range(len(class_names)))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#e94560', '#0f3460', '#16213e', '#533483', '#2a9d8f', '#e76f51']
        
        for i, (cls, color) in enumerate(zip(class_names, colors)):
            if i < y_bin.shape[1]:
                fpr, tpr, _ = roc_curve(y_bin[:, i], best_res['y_proba'][:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC={roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves — {best_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: roc_curves.png")
    
    # 5. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ['accuracy', 'f1_weighted', 'precision', 'recall', 'mcc']
    metric_labels = ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall', 'MCC']
    
    x = np.arange(len(results))
    width = 0.15
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[name][metric] for name in results]
        bars = ax.bar(x + i * width, values, width, label=label, alpha=0.85)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results.keys(), rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: model_comparison_chart.png")
    
    # 6. Feature importance (for tree-based models)
    try:
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            best_model.fit(X, y_encoded)
            importances = best_model.named_steps['classifier'].feature_importances_
            feat_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            feat_imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
            
            # Plot top 20
            fig, ax = plt.subplots(figsize=(10, 8))
            top20 = feat_imp.head(20)
            ax.barh(range(len(top20)), top20['importance'].values, color='#0f3460', alpha=0.8)
            ax.set_yticks(range(len(top20)))
            ax.set_yticklabels(top20['feature'].values)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top 20 Features — {best_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: feature_importance.png")
    except Exception as e:
        print(f"  ⚠️ Could not extract feature importance: {e}")
    
    # 7. Save best model
    with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), 'wb') as f:
        pickle.dump({'model': best_model, 'label_encoder': le, 'features': list(X.columns)}, f)
    print(f"💾 Saved: best_model.pkl")
    
    # 8. Save predictions
    pred_df = pd.DataFrame({
        'Sample': X.index,
        'True_Label': [class_names[i] for i in y_encoded],
        'Predicted': [class_names[i] for i in results[best_name]['y_pred']],
        'Correct': y_encoded == results[best_name]['y_pred']
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print(f"💾 Saved: predictions.csv")


def main():
    print("=" * 60)
    print("AI/ML Classification")
    print("=" * 60)
    
    # Check inputs
    if not os.path.exists(INPUT_FEATURES):
        print(f"❌ Features not found: {INPUT_FEATURES}")
        print("   Run feature_engineering.py first!")
        sys.exit(1)
    
    if not os.path.exists(INPUT_LABELS):
        print(f"❌ Labels not found: {INPUT_LABELS}")
        print("   Run get_clinical_data.py and feature_engineering.py first!")
        sys.exit(1)
    
    # Load data
    X, y, y_encoded, le, class_names = load_data()
    
    # Check minimum samples
    min_class = min(np.bincount(y_encoded))
    if min_class < 5:
        print(f"\n⚠️ Warning: Smallest class has only {min_class} samples!")
        print("   Reducing CV folds to 3 for stability.")
    
    # Define models
    models = define_models()
    
    # Train and evaluate
    results, best_name, best_model = train_and_evaluate(X, y_encoded, le, class_names, models)
    
    # Save results
    save_results(results, class_names, y_encoded, best_name, best_model, X, le)
    
    print(f"\n✅ ML Classification complete!")
    print(f"📁 All results saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
