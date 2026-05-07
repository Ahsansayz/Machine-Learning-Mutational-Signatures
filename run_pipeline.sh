#!/bin/bash
# ==============================================================
# Gastric Cancer Mutational Signature Analysis Pipeline
# Master Runner Script
# ==============================================================
# Usage: bash run_pipeline.sh [step_number]
#   bash run_pipeline.sh        # Run all steps
#   bash run_pipeline.sh 1      # Run only Step 1
#   bash run_pipeline.sh 3-6    # Run Steps 3 to 6
# ==============================================================

set -e

# Activate conda if available
if [ -f "/home/param/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/param/miniconda3/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🧬 Gastric Cancer Mutational Signature Pipeline        ║"
echo "║  AI-Driven Classification of TCGA-STAD Subtypes         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"
STEP=$1

run_step() {
    local num=$1
    local name=$2
    local script=$3
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Step $num: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -f "$script" ]; then
        python "$script"
        echo ""
        echo "  ✅ Step $num complete"
    else
        echo "  ❌ Script not found: $script"
        exit 1
    fi
}

should_run() {
    local step=$1
    if [ -z "$STEP" ]; then
        return 0  # Run all
    fi
    
    # Handle range (e.g., "3-6")
    if [[ "$STEP" == *-* ]]; then
        local start=$(echo "$STEP" | cut -d'-' -f1)
        local end=$(echo "$STEP" | cut -d'-' -f2)
        [ "$step" -ge "$start" ] && [ "$step" -le "$end" ]
        return $?
    fi
    
    # Handle single step
    [ "$step" -eq "$STEP" ]
    return $?
}

START_TIME=$(date +%s)

# Step 1: Build SBS96 Matrix
if should_run 1; then
    run_step 1 "Build 96-Channel SBS Trinucleotide Matrix" "build_sbs96_matrix.py"
fi

# Step 2: Extract Signatures
if should_run 2; then
    run_step 2 "De Novo Signature Extraction (NMF)" "extract_signatures.py"
fi

# Step 3: COSMIC Assignment
if should_run 3; then
    run_step 3 "COSMIC v3.4 Signature Assignment" "cosmic_assignment.py"
fi

# Step 4: Clinical Data
if should_run 4; then
    run_step 4 "Clinical Data Download & Integration" "get_clinical_data.py"
fi

# Step 5: Feature Engineering
if should_run 5; then
    run_step 5 "Feature Engineering" "feature_engineering.py"
fi

# Step 6: ML Classification
if should_run 6; then
    run_step 6 "AI/ML Classification" "ml_classification.py"
fi

# Step 7: Visualization
if should_run 7; then
    run_step 7 "Visualization & Interpretation" "visualize.py"
fi

# Step 8: Report
if should_run 8; then
    run_step 8 "Generate HTML Report" "generate_report.py"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🎉 PIPELINE COMPLETE!                                 ║"
echo "║  Total time: ${MINUTES}m ${SECONDS}s                              ║"
echo "║                                                          ║"
echo "║  📁 Outputs:                                            ║"
echo "║    output/sbs96_matrix.csv       (SBS96 matrix)         ║"
echo "║    output/signatures/            (Extracted signatures)  ║"
echo "║    output/cosmic_assignment/     (COSMIC mapping)        ║"
echo "║    output/ml_results/            (ML models & metrics)   ║"
echo "║    output/figures/               (Publication figures)   ║"
echo "║    output/report.html            (Full HTML report)      ║"
echo "╚══════════════════════════════════════════════════════════╝"
