#!/usr/bin/env python3
"""
Build the 96-channel SBS Trinucleotide Mutation Matrix
=============================================================
Reads GDC masked somatic mutation MAF files and builds the standard
96-channel Single Base Substitution (SBS) matrix using trinucleotide context.

The 96 channels = 6 substitution types × 16 trinucleotide contexts:
  Substitutions: C>A, C>G, C>T, T>A, T>C, T>G
  Each with 4×4 = 16 flanking base combinations (5'base × 3'base)

The CONTEXT column in GDC MAF files contains the trinucleotide sequence.

Input:  maf_files/*.maf (434 GDC masked somatic mutation MAF files)
Output: output/sbs96_matrix.csv (434 samples × 96 mutation types)
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
MAF_DIR = "maf_files"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sbs96_matrix.csv")

# ============================================================
# Define the standard 96 SBS mutation types in canonical order
# ============================================================
# Pyrimidine-based convention: mutations are always expressed
# relative to the pyrimidine (C or T) of the mutated base pair.
# So A>C becomes T>G on the complementary strand, etc.

PURINES = {'A', 'G'}
PYRIMIDINES = {'C', 'T'}
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
BASES = ['A', 'C', 'G', 'T']

# 6 substitution types (pyrimidine convention)
SUB_TYPES = ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']

def get_96_channels():
    """Generate the 96 SBS channel labels in standard order."""
    channels = []
    for sub in SUB_TYPES:
        ref = sub[0]  # C or T
        for five_prime in BASES:
            for three_prime in BASES:
                label = f"{five_prime}[{sub}]{three_prime}"
                channels.append(label)
    return channels

SBS96_CHANNELS = get_96_channels()
print(f"Total SBS96 channels: {len(SBS96_CHANNELS)}")


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq))


def classify_mutation(ref, alt, context):
    """
    Classify a single SNP into one of 96 SBS channels.
    
    Args:
        ref: Reference allele (single base)
        alt: Tumor alternate allele (single base)
        context: Trinucleotide context string from MAF CONTEXT column
                 (e.g., 'AACGATCG...' — center base is the mutated position)
    
    Returns:
        SBS96 channel label (e.g., 'A[C>T]G') or None if invalid
    """
    # Validate inputs
    if ref not in 'ACGT' or alt not in 'ACGT' or ref == alt:
        return None
    
    if context is None or len(context) < 3:
        return None
    
    # The CONTEXT column in GDC MAF typically contains a longer sequence
    # The center base is the mutated position
    # We need the base immediately before and after
    mid = len(context) // 2
    five_prime = context[mid - 1].upper()
    three_prime = context[mid + 1].upper()
    center = context[mid].upper()
    
    # Validate
    if five_prime not in 'ACGT' or three_prime not in 'ACGT':
        return None
    
    # If ref is a purine (A or G), convert to pyrimidine convention
    if ref in PURINES:
        ref = COMPLEMENT[ref]
        alt = COMPLEMENT[alt]
        five_prime, three_prime = COMPLEMENT[three_prime], COMPLEMENT[five_prime]
    
    sub = f"{ref}>{alt}"
    label = f"{five_prime}[{sub}]{three_prime}"
    
    return label


def process_maf_file(filepath):
    """
    Process a single MAF file and return SBS96 mutation counts.
    
    Returns:
        tuple: (sample_name, Counter of SBS96 channel counts)
    """
    try:
        # Read MAF file, skip comment lines
        df = pd.read_csv(filepath, sep='\t', comment='#', 
                         usecols=['Tumor_Sample_Barcode', 'Variant_Type', 
                                  'Reference_Allele', 'Tumor_Seq_Allele2', 'CONTEXT'],
                         low_memory=False)
    except Exception as e:
        print(f"  ⚠️ Error reading {os.path.basename(filepath)}: {e}")
        return None, Counter()
    
    # Filter to SNPs only (Single Nucleotide Polymorphisms)
    snps = df[df['Variant_Type'] == 'SNP'].copy()
    
    if len(snps) == 0:
        sample = df['Tumor_Sample_Barcode'].iloc[0] if len(df) > 0 else os.path.basename(filepath)
        return sample, Counter()
    
    # Get sample name (TCGA barcode)
    sample = snps['Tumor_Sample_Barcode'].iloc[0]
    
    # Classify each SNP
    counts = Counter()
    for _, row in snps.iterrows():
        ref = str(row['Reference_Allele']).upper()
        alt = str(row['Tumor_Seq_Allele2']).upper()
        context = str(row.get('CONTEXT', ''))
        
        channel = classify_mutation(ref, alt, context)
        if channel and channel in SBS96_CHANNELS:
            counts[channel] += 1
    
    return sample, counts


def main():
    print("=" * 60)
    print("Building 96-Channel SBS Trinucleotide Matrix")
    print("=" * 60)
    
    # Get all MAF files
    maf_files = sorted([f for f in os.listdir(MAF_DIR) if f.endswith('.maf')])
    print(f"\n📁 Found {len(maf_files)} MAF files in {MAF_DIR}/")
    
    if len(maf_files) == 0:
        print("❌ No MAF files found!")
        sys.exit(1)
    
    # Process each MAF file
    results = {}
    skipped = 0
    
    for filename in tqdm(maf_files, desc="Processing MAF files"):
        filepath = os.path.join(MAF_DIR, filename)
        sample, counts = process_maf_file(filepath)
        
        if sample is None:
            skipped += 1
            continue
        
        # Use TCGA barcode (first 15 chars) as sample ID for uniqueness
        # Full barcode: TCGA-CD-A4MJ-01A-11D-A25D-08
        # Patient ID:   TCGA-CD-A4MJ (first 12 chars)
        # Sample ID:    TCGA-CD-A4MJ-01A (first 15 chars)
        results[sample] = counts
    
    print(f"\n✅ Processed {len(results)} samples, skipped {skipped}")
    
    # Build the matrix with all 96 channels
    matrix = pd.DataFrame.from_dict(results, orient='index')
    
    # Ensure all 96 channels are present and in correct order
    for ch in SBS96_CHANNELS:
        if ch not in matrix.columns:
            matrix[ch] = 0
    
    matrix = matrix[SBS96_CHANNELS].fillna(0).astype(int)
    matrix.index.name = 'Sample'
    
    # Sort by total mutation count (descending)
    matrix['Total'] = matrix.sum(axis=1)
    matrix = matrix.sort_values('Total', ascending=False)
    total_col = matrix.pop('Total')
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matrix.to_csv(OUTPUT_FILE)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 SBS96 Matrix Summary")
    print(f"{'=' * 60}")
    print(f"  Samples:          {matrix.shape[0]}")
    print(f"  Mutation types:   {matrix.shape[1]} (should be 96)")
    print(f"  Total mutations:  {matrix.values.sum():,}")
    print(f"  Median per sample: {int(total_col.median()):,}")
    print(f"  Min per sample:   {int(total_col.min()):,}")
    print(f"  Max per sample:   {int(total_col.max()):,}")
    print(f"\n  Top 5 mutation types:")
    top5 = matrix.sum().sort_values(ascending=False).head(5)
    for ch, count in top5.items():
        print(f"    {ch}: {int(count):,}")
    
    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    print(f"✅ SBS96 Matrix build complete!")
    
    return matrix


if __name__ == "__main__":
    matrix = main()
