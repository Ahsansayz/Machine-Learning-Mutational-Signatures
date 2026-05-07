#!/usr/bin/env python3
"""
Clinical Data Download & Integration
=============================================
Downloads TCGA-STAD clinical data including molecular subtypes (EBV/MSI/GS/CIN),
survival data, and demographic information.

Maps MAF file UUIDs → TCGA barcodes → clinical attributes.

Input:  maf_files/*.maf (to extract TCGA barcodes)
Output: data/clinical_data.csv
"""

import os
import sys
import json
import pandas as pd
import numpy as np

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "clinical_data.csv")
MAF_DIR = "maf_files"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_barcodes_from_mafs():
    """Extract TCGA barcodes from MAF files."""
    print("  Extracting TCGA barcodes from MAF files...")
    
    barcodes = {}
    maf_files = sorted([f for f in os.listdir(MAF_DIR) if f.endswith('.maf')])
    
    for filename in maf_files:
        filepath = os.path.join(MAF_DIR, filename)
        try:
            df = pd.read_csv(filepath, sep='\t', comment='#', 
                             usecols=['Tumor_Sample_Barcode'], nrows=1)
            barcode = df['Tumor_Sample_Barcode'].iloc[0]
            # Patient ID = first 12 chars: TCGA-XX-XXXX
            patient_id = barcode[:12]
            # Sample ID = first 15 chars: TCGA-XX-XXXX-01A
            sample_id = barcode[:15]
            barcodes[filename] = {
                'full_barcode': barcode,
                'patient_id': patient_id,
                'sample_id': sample_id
            }
        except Exception as e:
            print(f"    ⚠️ Error reading {filename}: {e}")
    
    print(f"  Found {len(barcodes)} unique samples")
    return barcodes


def download_clinical_from_gdc(patient_ids):
    """Download clinical data from GDC API."""
    import urllib.request
    
    print("\n  Downloading clinical data from GDC API...")
    
    # GDC API endpoint for clinical data
    url = "https://api.gdc.cancer.gov/cases"
    
    all_clinical = []
    batch_size = 100
    patient_list = list(set(patient_ids))
    
    for i in range(0, len(patient_list), batch_size):
        batch = patient_list[i:i+batch_size]
        
        # Build filter
        filters = {
            "op": "in",
            "content": {
                "field": "submitter_id",
                "value": batch
            }
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join([
                "submitter_id",
                "demographic.gender",
                "demographic.race",
                "demographic.ethnicity",
                "demographic.year_of_birth",
                "demographic.vital_status",
                "demographic.days_to_death",
                "diagnoses.age_at_diagnosis",
                "diagnoses.tumor_stage",
                "diagnoses.primary_diagnosis",
                "diagnoses.site_of_resection_or_biopsy",
                "diagnoses.days_to_last_follow_up",
                "diagnoses.morphology",
                "diagnoses.tissue_or_organ_of_origin",
                "project.project_id"
            ]),
            "size": str(batch_size),
            "format": "json"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?{query_string}"
        
        try:
            req = urllib.request.Request(full_url)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
            
            for hit in data.get('data', {}).get('hits', []):
                clinical = {'patient_id': hit.get('submitter_id', '')}
                
                # Demographics
                demo = hit.get('demographic', {})
                clinical['gender'] = demo.get('gender', '')
                clinical['race'] = demo.get('race', '')
                clinical['vital_status'] = demo.get('vital_status', '')
                clinical['days_to_death'] = demo.get('days_to_death', '')
                
                # Diagnosis
                diags = hit.get('diagnoses', [{}])
                if diags:
                    diag = diags[0]
                    clinical['age_at_diagnosis'] = diag.get('age_at_diagnosis', '')
                    clinical['tumor_stage'] = diag.get('tumor_stage', '')
                    clinical['primary_diagnosis'] = diag.get('primary_diagnosis', '')
                    clinical['days_to_last_follow_up'] = diag.get('days_to_last_follow_up', '')
                
                # Project
                clinical['project'] = hit.get('project', {}).get('project_id', '')
                
                all_clinical.append(clinical)
                
        except Exception as e:
            print(f"    ⚠️ GDC API batch {i//batch_size + 1} error: {e}")
    
    print(f"  Downloaded clinical data for {len(all_clinical)} patients")
    return all_clinical


def download_subtypes_from_cbio():
    """
    Download molecular subtype data from cBioPortal.
    The TCGA-STAD paper (Nature 2014) defined 4 subtypes: EBV, MSI, GS, CIN
    """
    import urllib.request
    
    print("\n  Downloading molecular subtypes from cBioPortal...")
    
    # Try to get clinical data with subtypes
    url = "https://www.cbioportal.org/api/clinical-data/fetch?clinicalDataType=SAMPLE&projection=SUMMARY"
    
    # Clinical attribute IDs that might contain subtype info
    subtype_attrs = [
        "SUBTYPE",
        "MOLECULAR_SUBTYPE", 
        "TCGA_SUBTYPE",
        "CANCER_TYPE_DETAILED",
        "MSI_STATUS",
        "MSI_SCORE_MANTIS",
        "TMB_NONSYNONYMOUS"
    ]
    
    body = {
        "attributeIds": subtype_attrs,
        "studyViewFilter": {
            "studyIds": ["stad_tcga_pan_can_atlas_2018", "stad_tcga"]
        }
    }
    
    try:
        req_data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        
        # Parse into DataFrame
        subtype_data = {}
        for item in data:
            sample_id = item.get('sampleId', '')
            patient_id = item.get('patientId', '')
            attr = item.get('clinicalAttributeId', '')
            value = item.get('value', '')
            
            key = patient_id or sample_id
            if key not in subtype_data:
                subtype_data[key] = {'patient_id': key}
            subtype_data[key][attr] = value
        
        print(f"  Got subtype data for {len(subtype_data)} samples")
        return list(subtype_data.values())
        
    except Exception as e:
        print(f"  ⚠️ cBioPortal API error: {e}")
        return []


def download_tcga_subtypes_supplementary():
    """
    Download TCGA-STAD subtypes from the supplementary data of the TCGA paper.
    This is the most reliable source for molecular subtypes.
    """
    import urllib.request
    
    print("\n  Attempting to download TCGA supplementary subtype data...")
    
    # PanCancer Atlas subtype data
    url = "https://api.gdc.cancer.gov/cases"
    
    # Alternative: direct download of TCGA published subtypes
    # The subtypes were published in Nature 2014, Table S1
    
    # Try cBioPortal's clinical data endpoint (more reliable)
    try:
        url = "https://www.cbioportal.org/api/clinical-data/fetch?clinicalDataType=PATIENT&projection=SUMMARY"
        body = {
            "attributeIds": ["SUBTYPE"],
            "studyViewFilter": {
                "studyIds": ["stad_tcga_pan_can_atlas_2018"]
            }
        }
        
        req_data = json.dumps(body).encode('utf-8')
        req = urllib.request.Request(url, data=req_data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        
        subtypes = {}
        for item in data:
            pid = item.get('patientId', '')
            subtypes[pid] = item.get('value', '')
        
        print(f"  Got subtypes for {len(subtypes)} patients")
        return subtypes
    except Exception as e:
        print(f"  ⚠️ Could not download subtypes: {e}")
        return {}


def create_clinical_labels_from_signatures():
    """
    If we can't get official subtypes, create proxy labels from signature data.
    This uses known signature-subtype associations:
    - MSI: high SBS6/SBS14/SBS15/SBS20/SBS21/SBS26/SBS44
    - EBV: high SBS2/SBS13 (APOBEC)
    - CIN: high SBS17a/SBS17b
    - GS: primarily clock-like (SBS1/SBS5)
    """
    print("\n  ⚠️ Creating proxy subtype labels from mutation characteristics...")
    print("  (Will be updated with official labels if available)")
    
    # Check if we have COSMIC activities
    cosmic_file = "output/cosmic_assignment/cosmic_activities.csv"
    if os.path.exists(cosmic_file):
        activities = pd.read_csv(cosmic_file, index_col=0)
        
        # Normalize
        props = activities.div(activities.sum(axis=1) + 1e-10, axis=0)
        
        subtypes = {}
        for sample in props.index:
            row = props.loc[sample]
            
            # MSI signatures
            msi_sigs = [s for s in ['SBS6', 'SBS14', 'SBS15', 'SBS20', 'SBS21', 'SBS26', 'SBS44'] 
                       if s in row.index]
            msi_burden = row[msi_sigs].sum() if msi_sigs else 0
            
            # APOBEC/EBV signatures
            apobec_sigs = [s for s in ['SBS2', 'SBS13'] if s in row.index]
            apobec_burden = row[apobec_sigs].sum() if apobec_sigs else 0
            
            # CIN signatures (SBS17)
            cin_sigs = [s for s in ['SBS17a', 'SBS17b'] if s in row.index]
            cin_burden = row[cin_sigs].sum() if cin_sigs else 0
            
            # Classify
            if msi_burden > 0.3:
                subtypes[sample] = 'MSI'
            elif apobec_burden > 0.2:
                subtypes[sample] = 'EBV'
            elif cin_burden > 0.15:
                subtypes[sample] = 'CIN'
            else:
                subtypes[sample] = 'GS'
        
        return subtypes
    
    return {}


def generate_tmb(maf_dir):
    """Calculate Tumor Mutation Burden for each sample."""
    print("\n  Calculating TMB (Tumor Mutation Burden)...")
    
    tmb = {}
    exome_size_mb = 30.0  # Approximate WES capture size in Mb
    
    maf_files = [f for f in os.listdir(maf_dir) if f.endswith('.maf')]
    
    for filename in maf_files:
        filepath = os.path.join(maf_dir, filename)
        try:
            df = pd.read_csv(filepath, sep='\t', comment='#',
                             usecols=['Tumor_Sample_Barcode', 'Variant_Classification', 'Variant_Type'])
            
            barcode = df['Tumor_Sample_Barcode'].iloc[0]
            
            # Count non-synonymous mutations for TMB
            nonsyn_classes = [
                'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del',
                'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins',
                'Splice_Site', 'Nonstop_Mutation', 'Translation_Start_Site'
            ]
            
            nonsyn_count = df[df['Variant_Classification'].isin(nonsyn_classes)].shape[0]
            total_count = df.shape[0]
            
            tmb[barcode] = {
                'total_mutations': total_count,
                'nonsynonymous_mutations': nonsyn_count,
                'TMB': nonsyn_count / exome_size_mb
            }
        except:
            pass
    
    print(f"  Calculated TMB for {len(tmb)} samples")
    return tmb


def main():
    print("=" * 60)
    print("Clinical Data Download & Integration")
    print("=" * 60)
    
    # 1. Extract barcodes from MAF files
    barcodes = extract_barcodes_from_mafs()
    barcode_df = pd.DataFrame.from_dict(barcodes, orient='index')
    barcode_df.index.name = 'maf_file'
    
    patient_ids = barcode_df['patient_id'].unique().tolist()
    print(f"  Unique patients: {len(patient_ids)}")
    
    # 2. Download clinical data from GDC
    clinical_data = download_clinical_from_gdc(patient_ids)
    
    # 3. Download subtypes
    subtypes = download_tcga_subtypes_supplementary()
    
    # 4. Download additional subtype info from cBioPortal
    cbio_data = download_subtypes_from_cbio()
    
    # 5. Calculate TMB
    tmb_data = generate_tmb(MAF_DIR)
    
    # 6. Merge all data
    clinical_df = pd.DataFrame(clinical_data) if clinical_data else pd.DataFrame()
    
    # Create master table
    master = barcode_df.reset_index()
    
    # Add clinical data
    if not clinical_df.empty:
        master = master.merge(clinical_df, on='patient_id', how='left')
    
    # Add subtypes
    if subtypes:
        master['molecular_subtype'] = master['patient_id'].map(subtypes)
    
    # Add cBioPortal data  
    if cbio_data:
        cbio_df = pd.DataFrame(cbio_data)
        if 'SUBTYPE' in cbio_df.columns:
            subtype_map = dict(zip(cbio_df['patient_id'], cbio_df['SUBTYPE']))
            if 'molecular_subtype' not in master.columns:
                master['molecular_subtype'] = master['patient_id'].map(subtype_map)
            else:
                # Fill NaN values
                mask = master['molecular_subtype'].isna()
                master.loc[mask, 'molecular_subtype'] = master.loc[mask, 'patient_id'].map(subtype_map)
        
        if 'MSI_STATUS' in cbio_df.columns:
            msi_map = dict(zip(cbio_df['patient_id'], cbio_df['MSI_STATUS']))
            master['msi_status'] = master['patient_id'].map(msi_map)
    
    # Add TMB
    tmb_df = pd.DataFrame.from_dict(tmb_data, orient='index')
    if not tmb_df.empty:
        master = master.merge(tmb_df, left_on='full_barcode', right_index=True, how='left')
    
    # If no official subtypes available, create proxy labels
    if 'molecular_subtype' not in master.columns or master['molecular_subtype'].isna().all():
        proxy_subtypes = create_clinical_labels_from_signatures()
        if proxy_subtypes:
            master['molecular_subtype'] = master['full_barcode'].map(proxy_subtypes)
            master['subtype_source'] = 'signature_proxy'
            print("\n  ⚠️ Using proxy subtypes from signature profiles")
        else:
            print("\n  ⚠️ No subtype labels available. Will use unsupervised approach in ML step.")
    else:
        master['subtype_source'] = 'official_TCGA'
    
    # Convert age from days to years
    if 'age_at_diagnosis' in master.columns:
        master['age_years'] = pd.to_numeric(master['age_at_diagnosis'], errors='coerce') / 365.25
    
    # Save
    master.to_csv(OUTPUT_FILE, index=False)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"📊 Clinical Data Summary")
    print(f"{'=' * 60}")
    print(f"  Total samples: {len(master)}")
    print(f"  Unique patients: {master['patient_id'].nunique()}")
    
    if 'molecular_subtype' in master.columns:
        print(f"\n  Molecular Subtype Distribution:")
        for subtype, count in master['molecular_subtype'].value_counts().items():
            print(f"    {subtype}: {count}")
    
    if 'gender' in master.columns:
        print(f"\n  Gender Distribution:")
        for g, count in master['gender'].value_counts().items():
            print(f"    {g}: {count}")
    
    if 'TMB' in master.columns:
        print(f"\n  TMB Statistics:")
        print(f"    Mean: {master['TMB'].mean():.1f} mut/Mb")
        print(f"    Median: {master['TMB'].median():.1f} mut/Mb")
    
    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    print(f"✅ Clinical data integration complete!")
    
    return master


if __name__ == "__main__":
    main()
