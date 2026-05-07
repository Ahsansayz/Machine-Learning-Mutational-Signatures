# Data Directory

This folder is intended for raw and intermediate data files required by the Gastric Cancer Mutational Signatures pipeline.

## Required Data:
1. **Raw MAF files:** Place your somatic mutation data (Mutation Annotation Format) here. These will be parsed by `build_sbs96_matrix.py`.
2. **Clinical Data:** If you are using custom clinical data instead of fetching it via `get_clinical_data.py`, place your CSV/TSV files here.

*Note: Large genomic datasets should NOT be committed to version control. Add `*.maf` and `*.csv` to your `.gitignore` file before pushing.*
