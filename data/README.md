# Data directory

This directory is the repository for raw data.

- **eegmmidb_raw/** — place PhysioNet EEGMMIDB EDF files here.  The audit and preprocessing scripts will parse EDF headers, verify sampling rates, extract channel names, and reconstruct labels.
- **milimbeeg_raw/** — place MILimbEEG CSV files here.  Each CSV should include channel data and metadata encoded in the file path or header.

The audit scripts will traverse these directories and produce CSV summaries in `outputs/audits/`.  Processed data and subject‑wise splits will be stored in `outputs/` as part of later phases.