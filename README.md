# PML 2026 Coursework Workspace

This workspace prepares data for the Probabilistic Machine Learning coursework on mosquito audio detection.

## Data layout

- `data/raw/humbugdb/`: HumBugDB 0.0.1 files from Zenodo record 4904800, including four zip archives and metadata CSV.
- `data/raw/ood_negative/data1/`: 70 OOD negative audio samples (y01.wav - y70.wav).
- `data/raw/ood_negative/data2/`: 100 OOD negative audio samples randomly segmented from origin.wav (y01.wav - y100.wav).
- `data/raw/ood_positive/vasconcelos/`: 1697 Aedes Aegypti mosquito audio samples at 8kHz sampling rate (Vasconcelos et al., 2020).
- `data/raw/ood_positive/other/`: Additional OOD positive samples (m1.wav - m10.wav).

## Scripts

### Process HumBugDB and extract MFCC features

```powershell
uv run python .\scripts\process_humbugdb_from_zip.py
```

Processes audio from HumBugDB zip files and extracts MFCC features. Supports configurable parameters:

- `--zip-dir`: Directory containing humbugdb zip files (default: `data/raw/humbugdb/`)
- `--csv`: Path to metadata CSV file (default: `data/raw/humbugdb/neurips_2021_zenodo_0_0_1.csv`)
- `--output`: Output directory for processed MFCC features (default: `data/processed/humbugdb_mfcc/`)
- `--n-mfcc`: Number of MFCC coefficients (default: 13)
- `--sample-rate`: Target sample rate (default: 8000)

### Plot label duration distributions

```powershell
uv run python .\scripts\plot_label_duration_distribution.py
```

Generates histograms of duration distributions for the three HumBugDB sound types (mosquito, background, audio).

- `--input`: Metadata CSV path (default: `data/raw/humbugdb_0_0_1/neurips_2021_zenodo_0_0_1.csv`; use `--input data/raw/humbugdb/neurips_2021_zenodo_0_0_1.csv` for current layout)
- `--output`: Output image path (default: `outputs/humbugdb_label_duration_distribution.png`)
- `--bins`: Number of histogram bins (default: 40)
- `--linear`: Use linear x-axis instead of logarithmic

## Data Sources

See `docs/data_sources.md` for detailed source URLs, licenses, and data-use notes.
