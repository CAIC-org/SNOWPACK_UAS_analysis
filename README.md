# SNOWPACK_UAS_analysis

Automated workflow for correcting UAS-derived snow Digital Surface Models (DSM) using virtual Ground Control Points (vGCPs)

## Overview

This workflow corrects systematic errors in UAS DSMs of snow-covered terrain using a three-tier approach with automatic tier selection. The system validates corrections using Leave-One-Out cross-validation and bootstrap uncertainty estimation.

### Three-Tier Correction Approach

- **Tier 1 (PPK-only)**: No correction applied. Used when PPK accuracy is sufficient.
- **Tier 2 (Vertical Shift Correction)**: Removes uniform vertical bias by subtracting mean error across all vGCPs.
- **Tier 3 (Planar Trend Correction)**: Fits and removes a planar trend surface to correct spatially variable errors.

The workflow automatically selects the best correction tier based on configurable RMSE thresholds and improvement metrics.

## Features

- Automatic correction tier selection based on validation statistics
- Three-tier correction system (PPK-only, vertical shift, planar trend)
- Leave-One-Out cross-validation for accuracy assessment
- Bootstrap uncertainty estimation with confidence intervals
- Single file and batch processing modes
- Comprehensive statistical output and validation plots
- Standalone validation tool for post-processing accuracy checks
- YAML-based configuration for reproducibility
- Ability to choose which correction to apply foregoing the automatic selection
- Run correction and validation or only validation

## Requirements

### System Requirements

- Python 3.9 or higher
- GDAL library (system installation required)

### Python Dependencies

- rasterio >= 1.3.0
- geopandas >= 0.12.0
- numpy >= 1.21.0
- pandas >= 1.5.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- pyyaml >= 6.0

## Installation

### Option 1: Conda (Recommended)

Conda handles GDAL installation automatically. This is preferred for Windows. 

```bash
# Clone repository
git clone https://github.com/yourusername/SNOWPACK_UAS_analysis.git
cd SNOWPACK_UAS_analysis

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate uas_HS
```

### Option 2: Pip (Linux/Mac)

For Linux and Mac systems, install GDAL system libraries first, then use pip.

```bash
# Linux: Install GDAL system libraries
sudo apt-get install gdal-bin libgdal-dev

# Mac: Install GDAL via Homebrew
brew install gdal

# Check GDAL version
gdalinfo --version

# Clone repository
git clone https://github.com/yourusername/SNOWPACK_UAS_analysis.git
cd SNOWPACK_UAS_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install Python packages
pip install -r requirements.txt

# Install GDAL matching your system version
pip install GDAL==X.X.X  # Replace X.X.X with your GDAL version
```

## Data Structure

User project folder should be organized as follows:

```
project_folder/
├── config.yaml                    # Configuration file
├── uas_snow_correction.py         # Main correction script
├── uas_validation.py              # Standalone validation tool
├── data/
│   └── {aoi_name}/                # Area of Interest folder
│       ├── bareGround/
│       │   └── bare_dsm.tif       # Bare-ground reference DSM
│       ├── vGCP/
│       │   └── vgcp.shp           # Virtual Ground Control Points
│       └── snowOn/
│           └── snow_*.tif         # Snow-on DSM(s) to correct
└── outputs/
    └── {aoi_name}/                # Results folder (auto-created)
        ├── corrected/             # Corrected DSMs and snow depth
        ├── statistics/            # CSV statistics files
        └── plots/                 # Validation plots
```
Note: The project_folder can be named whatever the user wants, it does not need to be "project_folder". 

## Input Data Requirements

### 1. Bare-Ground DSM
- Format: GeoTIFF (.tif or .tiff)
- Must be in same CRS as snow-on DSMs
- Should represent snow-free terrain conditions
- Location: `data/{aoi_name}/bareGround/`

### 2. Snow-On DSM(s)
- Format: GeoTIFF (.tif or .tiff)
- Must be in same CRS as bare-ground DSM
- PPK-corrected DSMs output from Esri SiteScan (or alternative software)
- Location: `data/{aoi_name}/snowOn/`

### 3. Virtual Ground Control Points (vGCP)
- Format: Shapefile (.shp)
- Required fields:
  - `E`: Easting coordinate
  - `N`: Northing coordinate
  - `Elevation`: Bare-ground elevation (in same vertical datum as DSMs)
- Must be in same CRS as DSMs
- Minimum 3 points required (10+ recommended for robust correction, planar trend correction requires a minimum of 3 points)
- Location: `data/{aoi_name}/vGCP/`

## Configuration

Edit `config.yaml` to specify your data paths and processing parameters:

```yaml
# Correction thresholds
thresholds:
  tier1_rmse_max: 0.2          # Max RMSE for Tier 1 (meters)
  tier2_improvement_min: 0.20   # Min improvement for Tier 2 (20%)
  correction_mode: "auto"       # auto, ppk, vsc, ptc

# Validation settings
validation:
  run_bootstrap: true
  bootstrap_iterations: 250

# Coordinate Reference System
crs:
  horizontal: "EPSG:6342"       # NAD83(2011) UTM Zone 13N
  vertical: "NAVD88"

# Data paths
paths:
  aoi_name: "Widowmaker"
  bare_ground_file: "bare_ground_dsm.tif"
  vgcp_file: "vgcp_points.shp"
  snow_on_file: ""              # Single file mode
  snow_on_folder: "."           # Batch mode (any non-empty string)
```

### Correction Modes

- `auto`: Automatically selects best tier (default)
- `ppk` or `tier1` or `none`: Force Tier 1 (no correction)
- `vsc` or `tier2` or `vertical`: Force Tier 2 (vertical shift)
- `ptc` or `tier3` or `planar` or `tilt`: Force Tier 3 (planar trend)

## Usage

### Snow DSM Correction

#### Single File Mode

Process one snow-on DSM:

```bash
# Edit config.yaml:
# - Set snow_on_file: "user_dsm.tif"
# - Set snow_on_folder: ""

python uas_snow_correction.py --config config.yaml
```

#### Batch Mode

Process all DSMs in the snowOn folder:

```bash
# Edit config.yaml:
# - Set snow_on_file: ""
# - Set snow_on_folder: "."

python uas_snow_correction.py --config config.yaml
```

Or use the `--batch` flag to override config:

```bash
python uas_snow_correction.py --config config.yaml --batch
```

#### Additional Options

```bash
# Skip bootstrap (faster processing)
python uas_snow_correction.py --config config.yaml --no-bootstrap

# Verbose output (detailed logging)
python uas_snow_correction.py --config config.yaml --verbose

# Override AOI name from config
python uas_snow_correction.py --config config.yaml --aoi MySite

# Combine options
python uas_snow_correction.py --config config.yaml --batch --no-bootstrap --verbose
```

### Validation Only

Use the validation tool to check accuracy after correction:

#### Single File Validation

```bash
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_corrected.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation_results.csv \
  --bootstrap 250 \
  --plot-dir validation_plots/
```

#### Batch Validation

```bash
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/ \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output batch_validation.csv \
  --bootstrap 250 \
  --plot-dir validation_plots/
```

#### Validation Options

```bash
# Without bootstrap (faster)
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv

# With custom label
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv --label "Site_A_Feb"

# Verbose logging
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv --verbose
```

## Output Files

### Corrected DSMs and Snow Depth

- `{filename}_Tier{N}_DSM.tif`: Corrected snow-on DSM
- `{filename}_Tier{N}_snowHeight.tif`: Snow height raster (corrected DSM - bare-ground DSM)

### Statistics

- `{filename}_statistics.csv`: Comprehensive statistics for all correction tiers
- `batch_summary.csv`: Summary statistics for batch processing

### Plots

- `{filename}_residuals.png`: Residual distributions and spatial patterns
- `{filename}_bootstrap.png`: Bootstrap RMSE distribution with confidence intervals

## Interpretation Guide

### Key Statistics

- **RMSE (Root Mean Square Error)**: Overall magnitude of errors. Lower is better.
- **ME (Mean Error)**: Average bias. Should be close to 0 after correction. Note that there could still be issues that indicate tilt in the data 
if the ME is close to zero but the individual residual values have a high range (e.g. -4, 0.2, 3, -0.1).
- **MAE (Mean Absolute Error)**: Average magnitude of errors, less sensitive to outliers.
- **NMAD (Normalized Median Absolute Deviation)**: Robust measure of spread.
- **StdDev (Standard Deviation)**: Variability of errors around the mean.

### Correction Tier Selection Logic

The workflow uses these decision rules:

1. If Tier 1 RMSE < `tier1_rmse_max` → Use Tier 1 (no correction needed)
2. Else, if Tier 2 improves RMSE by > `tier2_improvement_min` → Use Tier 2
3. Else, if Tier 3 improves RMSE by > `tier2_improvement_min` → Use Tier 3
4. Otherwise → Use Tier 1 (corrections don't help)

### Validation Metrics

- **LOO (Leave-One-Out) RMSE**: Cross-validated accuracy estimate
- **Bootstrap Mean RMSE**: Average RMSE from resampling
- **Bootstrap 95% CI**: Confidence interval for RMSE estimate

## Troubleshooting

### GDAL Installation Issues

**Windows**: Use conda instead of pip. GDAL is difficult to install via pip on Windows.

**Linux/Mac**: Ensure system GDAL libraries are installed and Python GDAL version matches system version.

### CRS Mismatch Errors

Ensure all input files (bare-ground DSM, snow-on DSM, vGCP shapefile) are in the same coordinate reference system. Reproject if necessary using ArcGIS Pro, QGIS, or GDAL tools. The workflow does not handle incorrect CRS.  

### Insufficient vGCP Points

Minimum 3 vGCPs required. 

### High RMSE After Correction

- Check vGCP quality and distribution
- Verify DSM alignment and coregistration (i.e. have the same geographic area)
- Consider using manual correction mode to force specific tier
- Inspect residual plots for spatial patterns

## Author

Valerie Foley  
Email: valerie.foley@state.co.us
