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
- Per-point residual export for detailed analysis

## Prerequisites

Before installation, ensure you have:

- **Git**: Required for cloning the repository
  - Windows: Download from https://git-scm.com/download/windows
  - Mac: Install via Homebrew (`brew install git`) or download from git-scm.com
  - Linux: `sudo apt-get install git` (Ubuntu/Debian) or `sudo yum install git` (RHEL/CentOS)
  - Verify installation: `git --version`
  
- **Conda** (recommended for Windows) or Python 3.9+
  - Download Anaconda or Miniconda from https://docs.conda.io/en/latest/miniconda.html

- **ArcGIS Pro** (for snow depth calculation)
  - Required for the snow height calculator script
  - Includes Spatial Analyst extension

**Note:** After installing Git, you may need to restart your terminal or computer for the PATH to update.

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

### Option 1: Conda (Recommended for Windows)

Conda handles GDAL installation automatically.

```bash
# Clone repository
git clone https://github.com/CAIC-org/SNOWPACK_UAS_analysis.git
cd SNOWPACK_UAS_analysis

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate uas_HS
```

**If conda gets stuck on "Solving environment"**, see the [Conda Troubleshooting](#conda-solving-environment-stuck) section below.

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
git clone https://github.com/CAIC-org/SNOWPACK_UAS_analysis.git
cd SNOWPACK_UAS_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install Python packages
pip install -r requirements.txt

# Install GDAL matching your system version
pip install GDAL==X.X.X  # Replace X.X.X with your GDAL version
```

### Alternative: Download Without Git

If you prefer not to use Git:

1. Go to the repository on GitHub
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file to your desired location
5. Open terminal/PowerShell in the extracted folder
6. Continue with conda or pip installation steps above

## Data Structure

User project folder should be organized as follows:

```
project_folder/
├── config.yaml                    # Configuration file
├── uas_snow_correction.py         # Main correction script
├── uas_snow_height_calculator.py  # Snow depth calculation script (ArcGIS Pro)
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
        ├── corrected/             # Corrected DSMs
        ├── snowHeight/            # Snow depth rasters
        ├── statistics/            # CSV statistics files
        └── plots/                 # Validation plots
```
Note: The project_folder will likely be called SNOWPACK_UAS_analysis after the user clones the repo from git.

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

The workflow is split into two scripts that must be run sequentially:

### Step 1: Snow DSM Correction

The correction script automatically includes validation (LOO cross-validation and bootstrap uncertainty estimation).

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

### Step 2: Snow Depth Calculation

After correcting your DSMs, run the snow height calculator from ArcGIS Pro Python Command Prompt. This script uses ArcPy to properly handle raster alignment when subtracting the bare ground DSM from the corrected snow-on DSMs.

**How to access ArcGIS Pro Python Command Prompt:**
1. Open ArcGIS Pro
2. Go to Project tab -> Python -> Python Command Prompt
3. Or search Windows start menu for "Python Command Prompt (ArcGIS Pro)"

```bash
# Navigate to your project directory
cd path\to\SNOWPACK_UAS_analysis

# Run the snow height calculator
python uas_snow_height_calculator.py --config config.yaml

# With verbose output
python uas_snow_height_calculator.py --config config.yaml --verbose

# Override AOI name
python uas_snow_height_calculator.py --config config.yaml --aoi MySite
```

The snow height calculator will:
- Read all corrected DSMs from `outputs/{aoi_name}/corrected/`
- Read the bare ground DSM from your config
- Calculate snow depth using ArcPy Raster Calculator
- Save results to `outputs/{aoi_name}/snowHeight/`

**Important:** The snow height calculator must be run from ArcGIS Pro Python Command Prompt. It will not work in a standard Python environment because it requires ArcPy.

### Standalone Validation

Use the validation tool when you need to:
- Validate DSMs that were corrected outside this workflow
- Re-run validation with different parameters (e.g., more bootstrap iterations)
- Validate DSMs from other sources

#### Single File Validation

```bash
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_corrected.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation_results.csv \
  --bootstrap 250 \
  --plot-dir validation_plots/

# With per-point residuals export
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_corrected.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation_results.csv \
  --bootstrap 250 \
  --save-residuals
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

# With per-point residuals export
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/ \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output batch_validation.csv \
  --bootstrap 250 \
  --save-residuals
```

#### Validation Options

```bash
# Without bootstrap (faster)
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv

# With custom label
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv --label "Site_A_Feb"

# With per-point residuals
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv --save-residuals

# Verbose logging
python uas_validation.py --bare bare.tif --snow snow.tif --vgcp vgcp.shp --output stats.csv --verbose
```

**Note:** When using `--save-residuals`, per-point residuals are saved to a `point_residuals/` folder in the same directory as the output CSV.

## Output Files

### Corrected DSMs (from uas_snow_correction.py)

- `{filename}_noCorr_DSM.tif`: Tier 1 corrected DSM (PPK-only, no correction)
- `{filename}_VSC_DSM.tif`: Tier 2 corrected DSM (Vertical Shift Correction)
- `{filename}_PTC_DSM.tif`: Tier 3 corrected DSM (Planar Trend Correction)

### Snow Depth Rasters (from uas_snow_height_calculator.py)

- `{filename}_noCorr_snowHeight.tif`: Snow depth from Tier 1
- `{filename}_VSC_snowHeight.tif`: Snow depth from Tier 2
- `{filename}_PTC_snowHeight.tif`: Snow depth from Tier 3

### Statistics

- `{filename}_statistics.csv`: Summary statistics for all evaluated tiers
  - Includes statistics for all evaluated tiers (Tier 1, Tier 2, and/or Tier 3)
  - Shows which tier was selected and why
  - Contains LOO cross-validation metrics
  - Includes bootstrap uncertainty estimates (if enabled)
  
- `{filename}_point_residuals.csv`: Per-point residuals for each evaluated tier
  - Point_ID: Sequential point identifier (1, 2, 3, ...)
  - E, N: Easting and Northing coordinates
  - Z_bare: Bare-ground elevation from vGCP
  - Z_snow_ppk: PPK-corrected snow-on elevation
  - Tier1_Residual: Residual for Tier 1 (PPK-only)
  - Tier2_Residual: Residual for Tier 2 (if evaluated)
  - Tier3_Residual: Residual for Tier 3 (if evaluated)
  - Selected_Tier: Which tier was selected by the algorithm
  
- `batch_summary.csv`: Summary statistics for batch processing

### Plots

- `{filename}_residual_comparison.png`: Point-by-point comparison of residuals before and after correction
- `{filename}_tier_comparison.png`: Bar chart comparing RMSE across evaluated tiers

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

### Using Per-Point Residuals

The point residuals CSV allows detailed inspection of:
- Which vGCPs have the largest errors
- Spatial patterns in residuals
- How each correction tier affects individual points
- Quality control and outlier detection

## Troubleshooting

### Conda "Solving Environment" Stuck

If `conda env create -f environment.yml` gets stuck on "Solving environment", try these solutions:

#### Solution 1: Use Mamba (Fastest)

Mamba is a faster conda alternative:

```bash
# Cancel the stuck process with Ctrl+C

# Install mamba
conda install mamba -n base -c conda-forge

# Use mamba instead of conda
mamba env create -f environment.yml
conda activate uas_HS
```

#### Solution 2: Manual Environment Creation

Create the environment step-by-step:

```bash
# Create basic environment
conda create -n uas_HS python=3.9
conda activate uas_HS

# Install packages from conda-forge
conda install -c conda-forge gdal rasterio geopandas numpy pandas matplotlib seaborn pyyaml
```

#### Solution 3: Hybrid Approach (Conda + Pip)

Use conda for GDAL only, pip for everything else:

```bash
# Create environment with Python and GDAL
conda create -n uas_HS python=3.9
conda activate uas_HS
conda install -c conda-forge gdal

# Install remaining packages with pip
pip install rasterio geopandas numpy pandas matplotlib seaborn pyyaml
```

#### Solution 4: Clean Conda Cache

Sometimes conda's cache gets corrupted:

```bash
conda clean --all
conda env create -f environment.yml
```

#### Solution 5: Set Strict Channel Priority

```bash
conda config --set channel_priority strict
conda env create -f environment.yml
```

### Git Installation Issues

**"git is not recognized" error:**

1. Verify Git is installed: Check if `C:\Program Files\Git\cmd\git.exe` exists
2. Add to PATH: Add `C:\Program Files\Git\cmd` to your system PATH (see Installation Prerequisites)
3. Restart terminal: Close and reopen PowerShell/terminal completely
4. Restart computer: If PATH still doesn't update, restart your computer

**Alternative:** Use Git Bash (comes with Git for Windows) or download the repository as a ZIP file.

### GDAL Installation Issues

**Windows**: Use conda instead of pip. GDAL is difficult to install via pip on Windows.

**Linux/Mac**: Ensure system GDAL libraries are installed and Python GDAL version matches system version.

### ArcGIS Pro Issues

**ArcPy not available error:**

The snow height calculator must be run from ArcGIS Pro Python Command Prompt. To access it:

1. Open ArcGIS Pro
2. Go to Project tab -> Python -> Python Command Prompt
3. Or search Windows start menu for "Python Command Prompt (ArcGIS Pro)"

**Spatial Analyst not available:**

Ensure you have the Spatial Analyst extension enabled in ArcGIS Pro. The snow height calculator will automatically check out the extension when it runs.

### CRS Mismatch Errors

Ensure all input files (bare-ground DSM, snow-on DSM, vGCP shapefile) are in the same coordinate reference system. Reproject if necessary using ArcGIS Pro, QGIS, or GDAL tools. The workflow does not handle incorrect CRS.  

### Insufficient vGCP Points

Minimum 3 vGCPs required for planar trend correction. 10+ recommended for robust results.

### High RMSE After Correction

- Check vGCP quality and distribution
- Verify DSM alignment and coregistration (i.e. have the same geographic area)
- Consider using manual correction mode to force specific tier
- Inspect residual plots for spatial patterns
- Review per-point residuals CSV to identify problematic points

## Author

Valerie Foley  
Email: valerie.foley@state.co.us
