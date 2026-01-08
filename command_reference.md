# Command Reference

Quick reference guide for all commands and options in the SNOWPACK_UAS_analysis workflow.

## Table of Contents

- [Snow DSM Correction (uas_snow_correction.py)](#snow-dsm-correction)
- [Standalone Validation (uas_validation.py)](#standalone-validation)
- [Configuration File (config.yaml)](#configuration-file)
- [Common Workflows](#common-workflows)

---

## Snow-on DSM Correction

Main script for correcting UAS-derived snow-on DSMs using virtual ground control points (vGCPs).

### Basic Syntax

```bash
python uas_snow_correction.py --config <config_file> [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--config CONFIG` | Path to YAML configuration file |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batch` | Process all DSMs in folder (batch mode) | Single file mode |
| `--no-bootstrap` | Skip bootstrap uncertainty analysis (faster) | Bootstrap enabled |
| `--aoi AOI_NAME` | Override AOI name from config | Uses config value |
| `--verbose` | Enable detailed logging output | INFO level |

### Processing Modes

#### Single File Mode

Process one specific DSM file:

```bash
# Configure in config.yaml:
# snow_on_file: "your_dsm.tif"
# snow_on_folder: ""

python uas_snow_correction.py --config config.yaml
```

#### Batch Mode

Process all DSMs in the snowOn folder:

```bash
# Configure in config.yaml:
# snow_on_file: ""
# snow_on_folder: "."

python uas_snow_correction.py --config config.yaml --batch
```

### Examples

```bash
# Basic single file processing
python uas_snow_correction.py --config config.yaml

# Batch process with bootstrap
python uas_snow_correction.py --config config.yaml --batch

# Fast batch processing (no bootstrap)
python uas_snow_correction.py --config config.yaml --batch --no-bootstrap

# Override AOI name
python uas_snow_correction.py --config config.yaml --aoi SiteB

# Verbose output for debugging
python uas_snow_correction.py --config config.yaml --verbose

# Combine multiple options
python uas_snow_correction.py --config config.yaml --batch --no-bootstrap --verbose --aoi TestSite
```

### Output Files

Generated in `outputs/{aoi_name}/`:

```
corrected/
  ├── {filename}_Tier1_DSM.tif          # PPK-only corrected DSM
  ├── {filename}_Tier1_snowHeight.tif   # Snow depth from Tier 1
  ├── {filename}_Tier2_DSM.tif          # Vertical shift corrected DSM
  ├── {filename}_Tier2_snowHeight.tif   # Snow depth from Tier 2
  ├── {filename}_Tier3_DSM.tif          # Planar trend corrected DSM
  └── {filename}_Tier3_snowHeight.tif   # Snow depth from Tier 3

statistics/
  ├── {filename}_statistics.csv         # Summary statistics (RMSE, ME, etc.)
  ├── {filename}_point_residuals.csv    # Per-point residuals for all tiers
  └── batch_summary.csv                 # Batch processing summary

plots/
  ├── {filename}_residuals.png          # Per-point residuals visualization
  └── {filename}_bootstrap.png          # Bootstrap distribution
```

**Note on Output Files:** 
- Summary statistics CSV contains aggregate metrics (RMSE, ME, MAE, NMAD) for each evaluated tier
- Point residuals CSV contains individual residuals for each vGCP point for all evaluated tiers
- Only tiers that were evaluated will have corresponding residual columns in the point residuals CSV

---

## Standalone Validation

Independent validation tool for accuracy assessment of corrected DSMs.

### Basic Syntax

```bash
python uas_validation.py --bare <bare_dsm> --snow <snow_dsm> --vgcp <vgcp_shp> --output <output_csv> [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--bare BARE_DSM` | Path to bare-ground reference DSM |
| `--snow SNOW_DSM` | Path to snow-on DSM file or folder |
| `--vgcp VGCP_SHP` | Path to vGCP shapefile |
| `--output OUTPUT_CSV` | Path to output CSV file |

### Optional Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--bootstrap N` | Number of bootstrap iterations (0=disabled) | 0 |
| `--plot-dir DIR` | Directory for validation plots | No plots |
| `--label LABEL` | Label for output files | "Validation" |
| `--save-residuals` | Save per-point residuals to CSV | Disabled |
| `--verbose` | Enable detailed logging | INFO level |

### Examples

#### Single File Validation

```bash
# Basic validation without bootstrap
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_Tier2_DSM.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation.csv

# Full validation with bootstrap and plots
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_Tier2_DSM.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation.csv \
  --bootstrap 250 \
  --plot-dir validation_plots/ \
  --label SiteA_Dec15

# With per-point residuals export
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_Tier2_DSM.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation.csv \
  --bootstrap 250 \
  --save-residuals

# Verbose output for debugging
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_Tier2_DSM.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output validation.csv \
  --verbose
```

#### Batch Validation

```bash
# Validate all corrected DSMs in a folder
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/ \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output batch_validation.csv \
  --bootstrap 250 \
  --plot-dir validation_plots/

# Batch with per-point residuals
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/ \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output batch_validation.csv \
  --bootstrap 250 \
  --save-residuals

# Quick batch validation (no bootstrap)
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/ \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output batch_validation.csv
```

### Output Files

```
{output_csv}                    # Statistics CSV file

{plot_dir}/                     # If --plot-dir specified
  ├── {label}_residuals.png     # Residual analysis
  └── {label}_bootstrap.png     # Bootstrap distribution (if enabled)

point_residuals/                # If --save-residuals specified
  └── {label}_point_residuals.csv  # Per-point residuals
```

**Note:** Per-point residuals are saved to a `point_residuals/` folder in the same directory as the output CSV when using `--save-residuals` flag.

### CSV Output Columns

#### uas_snow_correction.py Statistics CSV

The statistics CSV from the correction script contains:

| Column | Description |
|--------|-------------|
| `DSM` | Filename of input snow-on DSM |
| `AOI` | Area of Interest name |
| `Selected_Tier` | Tier chosen by algorithm (Tier1/Tier2/Tier3) |
| `Reason` | Explanation for tier selection |
| `n_vGCPs` | Number of validation points used |
| `Tier1_ME` | Tier 1 Mean Error (meters) |
| `Tier1_RMSE` | Tier 1 Root Mean Square Error (meters) |
| `Tier1_NMAD` | Tier 1 Normalized Median Absolute Deviation |
| `Tier1_MAE` | Tier 1 Mean Absolute Error (meters) |
| `Tier2_ME` | Tier 2 Mean Error (meters) |
| `Tier2_RMSE` | Tier 2 Root Mean Square Error (meters) |
| `Tier2_NMAD` | Tier 2 NMAD |
| `Tier2_MAE` | Tier 2 Mean Absolute Error (meters) |
| `Tier3_ME` | Tier 3 Mean Error (meters) |
| `Tier3_RMSE` | Tier 3 Root Mean Square Error (meters) |
| `Tier3_NMAD` | Tier 3 NMAD |
| `Tier3_MAE` | Tier 3 Mean Absolute Error (meters) |
| `LOO_RMSE` | Leave-One-Out RMSE for selected tier |
| `LOO_ME` | Leave-One-Out Mean Error for selected tier |
| `Bootstrap_Mean` | Bootstrap mean RMSE (if enabled) |
| `Bootstrap_CI_Lower` | Bootstrap 95% CI lower bound |
| `Bootstrap_CI_Upper` | Bootstrap 95% CI upper bound |

#### uas_snow_correction.py Point Residuals CSV

The point residuals CSV contains per-point data for all evaluated tiers:

| Column | Description |
|--------|-------------|
| `Point_ID` | Sequential point identifier (1, 2, 3, ...) |
| `E` | Easting coordinate |
| `N` | Northing coordinate |
| `Z_bare` | Bare-ground elevation from vGCP |
| `Z_snow_ppk` | PPK-corrected snow-on elevation |
| `Tier1_Residual` | Residual for Tier 1 (PPK-only) |
| `Tier2_Residual` | Residual for Tier 2 (if evaluated) |
| `Tier3_Residual` | Residual for Tier 3 (if evaluated) |
| `Selected_Tier` | Which tier was selected (Tier1/Tier2/Tier3) |

**Note:** Only tiers that were evaluated will have corresponding residual columns. For example, if Tier 1 RMSE is below the threshold, only Tier1_Residual will be present.

#### uas_validation.py Statistics CSV

The standalone validation tool produces a statistics CSV with:

| Column | Description |
|--------|-------------|
| `Label` | User-specified or filename-derived label |
| `DSM` | Filename of validated DSM |
| `n` | Number of validation points |
| `ME` | Mean Error (meters) |
| `RMSE` | Root Mean Square Error (meters) |
| `MAE` | Mean Absolute Error (meters) |
| `StdDev` | Standard deviation (meters) |
| `Min` | Minimum residual (meters) |
| `Max` | Maximum residual (meters) |
| `NMAD` | Normalized Median Absolute Deviation |
| `LOO_n` | Leave-One-Out sample size |
| `LOO_ME` | LOO Mean Error |
| `LOO_RMSE` | LOO Root Mean Square Error |
| `LOO_MAE` | LOO Mean Absolute Error |
| `LOO_StdDev` | LOO standard deviation |
| `LOO_Min` | LOO minimum residual |
| `LOO_Max` | LOO maximum residual |
| `LOO_NMAD` | LOO NMAD |
| `Bootstrap_Mean` | Bootstrap mean RMSE (if enabled) |
| `Bootstrap_Std` | Bootstrap standard deviation |
| `Bootstrap_CI_Lower` | Bootstrap 95% CI lower bound |
| `Bootstrap_CI_Upper` | Bootstrap 95% CI upper bound |

#### uas_validation.py Point Residuals CSV

When using `--save-residuals` flag, per-point residuals CSV contains:

| Column | Description |
|--------|-------------|
| `Point_ID` | Sequential point identifier (1, 2, 3, ...) |
| `E` | Easting coordinate |
| `N` | Northing coordinate |
| `Z_bare` | Bare-ground elevation |
| `Z_snow` | Snow-on elevation (corrected or uncorrected) |
| `Residual` | Calculated residual (Z_snow - Z_bare) |
| `Label` | Validation label |

---

## Configuration File

YAML configuration file (`config.yaml`) controls all workflow parameters.

### Complete Configuration Template

```yaml
# -------- Thresholds --------
thresholds:
  # Maximum RMSE (meters) for Tier 1 to be acceptable
  tier1_rmse_max: 0.15

  # Minimum improvement (decimal) required for Tier 2/3
  tier2_improvement_min: 0.20

  # Correction mode
  # Options: auto, ppk, tier1, none, vsc, tier2, vertical, ptc, tier3, planar, tilt
  correction_mode: "auto"

# -------- Validation --------
validation:
  # Whether to run bootstrap uncertainty analysis
  run_bootstrap: true

  # Number of bootstrap iterations
  bootstrap_iterations: 250

# -------- CRS --------
crs:
  # Horizontal CRS (EPSG code format)
  horizontal: "EPSG:6342"

  # Vertical datum
  vertical: "NAVD88"

# -------- Paths --------
paths:
  # Area of Interest name (folder organization)
  aoi_name: "Widowmaker"

  # Bare ground DSM filename
  bare_ground_file: "bare_ground_dsm.tif"

  # Virtual GCP shapefile
  vgcp_file: "vgcp_points.shp"

  # Single snow-on DSM (empty for batch mode)
  snow_on_file: ""

  # Batch folder (empty for single file mode)
  snow_on_folder: "."
```

### Configuration Parameters

#### Thresholds Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tier1_rmse_max` | float | Maximum RMSE (m) for Tier 1 to be acceptable | 0.15 |
| `tier2_improvement_min` | float | Minimum relative improvement for Tier 2/3 | 0.20 |
| `correction_mode` | string | Correction tier selection mode | "auto" |

**Correction Mode Options:**
- `auto`: Automatic tier selection (recommended)
- `ppk`, `tier1`, `none`: Force Tier 1 (no correction)
- `vsc`, `tier2`, `vertical`: Force Tier 2 (vertical shift)
- `ptc`, `tier3`, `planar`, `tilt`: Force Tier 3 (planar trend)

#### Validation Section

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `run_bootstrap` | boolean | Enable bootstrap uncertainty analysis | true |
| `bootstrap_iterations` | integer | Number of bootstrap samples | 250 |

#### CRS Section

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `horizontal` | string | Horizontal CRS in EPSG format | "EPSG:6342" |
| `vertical` | string | Vertical datum name | "NAVD88" |

#### Paths Section

| Parameter | Type | Description |
|-----------|------|-------------|
| `aoi_name` | string | Area of Interest name (folder organization) |
| `bare_ground_file` | string | Filename of bare-ground DSM |
| `vgcp_file` | string | Filename of vGCP shapefile |
| `snow_on_file` | string | Single DSM filename (empty for batch mode) |
| `snow_on_folder` | string | Folder indicator (empty for single file mode) |

### Configuration Examples

#### Single File Processing

```yaml
paths:
  aoi_name: "SiteA"
  bare_ground_file: "bare_ground.tif"
  vgcp_file: "vgcp.shp"
  snow_on_file: "snow_20241215.tif"  # Specific file
  snow_on_folder: ""                 # Empty for single file
```

#### Batch Processing

```yaml
paths:
  aoi_name: "SiteA"
  bare_ground_file: "bare_ground.tif"
  vgcp_file: "vgcp.shp"
  snow_on_file: ""                   # Empty for batch
  snow_on_folder: "."                # Any non-empty string
```

#### Fast Processing (No Bootstrap)

```yaml
validation:
  run_bootstrap: false
  bootstrap_iterations: 0
```

#### Manual Tier Selection

```yaml
thresholds:
  tier1_rmse_max: 0.15
  tier2_improvement_min: 0.20
  correction_mode: "vsc"  # Force Tier 2 (vertical shift)
```

---

## Common Workflows

### Workflow 1: Single DSM Correction

```bash
# Step 1: Configure for single file
# Edit config.yaml:
#   snow_on_file: "snow_20241215.tif"
#   snow_on_folder: ""

# Step 2: Run correction (includes validation automatically)
python uas_snow_correction.py --config config.yaml

# Note: Validation (LOO + Bootstrap) is automatically included in uas_snow_correction.py
# No need to run uas_validation.py unless you want to re-validate with different parameters
```

### Workflow 2: Batch Processing Multiple DSMs

```bash
# Step 1: Configure for batch
# Edit config.yaml:
#   snow_on_file: ""
#   snow_on_folder: "."

# Step 2: Run batch correction (includes validation for each DSM)
python uas_snow_correction.py --config config.yaml --batch

# Note: Each DSM is automatically validated during correction
```

### Workflow 3: Using Standalone Validation Tool

The standalone validation tool is useful when you need to:
- Validate DSMs corrected outside this workflow
- Re-validate with different bootstrap iterations
- Validate DSMs from other sources

```bash
# Validate a single external DSM
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow external_corrected_dsm.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output external_validation.csv \
  --bootstrap 500 \
  --plot-dir validation_plots/

# Batch validate multiple external DSMs
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow folder_of_external_dsms/ \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output batch_external_validation.csv \
  --bootstrap 250
```

### Workflow 4: Fast Processing (No Bootstrap)

```bash
# Method 1: Disable in config.yaml
# validation:
#   run_bootstrap: false

python uas_snow_correction.py --config config.yaml --batch

# Method 2: Use command-line flag
python uas_snow_correction.py --config config.yaml --batch --no-bootstrap
```

### Workflow 5: Debugging and Troubleshooting

```bash
# Enable verbose logging for detailed output
python uas_snow_correction.py --config config.yaml --verbose

# Validation with verbose output
python uas_validation.py \
  --bare data/SiteA/bareGround/bare.tif \
  --snow outputs/SiteA/corrected/snow_Tier2_DSM.tif \
  --vgcp data/SiteA/vGCP/vgcp.shp \
  --output debug_validation.csv \
  --verbose
```

### Workflow 6: Multiple Sites Processing

```bash
# Process Site A
python uas_snow_correction.py --config config_siteA.yaml --batch

# Process Site B
python uas_snow_correction.py --config config_siteB.yaml --batch

# Or use AOI override with single config
python uas_snow_correction.py --config config.yaml --aoi SiteA --batch
python uas_snow_correction.py --config config.yaml --aoi SiteB --batch
```

### Workflow 7: Manual Tier Selection

```bash
# Test different correction tiers

# Force Tier 1 (PPK-only)
# Edit config.yaml: correction_mode: "ppk"
python uas_snow_correction.py --config config.yaml

# Force Tier 2 (Vertical Shift)
# Edit config.yaml: correction_mode: "vsc"
python uas_snow_correction.py --config config.yaml

# Force Tier 3 (Planar Trend)
# Edit config.yaml: correction_mode: "ptc"
python uas_snow_correction.py --config config.yaml
```

---

## Quick Reference Tables

### Command-Line Flags

| Script | Flag | Effect |
|--------|------|--------|
| `uas_snow_correction.py` | `--config FILE` | Specify config file (required) |
| | `--batch` | Enable batch processing |
| | `--no-bootstrap` | Skip bootstrap analysis |
| | `--aoi NAME` | Override AOI name |
| | `--verbose` | Detailed logging |
| `uas_validation.py` | `--bare FILE` | Bare-ground DSM (required) |
| | `--snow FILE/DIR` | Snow DSM or folder (required) |
| | `--vgcp FILE` | vGCP shapefile (required) |
| | `--output FILE` | Output CSV (required) |
| | `--bootstrap N` | Bootstrap iterations |
| | `--plot-dir DIR` | Output plot directory |
| | `--label NAME` | Output label |
| | `--save-residuals` | Save per-point residuals |
| | `--verbose` | Detailed logging |

### File Extensions

| Extension | Description |
|-----------|-------------|
| `.tif`, `.tiff` | GeoTIFF raster files (DSMs) |
| `.shp` | Shapefile (vGCP points) |
| `.yaml`, `.yml` | Configuration files |
| `.csv` | Statistics output |
| `.png` | Validation plots |

### Processing Time Estimates

| Task | Points | Bootstrap | Approximate Time |
|------|--------|-----------|------------------|
| Single DSM | 10 vGCPs | Disabled | 10-30 seconds |
| Single DSM | 10 vGCPs | 250 iterations | 30-60 seconds |
| Batch (10 DSMs) | 10 vGCPs | Disabled | 2-5 minutes |
| Batch (10 DSMs) | 10 vGCPs | 250 iterations | 5-10 minutes |

---

## Notes

- All paths in config file are relative to the config file location
- DSM files must be in GeoTIFF format (.tif or .tiff)
- vGCP shapefile must include E, N, and Elevation fields
- All input data must be in the same coordinate reference system
- Bootstrap iterations can be adjusted based on desired precision vs. processing time
- Verbose mode is useful for debugging but produces more console output
