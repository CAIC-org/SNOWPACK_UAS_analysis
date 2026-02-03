"""
UAS Snow Height Correction Workflow - Part 1: DSM Correction

Author: Valerie Foley
Last Updated: 2/3/2026

Description:
    Automated workflow for correcting systematic errors in UAS-derived snow-on DSMs
    using virtual Ground Control Points (vGCPs). Implements three-tier correction
    approach with automatic tier selection, validation, and batch processing.
    
    This script performs only the DSM correction. Snow depth calculation is done
    separately using the companion script: uas_snow_height_calculator.py

Three Tiers of Correction:
    Tier 1: PPK-only (no correction)
    Tier 2: Vertical Shift Correction (remove mean error) -- general bias correction
    Tier 3: Planar Trend Correction (fit and remove planar trend) -- spatially variable correction

Outputs:
    - Corrected DSMs saved to: outputs/{aoi_name}/corrected/
    - Statistics saved to: outputs/{aoi_name}/statistics/
    - Plots saved to: outputs/{aoi_name}/plots/

Next Step:
    After running this script, run uas_snow_height_calculator.py to calculate 
    snow depth from the corrected DSMs.

Usage:
    - Single File Mode: python uas_snow_correction.py --config config.yaml
    - Batch Mode: python uas_snow_correction.py --config config.yaml --batch
    - Fast Mode (no bootstrap): python uas_snow_correction.py --config config.yaml --no-bootstrap
    - Verbose Output: python uas_snow_correction.py --config config.yaml --verbose
    - note:
        - ensure the config file correctly specifies single vs batch by leaving the correct
        field as empty strings or filling them in (fields: snow_on_file or snow_on_folder)
        
Requirements:
    - rasterio, geopandas, numpy, pandas, matplotlib, seaborn
"""

# --------- Load Libraries --------
import os
import sys
import yaml
import argparse
from pathlib import Path
from glob import glob
import logging

import rasterio
from rasterio.transform import rowcol, xy
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def setup_logging(verbose=False):
    # Configure logging for the workflow
    # @param verbose: If True, set logging to DEBUG level, otherwise INFO
    # @returns: None
    # note:
    #   - INFO level: Shows major workflow steps
    #   - DEBUG level: Shows detailed processing information

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path):
    # Load and validate configuration from YAML file
    # @param config_path: Path to the YAML config file
    # @returns: dict - configuration dictionary with all user defined parameters

    # Convert to Path object for easier handling
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        # throw error if no config file found
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required = ['thresholds', 'validation', 'crs', 'paths']
    for key in required:
        if key not in config:
            # throw error if required config section(s) are missing
            raise ValueError(f"Missing required config section: {key}")
    
    logging.info(f"Loaded config: {config_path}")
    return config


def parse_args():
    # Parse command-line arguments
    # @param: None
    # @returns: argparse.Namespace with parsed arguments
    #     Arguments:
    #       - config: Path to config YAML file
    #       - batch: Boolean flag to enable batch processing
    #       - no_bootstrap: Boolean flag to skip bootstrap validation
    #       - aoi: Optional AOI name override
    #       - verbose: Boolean flag to enable verbose logging

    parser = argparse.ArgumentParser(
        description='UAS Snow Depth Correction Workflow (GDAL Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--batch', action='store_true', help='Batch process entire folder')
    parser.add_argument('--no-bootstrap', action='store_true', help='Skip bootstrap validation')
    parser.add_argument('--aoi', default=None, help='Override AOI name from config')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def create_folders(base_path, aoi_name):
    # Create directory structure for data and outputs
    # @param base_path: Base directory path for the project
    # @param aoi_name: Name of the Area of Interest
    # @returns: dict - dictionary containing paths to all created folders

    # Define folder structure
    folders = {
        'data_vgcp': base_path / 'data' / aoi_name / 'vGCP',
        'data_bare': base_path / 'data' / aoi_name / 'bareGround',
        'data_snow': base_path / 'data' / aoi_name / 'snowOn',
        'output_corrected': base_path / 'outputs' / aoi_name / 'corrected',
        'output_statistics': base_path / 'outputs' / aoi_name / 'statistics',
        'output_plots': base_path / 'outputs' / aoi_name / 'plots'
    }
    
    # Create each folder if it doesn't exist
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    return folders


def load_raster(path):
    # Load raster file and extract metadata
    # @param path: Path to raster file
    # @returns: tuple (data, meta) where:
    #   - data: 2D numpy array of raster values
    #   - meta: dict containing transform, crs, nodata, shape, dtype

    with rasterio.open(path) as src:
        # Read first band
        data = src.read(1)
        
        # Extract metadata
        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'nodata': src.nodata,
            'shape': data.shape,
            'dtype': data.dtype
        }
    
    logging.debug(f"Loaded raster: {Path(path).name}")
    return data, meta


def load_vgcp(path):
    # Load virtual Ground Control Points from shapefile
    # @param path: Path to vGCP shapefile
    # @returns: GeoDataFrame with E, N, Elevation fields

    # Load shapefile
    gdf = gpd.read_file(path)
    
    # Check for required fields
    required = ['E', 'N', 'Elevation']
    missing = [f for f in required if f not in gdf.columns]
    
    if missing:
        # throw error if required fields are missing from shapefile
        raise ValueError(f"vGCP shapefile missing required fields: {missing}")
    
    logging.info(f"Loaded {len(gdf)} vGCP points from {Path(path).name}")
    return gdf


def extract_at_points(raster_data, transform, coords):
    # Extract raster values at specified coordinate locations
    # @param raster_data: 2D numpy array of raster values
    # @param transform: Raster affine transformation
    # @param coords: Nx2 numpy array of (x, y) coordinates
    # @returns: numpy array of extracted values (NaN for out-of-bounds points)

    values = np.full(len(coords), np.nan)
    
    # Loop through each coordinate point
    for i, (x, y) in enumerate(coords):
        # Convert XY to row/col
        row, col = rowcol(transform, x, y)
        
        # Check if point is within raster bounds
        if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
            values[i] = raster_data[row, col]
        else:
            # log warning for points outside raster bounds
            logging.warning(f"Point {i+1} outside raster bounds")
    
    return values


def validate_crs(raster_path, shp_path, expected_h, expected_v):
    # Validate that raster and shapefile have matching CRS
    # @param raster_path: Path to raster file
    # @param shp_path: Path to shapefile
    # @param expected_h: Expected horizontal CRS (e.g., 'EPSG:6342')
    # @param expected_v: Expected vertical CRS (not currently validated)
    # @returns: tuple (success, message) where:
    #   - success: Boolean indicating if validation passed
    #   - message: String describing validation result
    # note:
    #   - Handles both simple CRS (EPSG:6342) and compound CRS (EPSG:6342+5703)
    #   - Compound CRS contains both horizontal and vertical components

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_string()
    
    shp_crs = gpd.read_file(shp_path).crs.to_string()
    
    # Extract EPSG code from expected_h (e.g., "EPSG:6342" -> "6342")
    expected_code = expected_h.split(':')[-1]
    
    # Check if expected EPSG code appears in the raster CRS string
    # This handles both "EPSG:6342" and compound CRS with AUTHORITY["EPSG","6342"]
    if expected_code not in raster_crs:
        return False, f"Raster CRS mismatch: expected {expected_h}, got {raster_crs}"
    
    if expected_code not in shp_crs:
        return False, f"Shapefile CRS mismatch: expected {expected_h}, got {shp_crs}"
    
    return True, "CRS validation passed"


def calculate_statistics(residuals):
    # Calculate error statistics from residuals
    # @param residuals: numpy array of residual values
    # @returns: dict with statistics (ME, MAE, RMSE, StdDev, n) or None if no valid data

    residuals_clean = residuals[~np.isnan(residuals)]
    
    if len(residuals_clean) == 0:
        return None
    
    stats = {
        'n': len(residuals_clean),
        'ME': np.mean(residuals_clean),
        'MAE': np.mean(np.abs(residuals_clean)),
        'RMSE': np.sqrt(np.mean(residuals_clean**2)),
        'StdDev': np.std(residuals_clean, ddof=1)
    }
    
    return stats


def tier1_ppk_only(vgcp_df):
    # Tier 1: PPK-only (no correction applied)
    # @param vgcp_df: DataFrame with vGCP data including Residual column
    # @returns: dict with tier1_stats and tier1_residuals

    residuals = vgcp_df['Residual'].values
    stats = calculate_statistics(residuals)
    
    return {
        'tier1_stats': stats,
        'tier1_residuals': residuals
    }


def tier2_vertical_shift(vgcp_df, snow_data, nodata_value):
    # Tier 2: Vertical Shift Correction (remove mean error)
    # @param vgcp_df: DataFrame with vGCP data including Residual column
    # @param snow_data: 2D numpy array of snow-on DSM values
    # @param nodata_value: NoData value to preserve
    # @returns: dict with tier2_shift, tier2_corrected, tier2_residuals, tier2_stats
    # note:
    #   - Preserves NoData values by masking them during correction

    me = vgcp_df['Residual'].mean()
    
    # Create a copy and apply correction only to valid data
    corrected = snow_data.copy()
    
    # Create mask for valid data (not NoData)
    if nodata_value is not None:
        valid_mask = snow_data != nodata_value
        corrected[valid_mask] = snow_data[valid_mask] - me
    else:
        corrected = snow_data - me
    
    residuals = vgcp_df['Residual'].values - me
    stats = calculate_statistics(residuals)
    
    return {
        'tier2_shift': me,
        'tier2_corrected': corrected,
        'tier2_residuals': residuals,
        'tier2_stats': stats
    }


def tier3_planar_trend(vgcp_df, snow_data, meta):
    # Tier 3: Planar Trend Correction (fit and remove planar trend)
    # @param vgcp_df: DataFrame with vGCP data including E, N, Residual columns
    # @param snow_data: 2D numpy array of snow-on DSM values
    # @param meta: dict containing raster metadata (transform, shape, nodata)
    # @returns: dict with tier3_coeffs, tier3_corrected, tier3_residuals, tier3_stats
    #           or None if fitting fails or insufficient points
    # note:
    #   - Preserves NoData values by masking them during correction

    if len(vgcp_df) < 3:
        logging.warning("Tier 3 requires >= 3 vGCP points")
        return None
    
    coords = vgcp_df[['E', 'N']].values
    residuals = vgcp_df['Residual'].values
    
    ones = np.ones((len(coords), 1))
    A = np.hstack([coords, ones])
    
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None)
    except np.linalg.LinAlgError:
        logging.warning("Tier 3 plane fit failed")
        return None
    
    height, width = meta['shape']
    y_coords, x_coords = np.meshgrid(
        np.arange(height),
        np.arange(width),
        indexing='ij'
    )
    
    transform = meta['transform']
    X = transform[2] + x_coords * transform[0]
    Y = transform[5] + y_coords * transform[4]
    
    correction = coeffs[0] * X + coeffs[1] * Y + coeffs[2]
    
    # Create a copy and apply correction only to valid data
    corrected = snow_data.copy()
    nodata_value = meta.get('nodata')
    
    if nodata_value is not None:
        valid_mask = snow_data != nodata_value
        corrected[valid_mask] = snow_data[valid_mask] - correction[valid_mask]
    else:
        corrected = snow_data - correction
    
    fitted = coeffs[0] * coords[:, 0] + coeffs[1] * coords[:, 1] + coeffs[2]
    new_residuals = residuals - fitted
    
    stats = calculate_statistics(new_residuals)
    
    return {
        'tier3_coeffs': coeffs,
        'tier3_corrected': corrected,
        'tier3_residuals': new_residuals,
        'tier3_stats': stats
    }


def select_tier(results, config):
    # Select best correction tier based on criteria
    # @param results: dict containing tier evaluation results
    # @param config: Configuration dictionary with thresholds
    # @returns: results dict updated with 'selected' tier and 'reason'
    # note:
    #   - Manual mode: Uses correction_mode from config if not 'auto'
    #   - Auto mode: Selects tier based on improvement thresholds and RMSE comparison

    tier1_stats = results['tier1_stats']
    tier2_stats = results.get('tier2_stats')
    
    # Get threshold from config (support both old and new parameter names)
    tier2_imp_min = config['thresholds'].get('tier2_improvement_threshold') or \
                    config['thresholds'].get('tier2_improvement_min', 0.20)
    
    # -------------------------------------------------------------------------
    # Manual mode tier selection (correction_mode specified)
    # -------------------------------------------------------------------------
    correction_mode = config['thresholds'].get('correction_mode', 'auto').lower()
    
    if correction_mode != 'auto':
        # Map correction mode strings to tier names
        if correction_mode in ['ppk', 'tier1', 'none']:
            results['selected'] = 'Tier1'
            results['reason'] = f"Manual mode: No correction (mode={correction_mode})"
        elif correction_mode in ['vsc', 'tier2', 'vertical']:
            results['selected'] = 'Tier2'
            results['reason'] = f"Manual mode: Vertical Shift Correction (mode={correction_mode})"
        elif correction_mode in ['ptc', 'tier3', 'planar', 'tilt']:
            if results.get('tier3_stats'):
                results['selected'] = 'Tier3'
                results['reason'] = f"Manual mode: Planar Trend Correction (mode={correction_mode})"
            else:
                results['selected'] = 'Tier2'
                results['reason'] = f"Manual mode requested Tier3 but it failed, using Tier2"
        else:
            # Unknown mode, default to auto
            logging.warning(f"Unknown correction_mode '{correction_mode}', using auto selection")
            correction_mode = 'auto'
    
    if correction_mode == 'auto':
        # -------------------------------------------------------------------------
        # Auto mode tier selection
        # -------------------------------------------------------------------------
        tier2_imp = (tier1_stats['RMSE'] - tier2_stats['RMSE']) / tier1_stats['RMSE']
        
        if tier2_imp < tier2_imp_min:
            # Tier 2 doesn't meet improvement threshold
            tier3_stats = results.get('tier3_stats')
            if tier3_stats and tier3_stats['RMSE'] < tier2_stats['RMSE']:
                results['selected'] = 'Tier3'
                results['reason'] = f"Tier 2 improvement ({tier2_imp*100:.1f}%) < threshold, Tier 3 better"
            else:
                results['selected'] = 'Tier1'
                results['reason'] = "No tier met improvement criteria"
        else:
            # Tier 2 meets improvement threshold
            tier3_stats = results.get('tier3_stats')
            if tier3_stats:
                # Compare Tier 2 and Tier 3
                tier3_vs_tier2 = (tier2_stats['RMSE'] - tier3_stats['RMSE']) / tier2_stats['RMSE']
                rmse_diff = tier2_stats['RMSE'] - tier3_stats['RMSE']
                
                # Tier 3 meaningfully better if >5% improvement OR >0.05m RMSE reduction
                if tier3_stats['RMSE'] < tier2_stats['RMSE'] and (tier3_vs_tier2 > 0.05 or rmse_diff > 0.05):
                    results['selected'] = 'Tier3'
                    results['reason'] = f"Tier 3 meaningfully better than Tier 2 ({tier3_vs_tier2*100:.1f}% improvement)"
                else:
                    results['selected'] = 'Tier2'
                    results['reason'] = f"Tier 2 meets threshold ({tier2_imp*100:.1f}%), Tier 3 not meaningfully better"
            else:
                results['selected'] = 'Tier2'
                results['reason'] = f"Tier 2 meets threshold ({tier2_imp*100:.1f}%)"
    
    logging.info(f"Selected: {results['selected']} - {results['reason']}")
    
    return results


def generate_filename(input_name, tier, suffix='DSM'):
    # Generate output filename with tier code
    # @param input_name: Input filename
    # @param tier: Selected tier ('Tier1', 'Tier2', or 'Tier3')
    # @param suffix: File suffix (default 'DSM')
    # @returns: str - generated output filename

    # Map tiers to codes
    tier_codes = {'Tier1': 'noCorr', 'Tier2': 'VSC', 'Tier3': 'PTC'}
    
    # Parse input filename
    stem = Path(input_name).stem
    ext = Path(input_name).suffix
    
    # Remove trailing -DSM if present
    if stem.endswith('-DSM'):
        stem = stem[:-4]
    
    # Construct output filename
    output_name = f"{stem}_{tier_codes[tier]}_{suffix}{ext}"
    
    return output_name


def save_raster(data, meta, path):
    # Save raster data to GeoTIFF file
    # @param data: 2D numpy array of raster values
    # @param meta: dict containing raster metadata (crs, transform, nodata, etc.)
    # @param path: Output file path
    # @returns: None
    # note:
    #   - Forces float32 dtype to match original raster format
    #   - Preserves NoData values properly

    # Ensure output directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Force float32 to match original format and reduce file size
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    # Write raster
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=np.float32,  # Force float32
        crs=meta['crs'],
        transform=meta['transform'],
        nodata=meta['nodata'],
        compress='LZW'
    ) as dst:
        dst.write(data, 1)
    
    logging.info(f"Saved: {Path(path).name}")


def loo_validation(vgcp_df, snow_data, snow_meta, tier, tier2_me=None, tier3_coeffs=None):
    # Leave-One-Out Cross Validation
    # @param vgcp_df: DataFrame with vGCP data
    # @param snow_data: 2D numpy array of snow-on DSM values (unused in current implementation)
    # @param snow_meta: Raster metadata dict (unused in current implementation)
    # @param tier: Selected tier ('Tier1', 'Tier2', or 'Tier3')
    # @param tier2_me: Mean error for Tier 2 (optional)
    # @param tier3_coeffs: Planar coefficients for Tier 3 (optional)
    # @returns: tuple (stats, loo_residuals) where:
    #   - stats: dict with LOO validation statistics or None
    #   - loo_residuals: numpy array of LOO residuals
    # note:
    #   - Tier2 requires >= 2 points, Tier3 requires >= 4 points
    #   - Each point is left out, correction fitted on remaining points, tested on left-out point

    n = len(vgcp_df)
    
    if (tier == 'Tier3' and n < 4) or (tier == 'Tier2' and n < 2):
        logging.warning(f"Insufficient points for LOO validation of {tier}")
        return None, None
    
    loo_residuals = np.zeros(n)
    
    # Loop through each point for LOO validation
    for i in range(n):
        # Create mask excluding current point
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        train_df = vgcp_df[mask].copy()
        test_point = vgcp_df.iloc[i]
        
        if tier == 'Tier2':
            # Tier 2: Predict using mean error from training set
            train_me = train_df['Residual'].mean()
            predicted_error = train_me
        elif tier == 'Tier3':
            # Tier 3: Predict using planar fit from training set
            if len(train_df) < 3:
                loo_residuals[i] = np.nan
                continue
            
            coords = train_df[['E', 'N']].values
            residuals = train_df['Residual'].values
            
            ones = np.ones((len(coords), 1))
            A = np.hstack([coords, ones])
            
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, residuals, rcond=None)
                predicted_error = coeffs[0] * test_point['E'] + coeffs[1] * test_point['N'] + coeffs[2]
            except:
                loo_residuals[i] = np.nan
                continue
        else:
            # Tier 1: No correction
            predicted_error = 0
        
        loo_residuals[i] = test_point['Residual'] - predicted_error
    
    stats = calculate_statistics(loo_residuals)
    
    return stats, loo_residuals


def bootstrap_uncertainty(vgcp_df, snow_data, snow_meta, tier, n_iterations, tier2_me=None, tier3_coeffs=None):
    # Bootstrap uncertainty estimation
    # @param vgcp_df: DataFrame with vGCP data
    # @param snow_data: 2D numpy array of snow-on DSM values (unused in current implementation)
    # @param snow_meta: Raster metadata dict (unused in current implementation)
    # @param tier: Selected tier ('Tier1', 'Tier2', or 'Tier3')
    # @param n_iterations: Number of bootstrap iterations
    # @param tier2_me: Mean error for Tier 2 (optional, unused)
    # @param tier3_coeffs: Planar coefficients for Tier 3 (optional, unused)
    # @returns: dict with bootstrap statistics (mean, std, ci_lower, ci_upper) or None
    # note:
    #   - Requires >= 3 points for Tier3, >= 2 points for others
    #   - Uses random sampling with replacement
    #   - Returns 95% confidence interval

    n = len(vgcp_df)
    
    if (tier == 'Tier3' and n < 3) or n < 2:
        logging.warning("Insufficient points for bootstrap")
        return None
    
    rmse_values = []
    
    # Run bootstrap iterations
    for _ in range(n_iterations):
        # Sample with replacement
        boot_idx = np.random.choice(n, n, replace=True)
        boot_df = vgcp_df.iloc[boot_idx].copy()
        
        if tier == 'Tier2':
            # Tier 2: Vertical shift correction
            me = boot_df['Residual'].mean()
            residuals = boot_df['Residual'].values - me
        elif tier == 'Tier3':
            # Tier 3: Planar trend correction
            if len(boot_df) < 3:
                continue
            
            coords = boot_df[['E', 'N']].values
            residuals_orig = boot_df['Residual'].values
            
            ones = np.ones((len(coords), 1))
            A = np.hstack([coords, ones])
            
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, residuals_orig, rcond=None)
                fitted = coeffs[0] * coords[:, 0] + coeffs[1] * coords[:, 1] + coeffs[2]
                residuals = residuals_orig - fitted
            except:
                continue
        else:
            # Tier 1: No correction
            residuals = boot_df['Residual'].values
        
        rmse = np.sqrt(np.mean(residuals**2))
        rmse_values.append(rmse)
    
    if len(rmse_values) < 10:
        logging.warning("Bootstrap failed to generate sufficient samples")
        return None
    
    rmse_values = np.array(rmse_values)
    
    return {
        'mean': np.mean(rmse_values),
        'std': np.std(rmse_values),
        'ci_lower': np.percentile(rmse_values, 2.5),
        'ci_upper': np.percentile(rmse_values, 97.5)
    }


def create_stats_csv(config, results, validation, snow_filename, output_path):
    # Create CSV file with statistics summary
    # @param config: Configuration dictionary
    # @param results: dict with tier evaluation results
    # @param validation: dict with LOO and bootstrap results
    # @param snow_filename: Input snow-on DSM filename
    # @param output_path: Output CSV file path
    # @returns: None

    # Build output data dictionary
    data = {
        'Filename': Path(snow_filename).name,
        'Selected_Tier': results['selected'],
        'Selection_Reason': results['reason'],
        'N_vGCP': results['tier1_stats']['n']
    }
    
    # Add statistics for each evaluated tier
    for tier in ['tier1', 'tier2', 'tier3']:
        if f'{tier}_stats' in results:
            stats = results[f'{tier}_stats']
            prefix = tier.upper()
            data[f'{prefix}_RMSE'] = stats['RMSE']
            data[f'{prefix}_ME'] = stats['ME']
            data[f'{prefix}_MAE'] = stats['MAE']
            data[f'{prefix}_StdDev'] = stats['StdDev']
    
    # Add tier-specific parameters
    if 'tier2_shift' in results:
        data['Tier2_Vertical_Shift'] = results['tier2_shift']
    
    if 'tier3_coeffs' in results:
        coeffs = results['tier3_coeffs']
        data['Tier3_Coeff_E'] = coeffs[0]
        data['Tier3_Coeff_N'] = coeffs[1]
        data['Tier3_Coeff_Intercept'] = coeffs[2]
    
    # Add LOO validation statistics
    if validation.get('loo'):
        data['LOO_RMSE'] = validation['loo']['RMSE']
        data['LOO_ME'] = validation['loo']['ME']
    
    # Add bootstrap statistics
    if validation.get('bootstrap'):
        data['Bootstrap_Mean'] = validation['bootstrap']['mean']
        data['Bootstrap_CI_Lower'] = validation['bootstrap']['ci_lower']
        data['Bootstrap_CI_Upper'] = validation['bootstrap']['ci_upper']
    
    # Convert to DataFrame and save
    df = pd.DataFrame([data])
    df.to_csv(output_path, index=False, float_format='%.6f')
    
    logging.info(f"Saved stats: {Path(output_path).name}")


def save_point_residuals(vgcp_df, results, snow_file, output_path):
    # Save per-point residuals to CSV for all evaluated tiers
    # @param vgcp_df: DataFrame with vGCP data including E, N, Z_bare, Z_snow, Residual
    # @param results: Tier evaluation results dict containing residuals for each tier
    # @param snow_file: Input snow-on DSM filename
    # @param output_path: Output CSV file path
    # @returns: None
    
    # Build output dataframe with base information
    residuals_df = pd.DataFrame({
        'Point_ID': range(1, len(vgcp_df) + 1),
        'E': vgcp_df['E'].values,
        'N': vgcp_df['N'].values,
        'Z_bare': vgcp_df['Z_bare'].values,
        'Z_snow_ppk': vgcp_df['Z_snow'].values,
        'Tier1_Residual': vgcp_df['Residual'].values
    })
    
    # Add Tier 2 residuals if Tier 2 was evaluated
    if 'tier2_residuals' in results:
        residuals_df['Tier2_Residual'] = results['tier2_residuals']
    
    # Add Tier 3 residuals if Tier 3 was evaluated
    if 'tier3_residuals' in results:
        residuals_df['Tier3_Residual'] = results['tier3_residuals']
    
    # Add a column indicating which tier was selected
    residuals_df['Selected_Tier'] = results['selected']
    
    # Save to CSV
    residuals_df.to_csv(output_path, index=False, float_format='%.6f')
    logging.info(f"Saved point residuals: {Path(output_path).name}")


def create_plots(vgcp_df, results, validation, output_folder, snow_file):
    # Create diagnostic plots for tier evaluation
    # @param vgcp_df: DataFrame with vGCP data
    # @param results: dict with tier evaluation results
    # @param validation: dict with LOO and bootstrap results
    # @param output_folder: Path to output folder for plots
    # @param snow_file: Input snow-on DSM filename
    # @returns: None
    # note:
    #   - Creates point-by-point residual comparison (before vs after correction)
    #   - Creates bar chart comparing RMSE across tiers
    
    stem = Path(snow_file).stem
    selected = results['selected']
    
    # -------------------------------------------------------------------------
    # Plot 1: Point-by-point residual comparison (No Correction vs Final Correction)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get residuals
    tier1_residuals = results['tier1_residuals']  # No correction
    final_residuals = results[f"{selected.lower()}_residuals"]  # Selected tier
    
    # Create point IDs
    point_ids = np.arange(1, len(tier1_residuals) + 1)
    
    # Plot both sets of residuals (removed alpha for solid colors)
    ax.plot(point_ids, tier1_residuals, 'o-', label='No Correction (Tier 1)', 
            color='grey', markersize=8, linewidth=2)
    ax.plot(point_ids, final_residuals, 'o-', label=f'After Correction ({selected})', 
            color='green', markersize=8, linewidth=2)
    
    # Add zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('vGCP Point ID', fontsize=12)
    ax.set_ylabel('Residual (m)', fontsize=12)
    ax.set_title(f'Residual Comparison: Before vs After Correction\n{stem}', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all point IDs
    ax.set_xticks(point_ids)
    
    plt.tight_layout()
    plt.savefig(output_folder / f"{stem}_residual_comparison.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # Plot 2: Tier comparison bar chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    tiers = []
    tier_labels = []  # For proper capitalization
    rmse_vals = []
    
    for tier in ['tier1', 'tier2', 'tier3']:
        if f'{tier}_stats' in results:
            tiers.append(tier)
            # Proper capitalization: "Tier 1", "Tier 2", "Tier 3"
            tier_labels.append(f"Tier {tier[-1]}")
            rmse_vals.append(results[f'{tier}_stats']['RMSE'])
    
    # Create bars
    bars = ax.bar(tier_labels, rmse_vals, color='lightblue')
    
    # Highlight selected tier
    selected_idx = tiers.index(selected.lower())
    bars[selected_idx].set_color('green')
    bars[selected_idx].set_label('Selected Tier')
    
    # Labels and formatting
    ax.set_ylabel('RMSE (m)', fontsize=12)
    ax.set_title(f'Tier Comparison - RMSE\n{stem}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (tier_label, rmse) in enumerate(zip(tier_labels, rmse_vals)):
        ax.text(i, rmse + 0.01, f'{rmse:.3f}m', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_folder / f"{stem}_tier_comparison.png", dpi=300)
    plt.close()


def process_single_dsm(config, folders, snow_file):
    # Process a single snow-on DSM file
    # @param config: Configuration dictionary
    # @param folders: dict of folder paths
    # @param snow_file: Filename of snow-on DSM to process
    # @returns: dict with processing results for this DSM

    logging.info(f"\nProcessing: {snow_file}")
    
    # Construct file paths
    vgcp_path = folders['data_vgcp'] / config['paths']['vgcp_file']
    bare_path = folders['data_bare'] / config['paths']['bare_ground_file']
    snow_path = folders['data_snow'] / snow_file
    
    # Validate paths exist
    for path, name in [(vgcp_path, 'vGCP'), (bare_path, 'Bare ground'), (snow_path, 'Snow-on')]:
        if not path.exists():
            # throw error if required file not found
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    # Validate CRS
    expected_h = config['crs']['horizontal']
    expected_v = config['crs']['vertical']
    
    success, msg = validate_crs(bare_path, vgcp_path, expected_h, expected_v)
    if not success:
        raise ValueError(msg)
    
    success, msg = validate_crs(snow_path, vgcp_path, expected_h, expected_v)
    if not success:
        raise ValueError(msg)
    
    # Load data
    logging.info("Loading data...")
    vgcp_gdf = load_vgcp(vgcp_path)
    bare_data, bare_meta = load_raster(bare_path)
    snow_data, snow_meta = load_raster(snow_path)
    
    # Extract values at vGCP locations
    coords = vgcp_gdf[['E', 'N']].values
    z_bare = extract_at_points(bare_data, bare_meta['transform'], coords)
    z_snow = extract_at_points(snow_data, snow_meta['transform'], coords)
    
    # Create DataFrame with extracted values
    vgcp_df = pd.DataFrame({
        'E': vgcp_gdf['E'].values,
        'N': vgcp_gdf['N'].values,
        'Z_reference': vgcp_gdf['Elevation'].values,
        'Z_bare': z_bare,
        'Z_snow': z_snow,
        'Residual': z_snow - vgcp_gdf['Elevation'].values
    })
    
    # Filter valid points (remove NaN values)
    valid_mask = ~(np.isnan(vgcp_df['Z_bare']) | np.isnan(vgcp_df['Z_snow']))
    vgcp_df = vgcp_df[valid_mask].reset_index(drop=True)
    
    n_valid = len(vgcp_df)
    n_total = len(vgcp_gdf)
    logging.info(f"Valid vGCP points: {n_valid}/{n_total}")
    
    if n_valid < 2:
        # throw error if insufficient valid points
        raise ValueError("Insufficient valid vGCP points (need >= 2)")
    
    # Run tier corrections
    logging.info("Running tier evaluations...")
    results = {}
    
    # Tier 1: PPK-only (no correction)
    results.update(tier1_ppk_only(vgcp_df))
    logging.info(f"Tier 1 RMSE: {results['tier1_stats']['RMSE']:.4f} m")
    
    # Tier 2: Vertical shift correction
    results.update(tier2_vertical_shift(vgcp_df, snow_data, snow_meta['nodata']))
    logging.info(f"Tier 2 RMSE: {results['tier2_stats']['RMSE']:.4f} m")
    
    # Tier 3: Planar trend correction
    tier3_result = tier3_planar_trend(vgcp_df, snow_data, snow_meta)
    if tier3_result:
        results.update(tier3_result)
        logging.info(f"Tier 3 RMSE: {results['tier3_stats']['RMSE']:.4f} m")
    
    # Select best tier based on criteria
    results = select_tier(results, config)
    
    # Get corrected DSM based on selected tier
    selected = results['selected']
    if selected == 'Tier1':
        corrected_dsm = snow_data.copy()
    elif selected == 'Tier2':
        corrected_dsm = results['tier2_corrected']
    else:  # Tier3
        corrected_dsm = results['tier3_corrected']
    
    # Run validation
    logging.info("Running validation...")
    validation = {}
    
    # Prepare parameters for validation
    tier2_me = results['tier1_stats']['ME'] if selected == 'Tier2' else None
    tier3_coeffs = results.get('tier3_coeffs') if selected == 'Tier3' else None
    
    # LOO validation
    loo_stats, loo_residuals = loo_validation(vgcp_df, snow_data, snow_meta, selected, tier2_me, tier3_coeffs)
    
    if loo_stats is not None:
        validation['loo'] = loo_stats
        validation['loo_residuals'] = loo_residuals
        logging.info(f"LOO RMSE: {loo_stats['RMSE']:.4f} m")
    else:
        validation['loo'] = None
        validation['loo_residuals'] = None
        logging.info("LOO validation skipped (insufficient points for selected tier)")
    
    # Bootstrap validation
    if config['validation']['run_bootstrap']:
        boot_stats = bootstrap_uncertainty(
            vgcp_df, snow_data, snow_meta, selected,
            config['validation']['bootstrap_iterations'],
            tier2_me, tier3_coeffs
        )
        if boot_stats is not None:
            validation['bootstrap'] = boot_stats
            logging.info(f"Bootstrap: mean={boot_stats['mean']:.4f} m, CI=[{boot_stats['ci_lower']:.4f}, {boot_stats['ci_upper']:.4f}]")
        else:
            validation['bootstrap'] = None
            logging.info("Bootstrap validation skipped (insufficient points for selected tier)")
    
    # Save outputs
    logging.info("Saving outputs...")
    
    # Save corrected DSM
    output_name = generate_filename(snow_file, selected, 'DSM')
    corrected_path = folders['output_corrected'] / output_name
    save_raster(corrected_dsm, snow_meta, str(corrected_path))
    
    logging.info("DSM correction complete. Run uas_snow_height_calculator.py for snow depth calculation.")
    
    # Save statistics CSV
    stats_name = Path(snow_file).stem + '_statistics.csv'
    create_stats_csv(config, results, validation, snow_file,
                    str(folders['output_statistics'] / stats_name))
    
    # Save per-point residuals CSV
    residuals_name = Path(snow_file).stem + '_point_residuals.csv'
    save_point_residuals(vgcp_df, results, snow_file,
                        str(folders['output_statistics'] / residuals_name))
    
    # Create plots
    create_plots(vgcp_df, results, validation, folders['output_plots'], snow_file)
    
    logging.info(f"Completed: {snow_file}")
    
    return results


def get_snow_dsms(snow_path):
    # Get list of snow-on DSM files to process
    # @param snow_path: Path to single file or directory
    # @returns: tuple (files, parent_dir) where:
    #   - files: list of filenames
    #   - parent_dir: parent directory path

    snow_path = Path(snow_path)
    
    if snow_path.is_file():
        # Single file mode
        return [snow_path.name], snow_path.parent
    
    elif snow_path.is_dir():
        # Batch mode - find all .tif and .tiff files
        files = []
        for ext in ['*.tif', '*.tiff']:
            files.extend([f.name for f in snow_path.glob(ext)])
        
        if not files:
            # throw error if no tif/tiff files found
            raise ValueError(f"No .tif/.tiff files found in {snow_path}")
        
        # Sort for consistent ordering
        files.sort()
        logging.info(f"Found {len(files)} DSM file(s) to process")
        
        return files, snow_path
    
    else:
        # throw error if path is invalid
        raise ValueError(f"Invalid path: {snow_path}")


def process_batch(config, folders):
    # Process multiple snow-on DSM files in batch mode
    # @param config: Configuration dictionary
    # @param folders: dict of folder paths
    # @returns: list of dicts with results for each file

    # Determine snow-on DSM files to process
    snow_input = config['paths'].get('snow_on_file')
    snow_folder = config['paths'].get('snow_on_folder')
    
    if snow_input:
        # Single file specified (even in batch mode)
        snow_files, _ = get_snow_dsms(folders['data_snow'] / snow_input)
    elif snow_folder:
        # Folder specified - process all files
        snow_files, _ = get_snow_dsms(folders['data_snow'])
    else:
        # throw error if neither file nor folder specified
        raise ValueError("Must specify either snow_on_file or snow_on_folder in config")
    
    # Process each DSM
    batch_results = []
    for i, snow_file in enumerate(snow_files, 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"DSM {i}/{len(snow_files)}: {snow_file}")
        logging.info(f"{'='*70}")
        
        try:
            # Process this DSM
            result = process_single_dsm(config, folders, snow_file)
            
            # Store summary info
            batch_results.append({
                'file': snow_file,
                'success': True,
                'selected': result['selected'],
                'rmse': result[f"{result['selected'].lower()}_stats"]['RMSE']
            })
        
        except Exception as e:
            # Log error but continue processing
            logging.error(f"Failed to process {snow_file}: {e}")
            batch_results.append({
                'file': snow_file,
                'success': False,
                'error': str(e)
            })
    
    # Create batch summary CSV
    summary_df = pd.DataFrame(batch_results)
    summary_path = folders['output_statistics'] / 'batch_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"\nBatch summary saved: {summary_path}")
    
    # Print summary
    success_count = sum(r['success'] for r in batch_results)
    logging.info(f"\nBatch complete: {success_count}/{len(batch_results)} successful")
    
    return batch_results


def main():
    # Main workflow function
    # @param: None (args come from command line and config file)
    # @returns: None (exits with status code)

    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.aoi:
            config['paths']['aoi_name'] = args.aoi
        if args.no_bootstrap:
            config['validation']['run_bootstrap'] = False
        
        # Setup folder structure
        base_path = Path(args.config).parent
        folders = create_folders(base_path, config['paths']['aoi_name'])
        
        # Determine processing mode
        if args.batch or config['paths'].get('snow_on_folder'):
            # Batch mode - process multiple DSMs
            logging.info("Running in BATCH mode")
            process_batch(config, folders)
        else:
            # Single file mode
            logging.info("Running in SINGLE file mode")
            snow_file = config['paths']['snow_on_file']
            process_single_dsm(config, folders, snow_file)
        
        # woohoo success
        logging.info("\n" + "="*70)
        logging.info(" Workflow Complete")
        logging.info("="*70)
    
    except Exception as e:
        # Error handling
        logging.error(f"\nWorkflow Failed: {e}")
        
        # Print full traceback if verbose
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
