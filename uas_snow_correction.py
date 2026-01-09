"""
UAS Snow Height Correction Workflow

Author: Valerie Foley
Last Updated: 1/9/2026

Description:
    Automated workflow for correcting systematic errors in UAS-derived snow-on DSMs
    using virtual Ground Control Points (vGCPs). Implements three-tier correction
    approach with automatic tier selection, validation, and batch processing.

Three Tiers of Correction:
    Tier 1: PPK-only (no correction)
    Tier 2: Vertical Shift Correction (remove mean error) -- general bias correction
    Tier 3: Planar Trend Correction (fit and remove planar trend) -- spatially variable correction

Usage:
    - Single File Mode: python uas_snow_correction.py --config config.yaml
    - Batch Mode: python uas_snow_correction.py --config config.yaml --batch
    - Fast Mode (no bootstrap): python uas_snow_correction.py --config config.yaml --no-bootstrap
    - Verbose Output: python uas_snow_correction.py --config config.yaml --verbose
    - note:
        - ensure the config file correctly specifies single vs batch by leaving the correct
        field as empty strings or filling them in (fields: snow_on_file or snow_on_folder)
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
        description='UAS Snow Depth Correction Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode
  python uas_snow_correction.py --config config.yaml

  # Batch mode
  python uas_snow_correction.py --config config.yaml --batch

  # Fast mode (no bootstrap)
  python uas_snow_correction.py --config config.yaml --no-bootstrap

  # Verbose output
  python uas_snow_correction.py --config config.yaml --verbose
        """
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
    #   - success: bool, True if CRS validation passes
    #   - message: str, error description if validation fails

    # Extract expected EPSG code from string
    expected_epsg = int(expected_h.split(':')[1])

    # Check raster CRS
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            # return error if raster has undefined CRS
            return False, f"Raster has undefined CRS: {raster_path}"

        # Try to get EPSG code directly
        raster_epsg = src.crs.to_epsg()

        # If compound CRS (returns None), search in CRS string
        if raster_epsg is None:
            crs_str = str(src.crs)
            logging.info(f"Compound CRS detected for {Path(raster_path).name}")

            # Check if expected EPSG appears in the CRS definition
            epsg_pattern = f'AUTHORITY["EPSG","{expected_epsg}"]'

            if epsg_pattern in crs_str:
                logging.info(f"Found {epsg_pattern} in compound CRS definition")
                raster_epsg = expected_epsg  # Accept it
            else:
                # return error if expected EPSG not found in compound CRS
                return False, f"Expected EPSG:{expected_epsg} not found in compound CRS"

        # Verify EPSG matches
        if raster_epsg != expected_epsg:
            # return error if raster CRS doesn't match expected
            return False, f"Raster CRS mismatch: expected EPSG:{expected_epsg}, got EPSG:{raster_epsg}"

    # Check shapefile CRS
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        # return error if shapefile has undefined CRS
        return False, f"Shapefile has undefined CRS: {shp_path}"

    shp_epsg = gdf.crs.to_epsg()
    if shp_epsg != expected_epsg:
        # return error if shapefile CRS doesn't match expected
        return False, f"Shapefile CRS mismatch: expected EPSG:{expected_epsg}, got EPSG:{shp_epsg}"

    # All checks passed
    logging.info(f"CRS validation passed: EPSG:{expected_epsg}")
    return True, ""


def calc_stats(residuals):
    # Calculate accuracy statistics from residuals
    # @param residuals: numpy array of residual values (measured - truth)
    # @returns: dict containing statistical metrics:
    #   - n: Number of valid points
    #   - ME: Mean Error (bias)
    #   - RMSE: Root Mean Square Error
    #   - MAE: Mean Absolute Error
    #   - StdDev: Standard Deviation
    #   - Min: Minimum residual
    #   - Max: Maximum residual
    #   - NMAD: Normalized Median Absolute Deviation

    # Remove NaN values
    valid = residuals[~np.isnan(residuals)]

    if len(valid) == 0:
        # throw error if no valid residuals found
        raise ValueError("No valid residuals found")

    # Calculate basic statistics
    stats = {
        'n': len(valid),
        'ME': float(np.mean(valid)),
        'RMSE': float(np.sqrt(np.mean(valid**2))),
        'MAE': float(np.mean(np.abs(valid))),
        'StdDev': float(np.std(valid, ddof=1)),
        'Min': float(np.min(valid)),
        'Max': float(np.max(valid))
    }

    # Calculate NMAD (robust statistic)
    median_val = np.median(valid)
    mad = np.median(np.abs(valid - median_val))
    stats['NMAD'] = float(1.4826 * mad)

    return stats


def apply_vertical_shift(data, me, nodata=None):
    # Apply vertical shift correction to remove bias
    # @param data: 2D numpy array of elevation data
    # @param me: Mean error to subtract (bias correction)
    # @param nodata: Nodata value to preserve (optional)
    # @returns: numpy array - corrected elevation data

    # Create copy to avoid modifying original
    corrected = data.copy()

    if nodata is not None:
        # Only correct valid data (not nodata pixels)
        mask = corrected != nodata
        corrected[mask] = corrected[mask] - me
    else:
        # Correct all pixels
        corrected = corrected - me

    return corrected


def fit_plane(vgcp_df):
    # Fit a plane to vGCP residuals using least squares
    # @param vgcp_df: DataFrame with columns E, N, Residual
    # @returns: tuple (a, b, c) - plane coefficients where Residual = a*E + b*N + c

    # Remove any NaN values
    valid = vgcp_df.dropna(subset=['E', 'N', 'Residual'])

    if len(valid) < 3:
        # throw error if not enough points for plane fit
        raise ValueError(f"Need at least 3 valid vGCPs for plane fit, only have {len(valid)}")

    # Build design matrix: Residual = a*E + b*N + c
    X = np.column_stack([
        valid['E'].values,      # Easting coordinates
        valid['N'].values,      # Northing coordinates
        np.ones(len(valid))     # Intercept term
    ])

    y = valid['Residual'].values

    # Solve least squares: minimize ||y - X*coeffs||^2
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)

    a, b, c = coeffs

    logging.info(f"Plane fit: a={a:.8f} m/m, b={b:.8f} m/m, c={c:.4f} m")

    return a, b, c


def create_correction_surface(shape, transform, a, b, c):
    # Create a correction surface by evaluating plane equation at all pixels
    # @param shape: Raster dimensions (rows, cols)
    # @param transform: Raster affine transformation
    # @param a: Plane coefficient for Easting
    # @param b: Plane coefficient for Northing
    # @param c: Plane intercept
    # @returns: 2D numpy array of correction values

    rows, cols = shape

    # Create meshgrid of all row/col indices (vectorized!)
    row_indices, col_indices = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

    # Convert ALL indices to X/Y coordinates using affine transform
    x = transform.c + col_indices * transform.a + row_indices * transform.b
    y = transform.f + col_indices * transform.d + row_indices * transform.e

    # Calculate correction for ALL pixels at once
    correction = a * x + b * y + c

    logging.debug(f"Created correction surface: {rows}x{cols} pixels (vectorized)")

    return correction


def apply_plane_correction(data, correction_surface, nodata=None):
    # Apply planar trend correction to elevation data
    # @param data: 2D numpy array of elevation data
    # @param correction_surface: 2D numpy array of correction values
    # @param nodata: Nodata value to preserve (optional)
    # @returns: numpy array - corrected elevation data

    # Create copy to avoid modifying original
    corrected = data.copy()

    if nodata is not None:
        # Only correct valid data pixels
        mask = corrected != nodata
        corrected[mask] = corrected[mask] - correction_surface[mask]
    else:
        # Correct all pixels
        corrected = corrected - correction_surface

    return corrected


def loo_validation(vgcp_df, snow_data, snow_meta, tier, tier2_me=None, tier3_coeffs=None):
    # Perform Leave-One-Out cross-validation
    # @param vgcp_df: DataFrame with E, N, Z_bare, Residual columns
    # @param snow_data: Original snow-on DSM numpy array
    # @param snow_meta: dict containing raster metadata
    # @param tier: Selected correction tier ('Tier1', 'Tier2', or 'Tier3')
    # @param tier2_me: Mean error for Tier2 correction (optional)
    # @param tier3_coeffs: Plane coefficients (a, b, c) for Tier3 (optional)
    # @returns: tuple (loo_stats, loo_res) where:
    #   - loo_stats: dict of statistics
    #   - loo_res: numpy array of LOO residuals
    # note:
    #   - For Tier3 with only 3 points, LOO cannot be performed (need 3 points to fit plane)
    #   - In this case, returns None for both stats and residuals

    # Remove any NaN values
    valid = vgcp_df.dropna(subset=['E', 'N', 'Residual']).copy()
    n = len(valid)
    
    # Check if LOO is possible for Tier 3
    if tier == 'Tier3' and n < 4:
        logging.warning(f"Cannot perform LOO validation for Tier 3 with only {n} points (need 4+)")
        logging.warning("Skipping LOO validation - correction results are still valid")
        return None, None

    # Array to store LOO residuals
    loo_res = np.zeros(n)

    # Loop through each vGCP
    for i in range(n):
        # Create training set (all points except i)
        train = valid[valid.index != valid.index[i]]

        # Get test point
        test_point = valid.iloc[i]
        coords = np.array([[test_point['E'], test_point['N']]])

        # Apply correction based on tier
        if tier == 'Tier1':
            # No correction - just use original residual
            loo_res[i] = test_point['Residual']

        elif tier == 'Tier2':
            # Recalculate ME using training set
            train_me = train['Residual'].mean()

            # Apply vertical shift correction
            corrected = apply_vertical_shift(snow_data, train_me, snow_meta['nodata'])

            # Extract corrected value at test point
            z_corr = extract_at_points(corrected, snow_meta['transform'], coords)[0]

            # Calculate LOO residual
            loo_res[i] = z_corr - test_point['Z_bare']

        elif tier == 'Tier3':
            # Refit plane using training set
            a, b, c = fit_plane(train)

            # Create correction surface
            surf = create_correction_surface(snow_meta['shape'], snow_meta['transform'], a, b, c)

            # Apply plane correction
            corrected = apply_plane_correction(snow_data, surf, snow_meta['nodata'])

            # Extract corrected value at test point
            z_corr = extract_at_points(corrected, snow_meta['transform'], coords)[0]

            # Calculate LOO residual
            loo_res[i] = z_corr - test_point['Z_bare']

    loo_stats = calc_stats(loo_res)
    logging.info(f"LOO RMSE: {loo_stats['RMSE']:.4f} m")

    # Return both stats and residuals (residuals needed for plotting)
    return loo_stats, loo_res


def bootstrap_uncertainty(vgcp_df, snow_data, snow_meta, tier, n_iter=250,
                         tier2_me=None, tier3_coeffs=None):
    # Estimate uncertainty using bootstrap resampling
    # @param vgcp_df: DataFrame with E, N, Z_bare, Residual columns
    # @param snow_data: Original snow-on DSM numpy array
    # @param snow_meta: dict containing raster metadata
    # @param tier: Selected correction tier ('Tier1', 'Tier2', or 'Tier3')
    # @param n_iter: Number of bootstrap iterations
    # @param tier2_me: Mean error for Tier2 correction (optional)
    # @param tier3_coeffs: Plane coefficients (a, b, c) for Tier3 (optional)
    # @returns: dict containing bootstrap statistics (mean, std, CI lower/upper, samples)
    #           OR None if bootstrap cannot be performed
    # note:
    #   - For Tier3 with only 3 points, bootstrap is unreliable (resample may have <3 unique points)
    #   - In this case, returns None

    # Set random seed for reproducibility
    np.random.seed(42)

    # Remove NaN values
    valid = vgcp_df.dropna(subset=['E', 'N', 'Residual']).copy()
    n = len(valid)
    
    # Check if bootstrap is reliable for Tier 3
    if tier == 'Tier3' and n < 4:
        logging.warning(f"Cannot perform reliable bootstrap for Tier 3 with only {n} points (need 4+)")
        logging.warning("Skipping bootstrap validation - correction results are still valid")
        return None

    # Array to store bootstrap RMSE values
    boot_rmse = np.zeros(n_iter)

    # Perform bootstrap iterations
    for i in range(n_iter):
        # Resample with replacement
        sample = valid.sample(n=n, replace=True).reset_index(drop=True)

        # Apply correction based on tier
        if tier == 'Tier1':
            # No correction - just calculate RMSE
            boot_rmse[i] = np.sqrt(np.mean(sample['Residual'].values**2))

        elif tier == 'Tier2':
            # Recalculate ME on bootstrap sample
            me = sample['Residual'].mean()

            # Apply correction
            corrected = apply_vertical_shift(snow_data, me, snow_meta['nodata'])

            # Extract values at sample points
            coords = np.column_stack([sample['E'].values, sample['N'].values])
            z_corr = extract_at_points(corrected, snow_meta['transform'], coords)

            # Calculate residuals
            res = z_corr - sample['Z_bare'].values
            boot_rmse[i] = np.sqrt(np.mean(res**2))

        elif tier == 'Tier3':
            try:
                # Refit plane on bootstrap sample
                a, b, c = fit_plane(sample)

                # Create correction surface
                surf = create_correction_surface(snow_meta['shape'], snow_meta['transform'], a, b, c)

                # Apply correction
                corrected = apply_plane_correction(snow_data, surf, snow_meta['nodata'])

                # Extract values at sample points
                coords = np.column_stack([sample['E'].values, sample['N'].values])
                z_corr = extract_at_points(corrected, snow_meta['transform'], coords)

                # Calculate residuals
                res = z_corr - sample['Z_bare'].values
                boot_rmse[i] = np.sqrt(np.mean(res**2))

            except Exception as e:
                # If iteration fails (e.g., singular matrix), use NaN
                boot_rmse[i] = np.nan
                # log debug message for failed iteration
                logging.debug(f"Bootstrap iteration {i+1} failed: {e}")

    # Remove failed iterations
    valid_samples = boot_rmse[~np.isnan(boot_rmse)]

    # Calculate bootstrap statistics
    boot_stats = {
        'mean': float(np.mean(valid_samples)),
        'std': float(np.std(valid_samples, ddof=1)),
        'ci_lower': float(np.percentile(valid_samples, 2.5)),
        'ci_upper': float(np.percentile(valid_samples, 97.5)),
        'samples': valid_samples
    }

    logging.info(f"Bootstrap: mean={boot_stats['mean']:.4f}, "
                f"CI=[{boot_stats['ci_lower']:.4f}, {boot_stats['ci_upper']:.4f}]")

    return boot_stats


def evaluate_tiers(vgcp_df, snow_data, snow_meta, config):
    # Evaluate all correction tiers and select best approach
    # @param vgcp_df: DataFrame with E, N, Z_bare, Residual columns
    # @param snow_data: Original snow-on DSM numpy array
    # @param snow_meta: dict containing raster metadata
    # @param config: Configuration dictionary
    # @returns: dict containing statistics for all tiers and selected tier

    results = {}

    # Check if manual mode is enabled
    manual_mode = config['thresholds'].get('correction_mode', 'auto')

    # Get thresholds from config
    tier1_rmse_max = config['thresholds']['tier1_rmse_max']
    tier2_imp_min = config['thresholds']['tier2_improvement_min']

    # -------------------------------------------------------------------------
    # Tier 1: PPK-only (no correction)
    # -------------------------------------------------------------------------
    tier1_stats = calc_stats(vgcp_df['Residual'].values)
    results['tier1_stats'] = tier1_stats

    logging.info(f"Tier 1 RMSE: {tier1_stats['RMSE']:.4f} m")

    # If manual mode is set to ppk/tier1, use it and return
    if manual_mode.lower() in ['ppk', 'tier1', 'none']:
        results['selected'] = 'Tier1'
        results['reason'] = "Manual mode: PPK-only selected in config"
        logging.info(f"Selected: {results['reason']}")
        return results

    # Check if Tier 1 is acceptable (only in auto mode)
    if manual_mode.lower() == 'auto' and tier1_stats['RMSE'] <= tier1_rmse_max:
        results['selected'] = 'Tier1'
        results['reason'] = f"Tier 1 RMSE ({tier1_stats['RMSE']:.4f}) <= threshold ({tier1_rmse_max:.4f})"
        logging.info(f"Selected: {results['reason']}")
        return results

    # -------------------------------------------------------------------------
    # Tier 2: Vertical shift correction
    # -------------------------------------------------------------------------
    # Apply vertical shift (remove bias)
    tier2_corrected = apply_vertical_shift(snow_data, tier1_stats['ME'], snow_meta['nodata'])

    # Extract corrected values at vGCP locations
    coords = np.column_stack([vgcp_df['E'].values, vgcp_df['N'].values])
    z_tier2 = extract_at_points(tier2_corrected, snow_meta['transform'], coords)

    # Calculate residuals for Tier 2
    tier2_res = z_tier2 - vgcp_df['Z_bare'].values
    tier2_stats = calc_stats(tier2_res)

    # Store results including per-point residuals
    results['tier2_stats'] = tier2_stats
    results['tier2_corrected'] = tier2_corrected
    results['tier2_residuals'] = tier2_res  # Store per-point residuals

    # Calculate improvement percentage
    tier2_imp = (tier1_stats['RMSE'] - tier2_stats['RMSE']) / tier1_stats['RMSE']

    logging.info(f"Tier 2 RMSE: {tier2_stats['RMSE']:.4f} m ({tier2_imp*100:.1f}% improvement)")

    # If manual mode is set to vsc/tier2, use it and return
    if manual_mode.lower() in ['vsc', 'tier2', 'vertical']:
        results['selected'] = 'Tier2'
        results['reason'] = "Manual mode: Vertical Shift Correction selected in config"
        logging.info(f"Selected: {results['reason']}")
        return results

    # -------------------------------------------------------------------------
    # Tier 3: Planar trend correction
    # -------------------------------------------------------------------------
    try:
        # Fit plane to residuals
        a, b, c = fit_plane(vgcp_df)

        # Create correction surface
        correction_surf = create_correction_surface(snow_meta['shape'], snow_meta['transform'], a, b, c)

        # Apply plane correction
        tier3_corrected = apply_plane_correction(snow_data, correction_surf, snow_meta['nodata'])

        # Extract corrected values at vGCP locations
        z_tier3 = extract_at_points(tier3_corrected, snow_meta['transform'], coords)

        # Calculate residuals for Tier 3
        tier3_res = z_tier3 - vgcp_df['Z_bare'].values
        tier3_stats = calc_stats(tier3_res)

        # Store results including per-point residuals
        results['tier3_stats'] = tier3_stats
        results['tier3_corrected'] = tier3_corrected
        results['tier3_coeffs'] = (a, b, c)
        results['tier3_residuals'] = tier3_res  # Store per-point residuals

        logging.info(f"Tier 3 RMSE: {tier3_stats['RMSE']:.4f} m")

    except Exception as e:
        # Tier 3 failed (e.g., not enough points for plane fit)
        # log warning for tier 3 failure
        logging.warning(f"Tier 3 failed: {e}")
        results['tier3_stats'] = None

    # If manual mode is set to ptc/tier3, use it and return
    if manual_mode.lower() in ['ptc', 'tier3', 'planar', 'tilt']:
        if results.get('tier3_stats'):
            results['selected'] = 'Tier3'
            results['reason'] = "Manual mode: Planar Trend Correction selected in config"
        else:
            results['selected'] = 'Tier2'
            results['reason'] = "Manual mode requested Tier3 but it failed, using Tier2"
        logging.info(f"Selected: {results['reason']}")
        return results

    # -------------------------------------------------------------------------
    # Auto mode tier selection
    # -------------------------------------------------------------------------
    if tier2_imp < tier2_imp_min:
        # Tier 2 doesn't meet improvement threshold
        if results.get('tier3_stats') and tier3_stats['RMSE'] < tier2_stats['RMSE']:
            results['selected'] = 'Tier3'
            results['reason'] = f"Tier 2 improvement ({tier2_imp*100:.1f}%) < threshold, Tier 3 better"
        else:
            results['selected'] = 'Tier1'
            results['reason'] = "No tier met improvement criteria"

    else:
        # Tier 2 meets improvement threshold
        if results.get('tier3_stats'):
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

    # Ensure output directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Write raster
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=meta['crs'],
        transform=meta['transform'],
        nodata=meta['nodata'],
        compress='LZW'
    ) as dst:
        dst.write(data, 1)

    logging.info(f"Saved: {Path(path).name}")


def calc_snow_depth(snow_corrected, bare, nodata=None):
    # Calculate snow depth from corrected snow-on and bare ground DSMs
    # @param snow_corrected: Corrected snow-on DSM numpy array
    # @param bare: Bare ground DSM numpy array
    # @param nodata: Nodata value to preserve (optional)
    # @returns: numpy array - snow depth raster (negative depths set to zero)

    # Calculate snow depth
    depth = snow_corrected - bare

    # Handle nodata
    if nodata is not None:
        # Identify nodata pixels
        mask = (snow_corrected == nodata) | (bare == nodata)
        depth[mask] = nodata

        # Set negative depths to zero (only for valid data)
        depth[(~mask) & (depth < 0)] = 0
    else:
        # No nodata - just set negatives to zero
        depth[depth < 0] = 0

    return depth


def create_stats_csv(config, results, validation, snow_filename, output_path):
    # Create CSV file with statistics summary
    # @param config: Configuration dictionary
    # @param results: Tier evaluation results dict
    # @param validation: Validation statistics dict (LOO, bootstrap)
    # @param snow_filename: Input snow-on DSM filename
    # @param output_path: Output CSV file path
    # @returns: None

    # Initialize data dictionary
    data = {
        'DSM': snow_filename,
        'AOI': config['paths']['aoi_name'],
        'Selected_Tier': results['selected'],
        'Reason': results['reason'],
        'n_vGCPs': results['tier1_stats']['n']
    }

    # Add statistics for each tier
    for tier_name in ['tier1', 'tier2', 'tier3']:
        if f'{tier_name}_stats' in results and results[f'{tier_name}_stats']:
            stats = results[f'{tier_name}_stats']
            prefix = tier_name.replace('tier', 'Tier')

            # Add all statistics
            data[f'{prefix}_ME'] = stats['ME']
            data[f'{prefix}_RMSE'] = stats['RMSE']
            data[f'{prefix}_NMAD'] = stats['NMAD']
            data[f'{prefix}_MAE'] = stats['MAE']

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


def create_plots(vgcp_df, results, validation, output_dir, filename):
    # Create visualization plots for correction results
    # @param vgcp_df: DataFrame with vGCP data
    # @param results: Tier evaluation results dict
    # @param validation: Validation statistics dict (LOO, bootstrap)
    # @param output_dir: Output directory for plots
    # @param filename: Base filename for plots
    # @returns: None

    # Set plotting style
    sns.set_style("whitegrid")

    # Get base filename
    base = Path(filename).stem
    selected = results['selected']

    # Create residual comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Tier 1 (before correction)
    tier1_res = vgcp_df['Residual'].dropna().values
    x_positions = range(1, len(tier1_res) + 1)
    ax1.bar(x_positions, tier1_res, color='coral', edgecolor='black', alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', lw=2)
    ax1.set_xlabel('vGCP Index')
    ax1.set_ylabel('Residual (m)')
    ax1.set_title(f'Tier 1 (PPK-only): RMSE = {results["tier1_stats"]["RMSE"]:.4f} m')
    ax1.set_xticks(x_positions)
    ax1.grid(alpha=0.3)

    # Selected tier (after correction)
    # Use actual tier residuals (not LOO residuals) to match CSV output
    if selected == 'Tier1':
        selected_res = vgcp_df['Residual'].dropna().values
    elif selected == 'Tier2':
        selected_res = results.get('tier2_residuals', vgcp_df['Residual'].dropna().values)
    else:  # Tier3
        selected_res = results.get('tier3_residuals', vgcp_df['Residual'].dropna().values)
    
    selected_rmse = results[f'{selected.lower()}_stats']['RMSE']

    x_positions = range(1, len(selected_res) + 1)
    ax2.bar(x_positions, selected_res, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', lw=2)
    ax2.set_xlabel('vGCP Index')
    ax2.set_ylabel('Residual (m)')
    ax2.set_title(f'{selected}: RMSE = {selected_rmse:.4f} m')
    ax2.set_xticks(x_positions)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / f'{base}_residuals.png', dpi=300)
    plt.close()

    # Create bootstrap distribution plot (if available)
    if validation.get('bootstrap'):
        fig, ax = plt.subplots(figsize=(10, 6))

        boot = validation['bootstrap']

        # Histogram
        ax.hist(boot['samples'], bins=30, color='steelblue',
               edgecolor='black', alpha=0.7, density=True)

        # Add lines for mean and CI
        ax.axvline(boot['mean'], color='red', linestyle='--', lw=2,
                  label=f"Mean = {boot['mean']:.4f} m")
        ax.axvline(boot['ci_lower'], color='orange', linestyle=':', lw=2, label='95% CI')
        ax.axvline(boot['ci_upper'], color='orange', linestyle=':', lw=2)

        ax.set_xlabel('RMSE (m)')
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap RMSE Distribution ({selected})')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{base}_bootstrap.png', dpi=300)
        plt.close()

    logging.info(f"Saved plots: {base}_*.png")


def process_single_dsm(config, folders, snow_file):
    # Process a single snow-on DSM file
    # @param config: Configuration dictionary
    # @param folders: dict of folder paths
    # @param snow_file: Snow-on DSM filename
    # @returns: dict - processing results

    logging.info("="*70)
    logging.info(f"Processing: {snow_file}")
    logging.info("="*70)

    # Build file paths
    bare_path = folders['data_bare'] / config['paths']['bare_ground_file']
    vgcp_path = folders['data_vgcp'] / config['paths']['vgcp_file']
    snow_path = folders['data_snow'] / snow_file

    # Load data
    logging.info("Loading data...")
    bare_data, bare_meta = load_raster(str(bare_path))
    snow_data, snow_meta = load_raster(str(snow_path))
    vgcp_gdf = load_vgcp(str(vgcp_path))

    # Note: CRS validation disabled for compound CRS
    logging.info("Skipping CRS validation (compound CRS)")

    # Extract values at vGCP locations
    logging.info("Extracting values at vGCP locations...")
    coords = np.column_stack([vgcp_gdf['E'].values, vgcp_gdf['N'].values])
    snow_z = extract_at_points(snow_data, snow_meta['transform'], coords)

    # Build DataFrame with all vGCP data
    vgcp_df = pd.DataFrame({
        'E': vgcp_gdf['E'].values,
        'N': vgcp_gdf['N'].values,
        'Z_bare': vgcp_gdf['Elevation'].values,
        'Z_snow': snow_z,
        'Residual': snow_z - vgcp_gdf['Elevation'].values
    })

    # Evaluate correction tiers
    logging.info("Evaluating correction tiers...")
    results = evaluate_tiers(vgcp_df, snow_data, snow_meta, config)

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
    save_raster(corrected_dsm, snow_meta, str(folders['output_corrected'] / output_name))

    # Calculate and save snow depth
    depth_name = generate_filename(snow_file, selected, 'snowHeight')
    snow_depth = calc_snow_depth(corrected_dsm, bare_data, snow_meta['nodata'])
    save_raster(snow_depth, snow_meta, str(folders['output_corrected'] / depth_name))

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
