"""
UAS Snow Depth Validation

Author: Valerie Foley, valerie.foley@state.co.us
Last Updated: 12/17/2025

Description:
    Standalone validation tool for accuracy assessment of snow-on DSMs.
    Can be run independently after corrections have been applied to validate
    accuracy using vGCPs. Supports both single file and batch folder processing.

Features:
    - Basic accuracy statistics - RMSE, ME, NMAD, MAE
    - Leave-One-Out cross-validation
    - Bootstrap uncertainty estimation
    - Validation plots
    - Single file and batch processing

Usage:
    Single file:
        python uas_validation.py --bare bare.tif --snow corrected.tif --vgcp vgcp.shp --output stats.csv

    Batch folder:
        python uas_validation.py --bare bare.tif --snow folder/ --vgcp vgcp.shp --output stats.csv
"""

# import libraries
import sys
import argparse
from pathlib import Path
import logging

import rasterio
from rasterio.transform import rowcol
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- SETUP --------
def setup_logging(verbose=False):
    # Configure logging for the validation tool
    # @param verbose (bool): If True, set logging to DEBUG level
    # @return: None
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args():
    # Parse command line arguments
    # @return argparse.Namespace: Parsed arguments
    # Arguments:
    #     --bare: Path to bare-ground DSM (required)
    #     --snow: Path to snow-on DSM file or folder (required)
    #     --vgcp: Path to vGCP shapefile (required)
    #     --output: Path to output CSV file (required)
    #     --bootstrap: Number of bootstrap iterations (0=disabled)
    #     --plot-dir: Directory for plots (optional)
    #     --label: Label for output files
    #     --verbose: Enable verbose logging

    parser = argparse.ArgumentParser(
        description='Standalone UAS Snow Depth Validation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file validation
  python uas_validation.py \\
    --bare data/SiteA/bareGround/bare.tif \\
    --snow outputs/SiteA/corrected/snow_corrected.tif \\
    --vgcp data/SiteA/vGCP/vgcp.shp \\
    --output validation.csv \\
    --bootstrap 250

  # Single file with per-point residuals
  python uas_validation.py \\
    --bare data/SiteA/bareGround/bare.tif \\
    --snow outputs/SiteA/corrected/snow_corrected.tif \\
    --vgcp data/SiteA/vGCP/vgcp.shp \\
    --output validation.csv \\
    --bootstrap 250 \\
    --save-residuals

  # Batch folder validation
  python uas_validation.py \\
    --bare data/SiteA/bareGround/bare.tif \\
    --snow outputs/SiteA/corrected/ \\
    --vgcp data/SiteA/vGCP/vgcp.shp \\
    --output validation_batch.csv \\
    --bootstrap 250 \\
    --save-residuals
        """
    )

    parser.add_argument('--bare', required=True, help='Bare-ground DSM path')
    parser.add_argument('--snow', required=True, help='Snow-on DSM path or folder')
    parser.add_argument('--vgcp', required=True, help='vGCP shapefile path')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--bootstrap', type=int, default=0, help='Bootstrap iterations (0=disabled)')
    parser.add_argument('--plot-dir', default=None, help='Directory for plots (optional)')
    parser.add_argument('--label', default='Validation', help='Label for output files')
    parser.add_argument('--save-residuals', action='store_true', help='Save per-point residuals to CSV')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


# -------- DATA LOADING --------

def load_raster(path):
    # Load raster file and return data with metadata
    #
    # @param path (str): Path to raster file
    # @return tuple: (data, metadata)
    #                data (np.ndarray): 2D raster array
    #                metadata (dict): Dictionary with 'transform' and 'nodata'
    #
    # note: Only reads first band, minimal metadata for validation

    with rasterio.open(path) as src:
        data = src.read(1)
        meta = {
            'transform': src.transform,
            'nodata': src.nodata
        }

    return data, meta


def load_vgcp(path):
    # Load vGCP shapefile and validate fields
    #
    # @param path (str): Path to vGCP shapefile
    # @return gpd.GeoDataFrame: GeoDataFrame with vGCP points
    #
    # note: required fields in vGCP shapefile: E, N, Elevation

    # Load shapefile
    gdf = gpd.read_file(path)

    # Check for required fields
    required = ['E', 'N', 'Elevation']
    missing = [f for f in required if f not in gdf.columns]

    if missing:
        # throw error if fields missing
        raise ValueError(f"vGCP shapefile missing required fields: {missing}")

    return gdf


def extract_at_points(data, transform, coords):
    # Extract raster values at given XY coordinates
    # @param data (np.ndarray): 2D raster array
    # @param transform (Affine): Rasterio affine transform
    # @param coords (np.ndarray): Nx2 array of (X, Y) coordinates
    # @return np.ndarray: Array of extracted values (NaN if out of bounds)
    #
    # note:
    #   - uses nearest neighbor sampling
    #   - returns NaN for points outside raster

    values = np.full(len(coords), np.nan)

    # Extract value at each point
    for i, (x, y) in enumerate(coords):
        # Convert XY to row/col
        row, col = rowcol(transform, x, y)

        # Check if within bounds
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            values[i] = data[row, col]

    return values

# -------- STATISTICS --------

def calc_stats(residuals):
    # Calculate accuracy statistics from residuals
    # @param residuals (np.ndarray): Array of residual values
    # @return dict: Dictionary of statistics:
    #             - n: Number of valid points
    #             - ME: Mean Error
    #             - RMSE: Root Mean Square Error
    #             - MAE: Mean Absolute Error
    #             - StdDev: Standard deviation
    #             - Min: Minimum residual
    #             - Max: Maximum residual
    #             - NMAD: Normalized Median Absolute Deviation

    # Remove NaN values
    valid = residuals[~np.isnan(residuals)]

    if len(valid) == 0:
        # throw error if no valid residuals
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

    # Calculate NMAD
    # NMAD is sensitive to outliers and robust measure of spread (not as useful for small samples)
    median = np.median(valid)
    mad = np.median(np.abs(valid - median))
    # NMAD = 1.4826 * median(|residuals - median|)
    stats['NMAD'] = float(1.4826 * mad)

    return stats


def loo_validation(residuals):
    # Perform Leave-One-Out cross-validation
    # @param residuals (np.ndarray): Array of residuals
    # @return dict: LOO statistics (same format as calc_stats)
    # note:
    #    For each point:
    #    1. Remove it from dataset
    #    2. Calculate mean of remaining points
    #    3. Predicted correction = mean of remaining
    #    4. LOO residual = actual - predicted
    #    essentialy testing how well the mean correction generalizes

    # Remove NaN values
    valid = residuals[~np.isnan(residuals)]
    n = len(valid)

    if n < 3:
        # throw error is fewer than 3 valid points
        raise ValueError(f"Need at least 3 valid points for LOO, have {n}")

    # Array for LOO residuals
    loo_res = np.zeros(n)

    # Perform LOO for each point
    for i in range(n):
        # Training set (all except i)
        train = np.delete(valid, i)

        # Predicted correction
        correction = np.mean(train)

        # LOO residual
        loo_res[i] = valid[i] - correction

    # Calculate statistics on LOO residuals
    return calc_stats(loo_res)


def bootstrap_uncertainty(residuals, n_iter=250):
    # Estimate uncertainty using bootstrap resampling
    # @param residuals (np.ndarray): Array of residuals
    # @param n_iter (int): Number of bootstrap iterations
    # @return dict: Bootstrap statistics
    #               - mean: Mean RMSE
    #               - std: Standard deviation of RMSE
    #               - ci_lower: Lower 95% CI
    #               - ci_upper: Upper 95% CI
    #               - samples: Array of all RMSE samples
    # note:
        # For each iteration:
        # 1. Resample with replacement
        # 2. Calculate RMSE
        # 3. Build distribution
        #
        # 95% CI from 2.5 and 97.5 percentiles

    # Set random seed for reproducibility
    np.random.seed(42)

    # Remove NaN values
    valid = residuals[~np.isnan(residuals)]
    n = len(valid)

    if n < 3:
        # throw error is fewer than 3 valid points
        raise ValueError(f"Need at least 3 valid points for bootstrap, have {n}")

    # Array for bootstrap RMSE values
    boot_rmse = np.zeros(n_iter)

    # Perform bootstrap iterations
    for i in range(n_iter):
        # Resample with replacement
        sample = np.random.choice(valid, size=n, replace=True)

        # Calculate RMSE on sample
        boot_rmse[i] = np.sqrt(np.mean(sample**2))

    # Calculate bootstrap statistics
    return {
        'mean': float(np.mean(boot_rmse)),
        'std': float(np.std(boot_rmse, ddof=1)),
        'ci_lower': float(np.percentile(boot_rmse, 2.5)),
        'ci_upper': float(np.percentile(boot_rmse, 97.5)),
        'samples': boot_rmse
    }


# -------- PLOTS --------

def create_plots(residuals, loo_stats, boot_stats, plot_dir, label):
    # Create validation plots
    # @param residuals (np.ndarray): Array of residuals
    # @param loo_stats (dict): LOO statistics
    # @param boot_stats (dict or None): Bootstrap statistics
    # @param plot_dir (str): Output directory
    # @param label (str): Label for filenames
    # @return: None
    # Creates:
    #     1. Residual histogram
    #     2. Bootstrap distribution (if bootstrap was run from the config file)

    # Create output directory
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Set plotting style
    sns.set_style("whitegrid")

    # Get valid residuals
    valid_res = residuals[~np.isnan(residuals)]

    # ---- Plot 1: Residual histogram ----
    fig, ax = plt.subplots(figsize=(8, 6))

    # Histogram
    ax.hist(valid_res, bins=20, edgecolor='black', alpha=0.7, color='steelblue')

    # Add reference lines
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.axvline(np.mean(valid_res), color='blue', linestyle='--', lw=2,
              label=f'Mean = {np.mean(valid_res):.3f} m')

    # Labels and formatting
    ax.set_xlabel('Residual (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{label} - Residual Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(plot_dir / f'{label}_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Plot 2: Bootstrap distribution ----
    if boot_stats:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram
        ax.hist(boot_stats['samples'], bins=30, edgecolor='black',
               alpha=0.7, density=True, color='steelblue')

        # Add reference lines
        ax.axvline(boot_stats['mean'], color='red', linestyle='--', lw=2,
                  label=f"Mean = {boot_stats['mean']:.4f} m")
        ax.axvline(boot_stats['ci_lower'], color='orange', linestyle=':', lw=2,
                  label='95% CI')
        ax.axvline(boot_stats['ci_upper'], color='orange', linestyle=':', lw=2)

        # Labels and formatting
        ax.set_xlabel('RMSE (m)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{label} - Bootstrap RMSE Distribution',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        # Save
        plt.tight_layout()
        plt.savefig(plot_dir / f'{label}_bootstrap.png', dpi=300, bbox_inches='tight')
        plt.close()

    logging.info(f"Saved plots: {label}_*.png")

def save_point_residuals(vgcp_gdf, bare_z, snow_z, residuals, label, output_path):
    # Save per-point residuals to CSV
    # @param vgcp_gdf (gpd.GeoDataFrame): GeoDataFrame with vGCP points
    # @param bare_z (np.ndarray): Bare-ground elevations at vGCP locations
    # @param snow_z (np.ndarray): Snow-on elevations at vGCP locations
    # @param residuals (np.ndarray): Calculated residuals (snow_z - bare_z)
    # @param label (str): Label for the validation
    # @param output_path (str): Path to output CSV file
    # @return: None
    
    # Build output dataframe
    residuals_df = pd.DataFrame({
        'Point_ID': range(1, len(vgcp_gdf) + 1),
        'E': vgcp_gdf['E'].values,
        'N': vgcp_gdf['N'].values,
        'Z_bare': bare_z,
        'Z_snow': snow_z,
        'Residual': residuals,
        'Label': label
    })
    
    # Save to CSV
    residuals_df.to_csv(output_path, index=False, float_format='%.6f')
    logging.info(f"Saved point residuals: {Path(output_path).name}")

# -------- SINGLE FILE VALIDATION --------

def validate_single(bare_path, snow_path, vgcp_path, label, bootstrap_iter=0, plot_dir=None, save_residuals=False, residuals_output=None):
    # Validate a single DSM
    # @param bare_path (str): Path to bare-ground DSM
    # @param snow_path (str): Path to snow-on DSM
    # @param vgcp_path (str): Path to vGCP shapefile
    # @param label (str): Label for output
    # @param bootstrap_iter (int): Number of bootstrap iterations (0=disabled)
    # @param plot_dir (str): Directory for plots (optional)
    # @param save_residuals (bool): Whether to save per-point residuals CSV
    # @param residuals_output (str): Path for residuals CSV output (optional)
    # @return dict: Dictionary with all validation results
    # note:
    #   validation workflow:
        # 1. Load data
        # 2. Extract values at vGCP locations
        # 3. Calculate residuals
        # 4. Calculate statistics
        # 5. Run LOO validation
        # 6. Run bootstrap (if enabled)
        # 7. Create plots (if enabled)
        # 8. Save per-point residuals (if enabled)

    logging.info(f"Validating: {Path(snow_path).name}")

    # ---- Load data ----
    bare_data, bare_meta = load_raster(bare_path)
    snow_data, snow_meta = load_raster(snow_path)
    vgcp_gdf = load_vgcp(vgcp_path)

    # ---- Extract values at vGCP locations ----
    coords = np.column_stack([vgcp_gdf['E'].values, vgcp_gdf['N'].values])

    bare_z = extract_at_points(bare_data, bare_meta['transform'], coords)
    snow_z = extract_at_points(snow_data, snow_meta['transform'], coords)

    # ---- Calculate residuals ----
    residuals = snow_z - bare_z
    n_valid = np.sum(~np.isnan(residuals))

    logging.info(f"Calculated {n_valid} valid residuals")

    # ---- Calculate basic statistics ----
    stats = calc_stats(residuals)
    logging.info(f"RMSE: {stats['RMSE']:.4f} m, ME: {stats['ME']:.4f} m")

    # ---- LOO validation ----
    loo_stats = loo_validation(residuals)
    logging.info(f"LOO RMSE: {loo_stats['RMSE']:.4f} m")

    # ---- Bootstrap ----
    boot_stats = None
    if bootstrap_iter > 0:
        boot_stats = bootstrap_uncertainty(residuals, bootstrap_iter)
        logging.info(f"Bootstrap: {boot_stats['mean']:.4f} m, "
                    f"CI=[{boot_stats['ci_lower']:.4f}, {boot_stats['ci_upper']:.4f}]")

    # ---- Create plots ----
    if plot_dir:
        create_plots(residuals, loo_stats, boot_stats, plot_dir, label)
    
    # ---- Save per-point residuals ----
    if save_residuals and residuals_output:
        save_point_residuals(vgcp_gdf, bare_z, snow_z, residuals, label, residuals_output)

    # ---- Combine results ----
    results = {
        'Label': label,
        'DSM': Path(snow_path).name,
        **{f'{k}': v for k, v in stats.items()},
        **{f'LOO_{k}': v for k, v in loo_stats.items()}
    }

    # Add bootstrap statistics
    if boot_stats:
        results.update({
            'Bootstrap_Mean': boot_stats['mean'],
            'Bootstrap_Std': boot_stats['std'],
            'Bootstrap_CI_Lower': boot_stats['ci_lower'],
            'Bootstrap_CI_Upper': boot_stats['ci_upper']
        })

    return results

# -------- BATCH VALIDATION --------

def validate_batch(bare_path, snow_folder, vgcp_path, bootstrap_iter=0, plot_dir=None, save_residuals=False, residuals_dir=None):
    # Validate multiple DSMs in a folder
    # @param bare_path (str): Path to bare-ground DSM
    # @param snow_folder (str): Path to folder with snow-on DSMs
    # @param vgcp_path (str): Path to vGCP shapefile
    # @param bootstrap_iter (int): Number of bootstrap iterations
    # @param plot_dir (str): Directory for plots (optional)
    # @param save_residuals (bool): Whether to save per-point residuals
    # @param residuals_dir (str): Directory for residuals CSV files (optional)
    # @return list: List of result dictionaries, one per DSM
    # note:
    #     - Processes all .tif and .tiff files in folder
    #     - Log but dont stop error - if one file fails, continue with others
    #     - Returns results for validations

    snow_folder = Path(snow_folder)

    # Find all DSM files in folder
    snow_files = []
    for ext in ['*.tif', '*.tiff']:
        snow_files.extend(list(snow_folder.glob(ext)))

    if not snow_files:
        raise ValueError(f"No .tif/.tiff files found in {snow_folder}")

    # Sort for consistent ordering
    snow_files.sort()
    logging.info(f"Found {len(snow_files)} DSM file(s) to validate")

    # Process each DSM
    results = []
    for i, snow_file in enumerate(snow_files, 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"DSM {i}/{len(snow_files)}")
        logging.info(f"{'='*70}")

        try:
            # Use filename (w/o extension) as label
            label = Path(snow_file).stem
            
            # Set up residuals output path if enabled
            residuals_output = None
            if save_residuals and residuals_dir:
                Path(residuals_dir).mkdir(parents=True, exist_ok=True)
                residuals_output = str(Path(residuals_dir) / f"{label}_point_residuals.csv")

            # Validate current DSM
            result = validate_single(
                bare_path,
                str(snow_file),
                vgcp_path,
                label,
                bootstrap_iter,
                plot_dir,
                save_residuals,
                residuals_output
            )

            results.append(result)

        except Exception as e:
            # Log error if a singel file failed but continue with next file(s)
            logging.error(f"Failed to validate {snow_file.name}: {e}")

    return results

# -------- MAIN --------

def main():
    # Main entry point for standalone validation
    # @param None (uses command line arguments and use config file)
    # @return: None (exits with status code)
    #
    # note:
    #     - Parses arguments
    #     - Decides single or batch mode
    #     - Runs validation
    #     - Saves results to CSV

    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        logging.info("="*70)
        logging.info("UAS Snow Depth Validation (Standalone)")
        logging.info("="*70)

        # Determine processing mode
        snow_path = Path(args.snow)
        
        # Determine residuals output path
        residuals_dir = None
        residuals_output = None
        if args.save_residuals:
            output_dir = Path(args.output).parent
            residuals_dir = output_dir / 'point_residuals'
            residuals_dir.mkdir(parents=True, exist_ok=True)

        if snow_path.is_file():
            # Single file mode
            logging.info("Running in SINGLE file mode")
            
            # Set residuals output path
            if args.save_residuals:
                residuals_output = str(residuals_dir / f"{args.label}_point_residuals.csv")
            
            result = validate_single(
                args.bare,
                str(snow_path),
                args.vgcp,
                args.label,
                args.bootstrap,
                args.plot_dir,
                args.save_residuals,
                residuals_output
            )
            results = [result]

        elif snow_path.is_dir():
            # Batch mode
            logging.info("Running in BATCH mode")
            results = validate_batch(
                args.bare,
                str(snow_path),
                args.vgcp,
                args.bootstrap,
                args.plot_dir,
                args.save_residuals,
                residuals_dir
            )

        else:
            # throw error if the snow-on DSM path is invalid
            raise ValueError(f"Invalid snow path: {snow_path}")

        # Save results to CSV
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False, float_format='%.6f') # float_format for precision -- 6 decimal places
        logging.info(f"\nResults Saved: {args.output}")
        
        if args.save_residuals:
            logging.info(f"Per-point residuals saved to: {residuals_dir}")

        # woohoo success!
        logging.info("\n" + "="*70)
        logging.info("\nValidation Complete")
        logging.info("="*70)

    except Exception as e:
        # Error handling
        logging.error(f"\nValidation Failed: {e}")

        # Print full traceback if verbose
        if args.verbose:
            import traceback
            traceback.print_exc()

        sys.exit(1)


if __name__ == "__main__":
    main()
