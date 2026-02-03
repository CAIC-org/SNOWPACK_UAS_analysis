"""
UAS Snow Height Calculator - Part 2: Snow Depth Calculation with ArcPy

Author: Valerie Foley
Last Updated: 1/29/2026

Description:
    Calculates snow depth from corrected DSMs using ArcPy Raster Calculator.
    This script automatically handles raster alignment, resampling, and extent matching.
    
    Run this AFTER uas_snow_correction.py has created corrected DSMs.

Input:
    - Corrected DSMs from: outputs/{aoi_name}/corrected/
    - Bare ground DSM from: data/{aoi_name}/bareGround/

Output:
    - Snow depth rasters → outputs/{aoi_name}/snowHeight/

Usage:
    Run from ArcGIS Pro Python Command Prompt:
    python uas_snow_height_calculator.py --config config.yaml

Requirements:
    - ArcGIS Pro with Spatial Analyst extension
    - Run from ArcGIS Pro Python Command Prompt
"""

# --------- Load Libraries --------
import os
import sys
import yaml
import argparse
from pathlib import Path
import logging

# ArcPy imports
try:
    import arcpy
    from arcpy.sa import *
    arcpy.CheckOutExtension("Spatial")
    ARCPY_AVAILABLE = True
except Exception as e:
    ARCPY_AVAILABLE = False
    print(f"ERROR: ArcPy not available: {e}")
    print("This script must be run from ArcGIS Pro Python Command Prompt")
    sys.exit(1)


def setup_logging(verbose=False):
    # Configure logging
    # @param verbose: If True, DEBUG level, otherwise INFO
    # @returns: None

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path):
    # Load configuration from YAML file
    # @param config_path: Path to config YAML file
    # @returns: dict - configuration dictionary

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required = ['paths', 'crs']
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    logging.info(f"Loaded config: {config_path}")
    return config


def parse_args():
    # Parse command-line arguments
    # @param: None
    # @returns: argparse.Namespace with parsed arguments

    parser = argparse.ArgumentParser(
        description='UAS Snow Height Calculator (Part 2: ArcPy Snow Depth)'
    )
    
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--aoi', default=None, help='Override AOI name from config')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def create_output_folder(base_path, aoi_name):
    # Create output folder for snow height rasters
    # @param base_path: Base directory path
    # @param aoi_name: Name of Area of Interest
    # @returns: dict of folder paths

    folders = {
        'data_bare': base_path / 'data' / aoi_name / 'bareGround',
        'input_corrected': base_path / 'outputs' / aoi_name / 'corrected',
        'output_snowheight': base_path / 'outputs' / aoi_name / 'snowHeight'
    }
    
    # Create output folder if it doesn't exist
    folders['output_snowheight'].mkdir(parents=True, exist_ok=True)
    
    return folders


def get_corrected_dsms(corrected_folder):
    # Get list of corrected DSM files to process
    # @param corrected_folder: Path to corrected DSMs folder
    # @returns: list of file paths

    corrected_folder = Path(corrected_folder)
    
    if not corrected_folder.exists():
        raise FileNotFoundError(f"Corrected DSM folder not found: {corrected_folder}")
    
    # Find all .tif and .tiff files
    files = []
    for ext in ['*.tif', '*.tiff']:
        files.extend(list(corrected_folder.glob(ext)))
    
    if not files:
        raise ValueError(f"No corrected DSM files found in {corrected_folder}")
    
    # Sort for consistent ordering
    files.sort()
    logging.info(f"Found {len(files)} corrected DSM(s) to process")
    
    return files


def generate_output_name(corrected_dsm_name):
    # Generate output filename for snow height raster
    # @param corrected_dsm_name: Input corrected DSM filename
    # @returns: str - output filename
    # note:
    #   - Replaces '_DSM' with '_snowHeight'
    #   - Forces .tif extension (ArcPy requires .tif, not .tiff)

    stem = Path(corrected_dsm_name).stem
    
    # Replace _DSM with _snowHeight
    if stem.endswith('_DSM'):
        output_stem = stem[:-4] + '_snowHeight'
    else:
        output_stem = stem + '_snowHeight'
    
    # Force .tif extension (ArcPy doesn't like .tiff)
    return output_stem + '.tif'


def calculate_snow_depth(corrected_dsm_path, bare_dsm_path, output_path):
    # Calculate snow depth using ArcPy Raster Calculator
    # @param corrected_dsm_path: Path to corrected snow-on DSM
    # @param bare_dsm_path: Path to bare ground DSM
    # @param output_path: Path for output snow depth raster
    # @returns: True if successful, False otherwise
    # note:
    #   - Uses ArcPy's Minus tool for raster subtraction
    #   - Automatically handles raster alignment, resampling, extent matching
    #   - Preserves negative values (no special handling)

    try:
        logging.info(f"Processing: {Path(corrected_dsm_path).name}")
        
        # Load rasters
        snow_raster = Raster(str(corrected_dsm_path))
        bare_raster = Raster(str(bare_dsm_path))
        
        # Perform subtraction - ArcPy handles alignment automatically
        depth_raster = Minus(snow_raster, bare_raster)
        
        # Save output (no negative value handling)
        depth_raster.save(str(output_path))
        
        logging.info(f"  -> Saved: {Path(output_path).name}")
        return True
        
    except Exception as e:
        logging.error(f"  -> Failed: {e}")
        return False


def process_all_dsms(config, folders):
    # Process all corrected DSMs to calculate snow depth
    # @param config: Configuration dictionary
    # @param folders: Folder paths dictionary
    # @returns: dict with summary statistics

    # Get bare ground DSM path
    bare_dsm_path = folders['data_bare'] / config['paths']['bare_ground_file']
    
    if not bare_dsm_path.exists():
        raise FileNotFoundError(f"Bare ground DSM not found: {bare_dsm_path}")
    
    logging.info(f"Bare ground DSM: {bare_dsm_path.name}")
    
    # Get list of corrected DSMs
    corrected_dsms = get_corrected_dsms(folders['input_corrected'])
    
    # Process each corrected DSM
    results = []
    
    for i, corrected_dsm_path in enumerate(corrected_dsms, 1):
        logging.info(f"\n[{i}/{len(corrected_dsms)}] {corrected_dsm_path.name}")
        
        # Generate output filename
        output_name = generate_output_name(corrected_dsm_path.name)
        output_path = folders['output_snowheight'] / output_name
        
        # Calculate snow depth
        success = calculate_snow_depth(
            str(corrected_dsm_path),
            str(bare_dsm_path),
            str(output_path)
        )
        
        results.append({
            'input': corrected_dsm_path.name,
            'output': output_name,
            'success': success
        })
    
    # Summary
    success_count = sum(1 for r in results if r['success'])
    
    logging.info("\n" + "="*70)
    logging.info(f" Processed: {success_count}/{len(results)} successful")
    logging.info("="*70)
    
    if success_count < len(results):
        logging.warning(f"{len(results) - success_count} file(s) failed")
    
    return results


def main():
    # Main workflow function
    # @param: None
    # @returns: None

    args = parse_args()
    setup_logging(args.verbose)
    
    # Check ArcPy availability
    if not ARCPY_AVAILABLE:
        logging.error("ArcPy is not available.")
        logging.error("Please run this script from ArcGIS Pro Python Command Prompt.")
        sys.exit(1)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override AOI name if provided
        if args.aoi:
            config['paths']['aoi_name'] = args.aoi
        
        # Setup folders
        base_path = Path(args.config).parent
        aoi_name = config['paths']['aoi_name']
        folders = create_output_folder(base_path, aoi_name)
        
        logging.info(f"\nAOI: {aoi_name}")
        logging.info(f"Input folder: {folders['input_corrected']}")
        logging.info(f"Output folder: {folders['output_snowheight']}\n")
        
        # Process all corrected DSMs
        results = process_all_dsms(config, folders)
        
        # Success message
        logging.info("\n" + "="*70)
        logging.info(" Snow Height Calculation Complete!")
        logging.info("="*70)
        logging.info(f"\nOutputs saved to: {folders['output_snowheight']}")
        logging.info("="*70)
    
    except Exception as e:
        logging.error(f"\nWorkflow Failed: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)
    
    finally:
        # Check in Spatial Analyst extension
        try:
            arcpy.CheckInExtension("Spatial")
        except:
            pass


if __name__ == "__main__":
    main()
