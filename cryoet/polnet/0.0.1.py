# /// script
# title = "Generate Synthetic CryoET Data with Polnet"
# description = "Generates synthetic cryo-electron tomography data using Polnet, including protein placements and membrane simulations"
# author = "Kyle Harrington and Jonathan Schwartz"
# license = "MIT"
# version = "0.1.0"
# keywords = ["synthetic data", "deep learning", "cryoet", "tomogram", "protein structure"]
# repository = "https://github.com/anmartinezs/polnet"
# documentation = "https://github.com/anmartinezs/polnet#readme"
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.9",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
# ]
# requires-python = ">=3.9"
# dependencies = [
#     "mrcfile",
#     "numpy",
#     "pandas",
#     "copick",
#     "typer",
#     "polnet @ git+https://github.com/kephale/polnet@scaling",
#     "copick-utils @ git+https://github.com/copick/copick-utils"
# ]
# ///

import os
import typer
import mrcfile
import copick
import numpy as np
import pandas as pd
from typing import Optional, List
from scipy.spatial.transform import Rotation as R
from gui.core.all_features2 import all_features2 
from copick_utils.writers import write

app = typer.Typer(help="Generate synthetic cryo-electron tomography data using Polnet")

def add_points(copick_run, csvFile, in_user_id, in_session_id):
    """Add protein coordinates and orientations to the Copick run."""
    unique_labels = np.unique(csvFile['Label'])
    for ii in range(1, len(unique_labels)+1):
        try:
            proteinName = csvFile[csvFile['Label'] == ii]['Code'].iloc[0].split('_')[0]
            points = csvFile[csvFile['Label'] == ii][['X', 'Y', 'Z']]
            qOrientations = csvFile[csvFile['Label'] == ii][['Q2', 'Q3', 'Q4', 'Q1']]

            orientations = np.zeros([points.shape[0], 4, 4])
            for jj in range(points.shape[0]):
                v = qOrientations.iloc[jj].to_numpy()
                r = R.from_quat(v)         
                orientations[jj,:3,:3] = r.as_matrix()  
            orientations[:,3,3] = 1

            picks = copick_run.new_picks(object_name=proteinName, 
                                        user_id=in_user_id, 
                                        session_id=in_session_id) 
            picks.from_numpy(points.to_numpy(), orientations)               
        except Exception as e: 
            pass

def parse_pns_config(file_path):
    """Parse PNS/PMS configuration file"""
    config = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            # Split on first '=' only
            parts = line.strip().split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                config[key] = value
    return config

def get_pns_info(pns_path):
    """Extract necessary information from PNS/PMS file"""
    config = parse_pns_config(pns_path)
    return {
        'mmer_id': config.get('MMER_ID', ''),
        'mmer_svol': config.get('MMER_SVOL', ''),
        'mmer_iso': float(config.get('MMER_ISO', '0')),
        'pmer_l': float(config.get('PMER_L', '1.1')),
        'pmer_l_max': float(config.get('PMER_L_MAX', '3000')),
        'pmer_occ': float(config.get('PMER_OCC', '0.2')),
        'pmer_over_tol': float(config.get('PMER_OVER_TOL', '0.05')),
        'pmer_reverse_normals': config.get('PMER_REVERSE_NORMALS', 'False').lower() == 'true',
        'is_membrane': 'PMER_REVERSE_NORMALS' in config
    }

# Modify the add_points function to use the configuration information
def add_points(copick_run, csvFile, in_user_id, in_session_id):
    """Add protein coordinates and orientations to the Copick run."""
    unique_labels = np.unique(csvFile['Label'])
    for ii in range(1, len(unique_labels)+1):
        try:
            protein_info = csvFile[csvFile['Label'] == ii]
            if len(protein_info) == 0:
                continue
                
            proteinName = protein_info['Code'].iloc[0].split('_')[0]
            points = protein_info[['X', 'Y', 'Z']]
            qOrientations = protein_info[['Q2', 'Q3', 'Q4', 'Q1']]

            orientations = np.zeros([points.shape[0], 4, 4])
            for jj in range(points.shape[0]):
                v = qOrientations.iloc[jj].to_numpy()
                r = R.from_quat(v)         
                orientations[jj,:3,:3] = r.as_matrix()  
            orientations[:,3,3] = 1

            picks = copick_run.new_picks(object_name=proteinName, 
                                       user_id=in_user_id, 
                                       session_id=in_session_id) 
            picks.from_numpy(points.to_numpy(), orientations)               
        except Exception as e: 
            print(f"Error processing label {ii}: {e}")
            continue

def extract_membrane_segmentation(segVol, csvFile, pickable_objects):
    """Extract membrane segmentation from the volume."""
    membranes = np.zeros(segVol.shape, dtype=np.uint8)
    unique_labels = np.unique(csvFile['Label'])
    for ii in range(1, len(unique_labels)+1):
        proteinName = csvFile[csvFile['Label'] == ii]['Code'].iloc[0].split('_')[0]
        if proteinName not in pickable_objects:
            membranes[segVol == ii] = 1
    return membranes

def split_args(arg: str) -> List[str]:
    """Split comma-separated string arguments."""
    return arg.split(',') if arg else []

def split_float_args(arg: str) -> List[float]:
    """Split and convert to float comma-separated string arguments."""
    return [float(x) for x in arg.split(',')] if arg else []

def split_int_args(arg: str) -> List[int]:
    """Split and convert to int comma-separated string arguments."""
    return [int(x) for x in arg.split(',')] if arg else []

@app.command()
def generate_synthetic_data(
    config_path: str = typer.Argument(..., help="Path to the Copick configuration file"),
    proteins_list: str = typer.Option(None, help="Comma-separated list of protein file paths"),
    num_tomos_per_snr: int = typer.Option(1, help="Number of tomograms to produce per SNR"),
    snr: str = typer.Option("0.5", help="Comma-separated list of SNRs to simulate"),
    tilt_range: str = typer.Option("-60,60,3", help="Min,Max,Delta for tilt range"),
    tomo_shape: str = typer.Option("630,630,200", help="Dimensions of tomogram in pixels (X,Y,Z)"),
    misalignment: str = typer.Option("1,5,0.5", help="Min,Max,Sigma for tilt misalignments"),
    voxel_size: float = typer.Option(10.0, help="Voxel size for simulated tomograms"),
    mb_proteins_list: Optional[str] = typer.Option(None, help="Comma-separated list of membrane protein file paths"),
    membranes_list: Optional[str] = typer.Option(None, help="Comma-separated list of membrane file paths"),
):
    """Generate synthetic cryo-ET data using Polnet."""
    
    # Convert string inputs to lists
    PROTEINS_LIST = split_args(proteins_list)
    MB_PROTEINS_LIST = split_args(mb_proteins_list) if mb_proteins_list else []
    MEMBRANES_LIST = split_args(membranes_list) if membranes_list else []
    
    # Parse protein configuration files
    protein_configs = []
    for pns_file in PROTEINS_LIST:
        try:
            config = get_pns_info(pns_file)
            protein_configs.append((pns_file, config))
        except Exception as e:
            typer.echo(f"Error parsing protein config {pns_file}: {e}", err=True)
            continue
    
    mb_protein_configs = []
    for pms_file in MB_PROTEINS_LIST:
        try:
            config = get_pns_info(pms_file)
            mb_protein_configs.append((pms_file, config))
        except Exception as e:
            typer.echo(f"Error parsing membrane protein config {pms_file}: {e}", err=True)
            continue
    
    # Validate config path
    if not os.path.exists(config_path):
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
        
    # Setup constants with debug logging
    SESSION_ID = '0'
    USER_ID = 'polnet'
    SEGMENTATION_NAME = 'membrane'
    TOMO_TYPE = 'wbp'
    
    # Initialize Copick with debug logging
    try:
        root = copick.from_file(config_path)
        objects = root.pickable_objects
        pickable_objects = [o.name for o in objects if o.is_particle]
        typer.echo(f"Debug: Pickable objects: {pickable_objects}")
    except Exception as e:
        typer.echo(f"Error initializing Copick: {e}", err=True)
        raise typer.Exit(1)
    
    # Setup tomography parameters with debug logging
    NTOMOS = 1
    SURF_DEC = 0.9
    MMER_TRIES = 20
    PMER_TRIES = 100
    VOI_SHAPE = split_int_args(tomo_shape)
    x, y, z = VOI_SHAPE
    VOI_OFFS = ((int(x * 0.025), int(x * 0.975)), 
                (int(y * 0.025), int(y * 0.975)), 
                (int(z * 0.025), int(z * 0.975)))
    
    typer.echo(f"Debug: VOI_SHAPE: {VOI_SHAPE}")
    typer.echo(f"Debug: VOI_OFFS: {VOI_OFFS}")
    
    # Parse tilt and SNR parameters with debug logging
    DETECTOR_SNR = split_float_args(snr)
    minAng, maxAng, angIncr = split_float_args(tilt_range)
    TILT_ANGS = [angle for angle in 
                (minAng + i * angIncr for i in range(int((maxAng - minAng) / angIncr) + 1))
                if angle <= maxAng]
    
    MALIGNS = split_float_args(misalignment)
    MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA = MALIGNS
    
    typer.echo(f"Debug: TILT_ANGS: {TILT_ANGS}")
    typer.echo(f"Debug: MALIGNS: {MALIGNS}")
    
    # Generate tomograms for each SNR
    run_ids = [run.name for run in root.runs] 
    currTSind = int(max(run_ids, key=lambda x: int(x.split('_')[1])).split('_')[1]) + 1 if run_ids else 1
    
    with typer.progressbar(DETECTOR_SNR, label="Processing SNR values") as snr_progress:
        for SNR in snr_progress:
            for ii in range(num_tomos_per_snr):
                RUN_NAME = f'TS_{currTSind}'
                permanent_dir = f"./tmp_polnet_output/{RUN_NAME}"
                
                copick_run = root.get_run(RUN_NAME) or root.new_run(RUN_NAME)
                
                TEM_DIR = os.path.join(permanent_dir, 'tem')
                TOMOS_DIR = os.path.join(permanent_dir, 'tomos')
                
                # Generate features with parsed configurations
                try:
                    # Extract just the file paths for the all_features2 call
                    protein_paths = [p[0] for p in protein_configs]
                    mb_protein_paths = [p[0] for p in mb_protein_configs]
                    
                    all_features2(NTOMOS, VOI_SHAPE, permanent_dir, VOI_OFFS, 
                                voxel_size, MMER_TRIES, PMER_TRIES,
                                MEMBRANES_LIST, [], protein_paths, 
                                mb_protein_paths, SURF_DEC,
                                TILT_ANGS, SNR, MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA)
                                
                except Exception as e:
                    typer.echo(f"Error generating features for {RUN_NAME}: {e}", err=True)
                    continue
                
                # Process and save results
                try:
                    points_csv_path = os.path.join(permanent_dir, 'tomos_motif_list.csv')
                    csvFile = pd.read_csv(points_csv_path, delimiter='\t')
                    
                    # Use the parsed configurations when adding points
                    add_points(copick_run, csvFile, USER_ID, SESSION_ID)
                    
                    # Save tomogram
                    vol = mrcfile.read(os.path.join(TEM_DIR, 'out_rec3d.mrc'))
                    write.tomogram(copick_run, vol, voxel_size=voxel_size)
                    
                    # Save segmentation
                    ground_truth = mrcfile.read(os.path.join(TOMOS_DIR, 'tomo_lbls_0.mrc'))
                    membranes = extract_membrane_segmentation(ground_truth, csvFile, pickable_objects)
                    write.segmentation(copick_run, membranes, USER_ID, 
                                     name='membrane', voxel_size=voxel_size, 
                                     multilabel=False)
                    
                    typer.echo(f"Successfully processed {RUN_NAME}")
                except Exception as e:
                    typer.echo(f"Error processing {RUN_NAME}: {e}", err=True)
                
                currTSind += 1

    typer.echo("Processing complete!")

if __name__ == "__main__":
    app()