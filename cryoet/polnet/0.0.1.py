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
#     "polnet @ git+https://github.com/jtschwar/polnet",
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
    proteins_list: str = typer.Argument(..., help="Comma-separated list of protein file paths"),
    num_tomos_per_snr: int = typer.Option(1, help="Number of tomograms to produce per SNR"),
    snr: str = typer.Option("0.5", help="Comma-separated list of SNRs to simulate"),
    tilt_range: str = typer.Option("-60,60,3", help="Min,Max,Delta for tilt range"),
    tomo_shape: str = typer.Option("630,630,200", help="Dimensions of tomogram in pixels (X,Y,Z)"),
    misalignment: str = typer.Option("1,5,0.5", help="Min,Max,Sigma for tilt misalignments"),
    voxel_size: float = typer.Option(10.0, help="Voxel size for simulated tomograms"),
    mb_proteins_list: Optional[str] = typer.Option(None, help="Comma-separated list of membrane protein file paths"),
    membranes_list: Optional[str] = typer.Option(None, help="Comma-separated list of membrane file paths"),
):
    """
    Generate synthetic cryo-ET data using Polnet.
    
    This tool creates synthetic cryo-electron tomography data including protein placements
    and membrane simulations. It uses the Polnet framework to generate realistic synthetic
    data for training and testing purposes.
    """
    # Convert string inputs to lists
    PROTEINS_LIST = split_args(proteins_list)
    MB_PROTEINS_LIST = split_args(mb_proteins_list) if mb_proteins_list else []
    MEMBRANES_LIST = split_args(membranes_list) if membranes_list else []
    
    # Validate config path
    if not os.path.exists(config_path):
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(1)
        
    # Setup constants
    SESSION_ID = '0'
    USER_ID = 'polnet'
    SEGMENTATION_NAME = 'membrane'
    TOMO_TYPE = 'wbp'
    
    # Initialize Copick
    try:
        root = copick.from_file(config_path)
        objects = root.pickable_objects
        pickable_objects = [o.name for o in objects if o.is_particle]
    except Exception as e:
        typer.echo(f"Error initializing Copick: {e}", err=True)
        raise typer.Exit(1)
    
    # Setup tomography parameters
    NTOMOS = 1
    SURF_DEC = 0.9
    MMER_TRIES = 20
    PMER_TRIES = 100
    VOI_SHAPE = split_int_args(tomo_shape)
    x, y, z = VOI_SHAPE
    VOI_OFFS = ((int(x * 0.025), int(x * 0.975)), 
                (int(y * 0.025), int(y * 0.975)), 
                (int(z * 0.025), int(z * 0.975)))
    
    # Parse tilt and SNR parameters
    DETECTOR_SNR = split_float_args(snr)
    minAng, maxAng, angIncr = split_float_args(tilt_range)
    TILT_ANGS = [angle for angle in 
                (minAng + i * angIncr for i in range(int((maxAng - minAng) / angIncr) + 1))
                if angle <= maxAng]
    
    MALIGNS = split_float_args(misalignment)
    MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA = MALIGNS
    
    # Print configuration
    typer.echo("Configuration Summary:")
    typer.echo(f"Config Path: {config_path}")
    typer.echo(f"Proteins: {', '.join(PROTEINS_LIST)}")
    typer.echo(f"Membrane Proteins: {', '.join(MB_PROTEINS_LIST)}")
    typer.echo(f"Membranes: {', '.join(MEMBRANES_LIST)}")
    typer.echo(f"SNR values: {DETECTOR_SNR}")
    typer.echo(f"Tilt range: {minAng} to {maxAng} with {angIncr}° increments")
    
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
                
                # Generate features
                try:
                    all_features2(NTOMOS, VOI_SHAPE, permanent_dir, VOI_OFFS, 
                                voxel_size, MMER_TRIES, PMER_TRIES,
                                MEMBRANES_LIST, [], PROTEINS_LIST, 
                                MB_PROTEINS_LIST, SURF_DEC,
                                TILT_ANGS, SNR, MALIGN_MIN, MALIGN_MAX, MALIGN_SIGMA)
                except Exception as e:
                    typer.echo(f"Error generating features for {RUN_NAME}: {e}", err=True)
                    continue
                
                # Process and save results
                try:
                    points_csv_path = os.path.join(permanent_dir, 'tomos_motif_list.csv')
                    csvFile = pd.read_csv(points_csv_path, delimiter='\t')
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