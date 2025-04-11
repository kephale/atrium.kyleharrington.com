# /// script
# title = "kaggle-csv-to-copick"
# description = "Import particle picks from CSV files into a Copick project"
# author = "Kyle Harrington <copick@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["copick", "cryo-em", "particle-picking", "import"]
# repository = "https://github.com/kyleharrington/copick"
# documentation = "https://github.com/kyleharrington/copick#readme"
# homepage = "https://cellcanvas.org"
# classifiers = [
# "Development Status :: 3 - Alpha",
# "Intended Audience :: Science/Research",
# "License :: OSI Approved :: MIT License",
# "Programming Language :: Python :: 3",
# "Topic :: Scientific/Engineering :: Bio-Informatics",
# ]
# requires-python = ">=3.8"
# dependencies = [
# "copick",
# "pandas",
# "argparse",
# ]
# ///

import pandas as pd
import argparse
import copick
from copick.models import CopickPoint

def load_and_process_data(picks_file: str, mapping_file: str):
    """
    Load and process the picks and mapping CSV files.
    """
    # Read the CSVs
    picks_df = pd.read_csv(picks_file)
    mapping_df = pd.read_csv(mapping_file)
    
    # Create a mapping dictionary from original to mapped names
    if mapping_file:
        name_mapping = dict(zip(mapping_df['orig_id'], mapping_df['experiment']))
        
        # Replace the mapped names with original names
        picks_df['experiment'] = picks_df['experiment'].map(
            {v: k for k, v in name_mapping.items()}
        )
        
    return picks_df

def add_picks_to_copick(picks_df: pd.DataFrame, copick_config_path: str, session_id: str, user_id: str):
    """
    Add picks to an existing Copick project.
    
    Args:
        picks_df (pd.DataFrame): DataFrame containing the picks data
        copick_config_path (str): Path to the Copick config file
        session_id (str): ID for the picking session
        user_id (str): ID of the user performing the import
    """
    # Load the Copick project
    root = copick.from_file(copick_config_path)
    
    # Group picks by experiment and particle type
    grouped = picks_df.groupby(['experiment', 'particle_type'])
    
    for (run_name, particle_type), group in grouped:
        print(f"Processing {run_name} - {particle_type}")
        
        # Get or create the run
        copick_run = root.get_run(run_name)
        if not copick_run:
            print(f"Run {run_name} not found in project, skipping...")
            continue
            
        # Create new pick set with provided session and user IDs
        pick_set = copick_run.new_picks(
            particle_type,
            session_id=session_id,
            user_id=user_id
        )
        
        # Convert coordinates to CopickPoints
        points = []
        for _, row in group.iterrows():
            point = CopickPoint(
                location={
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z'])
                }
            )
            points.append(point)
        
        pick_set.points = points
        pick_set.store()
        
        print(f"Added {len(points)} picks for {particle_type} in {run_name}")

def main():
    parser = argparse.ArgumentParser(description='Import picks to Copick project')
    parser.add_argument('--picks', required=True, help='Path to picks CSV file')
    parser.add_argument('--mapping', required=False, help='Path to name mapping CSV file')
    parser.add_argument('--config', required=True, help='Path to Copick config file')
    parser.add_argument('--session-id', required=True, help='Session ID for the picking session')
    parser.add_argument('--user-id', required=True, help='User ID performing the import')
    
    args = parser.parse_args()
    
    # Load and process the data
    mapping = args.mapping if args.mapping else None
    picks_df = load_and_process_data(args.picks, mapping)
    
    # Add picks to Copick project with provided session and user IDs
    add_picks_to_copick(picks_df, args.config, args.session_id, args.user_id)
    
    print("Done!")

if __name__ == "__main__":
    main()