# /// script
# title = "Napari Precomputed Mesh LOD Viewer"
# description = "A Python script to view precomputed mesh data in napari with specific LOD levels"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["mesh", "3D", "visualization", "napari", "neuroglancer", "LOD"]
# documentation = "https://napari.org/stable/api/napari.html"
# requires-python = ">=3.11"
# dependencies = [
#     "napari",
#     "numpy",
#     "PyQt5", 
#     "draco",
#     "trimesh"
# ]
# ///

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="View precomputed mesh data with specific LOD levels")
    parser.add_argument("--mesh-dir", type=str, required=True,
                       help="Directory containing precomputed mesh data")
    parser.add_argument("--proxy-mode", action="store_true",
                       help="Use simplified proxy meshes instead of loading full meshes")
    parser.add_argument("--num-meshes", type=int, default=5,
                       help="Number of meshes to load (default: 5, use 0 for all)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # First check if the meshes directory exists
    mesh_dir = Path(args.mesh_dir)
    if not mesh_dir.exists() or not mesh_dir.is_dir():
        print(f"Error: Mesh directory {mesh_dir} does not exist or is not a directory.")
        print("\nTip: If using 'minimal-ng-precomputed', ensure you've generated the meshes first:")
        print("     uv run meshes/minimal-ng-precomputed/0.0.1.py")
        sys.exit(1)
    
    # Verify if the mesh directory has files
    files = list(mesh_dir.glob("*"))
    if not files:
        print(f"Error: Mesh directory {mesh_dir} is empty.")
        print("\nTip: If using 'minimal-ng-precomputed', ensure you've generated the meshes first:")
        print("     uv run meshes/minimal-ng-precomputed/0.0.1.py")
        sys.exit(1)
    
    # Get script directory for view_in_napari.py
    script_dir = Path(__file__).parent
    view_script = script_dir / "view_in_napari.py"
    
    if not view_script.exists():
        print(f"Error: Cannot find the viewer script at {view_script}")
        sys.exit(1)
    
    # Show menu for LOD selection
    print("====== Neuroglancer Mesh LOD Viewer ======")
    print(f"Mesh directory: {mesh_dir}")
    print("\nPlease select an LOD level to view:")
    print(" 0: Highest detail (LOD 0)")
    print(" 1: Medium detail (LOD 1)")
    print(" 2: Lowest detail (LOD 2)")
    print(" a: View all mesh LODs together (one per mesh)")
    print(" s1: Show all LOD levels simultaneously (aligned)")
    print(" s2: Show all LOD levels simultaneously (separated)")
    
    choice = input("\nEnter your choice (0/1/2/a/s1/s2): ").strip().lower()
    
    # Prepare command
    if choice in ["s1", "s2"]:
        # Use the special view_all_lods script
        view_script = script_dir / "view_all_lods.py"
        if not view_script.exists():
            print(f"Error: Cannot find the all-LODs viewer script at {view_script}")
            sys.exit(1)
            
        cmd = [sys.executable, str(view_script), "--mesh-dir", str(mesh_dir)]
        
        # Set view mode based on choice
        if choice == "s1":
            cmd.extend(["--view-mode", "aligned"])
            print("\nLaunching napari viewer with all LOD levels aligned...")
            print("(LODs will be shown color-coded: LOD 0 = Red, LOD 1 = Green, LOD 2 = Blue)")
        else:  # s2
            cmd.extend(["--view-mode", "separated"])
            print("\nLaunching napari viewer with all LOD levels spatially separated...")
            print("(LODs will be shown color-coded: LOD 0 = Red, LOD 1 = Green, LOD 2 = Blue)")
    else:
        # Use the regular view_in_napari script
        cmd = [sys.executable, str(view_script), "--mesh-dir", str(mesh_dir)]

        if args.debug:
            cmd.append("--debug")
        
        if args.proxy_mode:
            cmd.append("--proxy-mode")
        
        if args.num_meshes > 0:
            cmd.extend(["--num-meshes", str(args.num_meshes)])
        
        # Add LOD parameter based on user choice
        if choice in ["0", "1", "2"]:
            cmd.extend(["--lod", choice])
            print(f"\nLaunching napari viewer with LOD level {choice}...")
        elif choice == "a":
            print("\nLaunching napari viewer with all LOD levels...")
        else:
            print(f"\nInvalid choice '{choice}'. Defaulting to LOD 0...")
            cmd.extend(["--lod", "0"])
    
    # Execute view_in_napari.py with appropriate arguments
    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error executing viewer: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
