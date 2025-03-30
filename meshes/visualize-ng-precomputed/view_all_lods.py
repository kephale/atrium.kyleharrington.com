# /// script
# title = "Napari Precomputed Mesh Viewer with All LODs"
# description = "A Python script to visualize all LODs of precomputed mesh data simultaneously in napari"
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
import json
import sys
import os
import tempfile
import subprocess
import numpy as np
import trimesh
import napari
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import the PrecomputedMeshLoader class
# We're placing the script in the same directory so this relative import should work
from view_in_napari import PrecomputedMeshLoader

def main():
    parser = argparse.ArgumentParser(description="View all LOD levels of precomputed mesh data simultaneously")
    parser.add_argument("--mesh-dir", type=str, required=True,
                      help="Directory containing precomputed mesh data")
    parser.add_argument("--num-meshes", type=int, default=3,
                      help="Number of meshes to load (default: 3)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    parser.add_argument("--color-by-lod", action="store_true", default=True,
                      help="Color meshes by LOD level (default: True)")
    parser.add_argument("--opacity", type=float, default=0.5,
                      help="Opacity for meshes (default: 0.5)")
    parser.add_argument("--view-mode", type=str, choices=["aligned", "separated"], default="aligned",
                      help="View mode: 'aligned' for perfectly aligned LODs, 'separated' for spatially separated LODs (default: aligned)")
    parser.add_argument("--position-offset", type=float, default=10.0,
                      help="Position offset between LOD levels in 'separated' mode (default: 10.0)")
    
    args = parser.parse_args()
    
    # Initialize the mesh loader
    mesh_loader = PrecomputedMeshLoader(args.mesh_dir, debug=args.debug)
    
    # Validate the directory and get valid mesh IDs
    valid, message, valid_meshes = mesh_loader.validate_directory()
    
    if not valid:
        print(f"Error: {message}")
        print("\nTip: If using 'minimal-ng-precomputed', ensure you've generated the meshes first:")
        print("     uv run meshes/minimal-ng-precomputed/0.0.1.py")
        sys.exit(1)
        
    print(f"\nFound {len(valid_meshes)} valid meshes in {args.mesh_dir}")
    
    # Decide how many meshes to load
    if args.num_meshes > 0 and args.num_meshes < len(valid_meshes):
        load_meshes = valid_meshes[:args.num_meshes]
        print(f"Loading first {args.num_meshes} of {len(valid_meshes)} meshes")
        print(f"Loading mesh IDs: {load_meshes}")
    else:
        load_meshes = valid_meshes
        print(f"Loading all {len(valid_meshes)} meshes")
    
    # Initialize napari viewer with 3D mode
    viewer = napari.Viewer(ndisplay=3)
    
    # Define LOD colors if color-by-lod is enabled
    lod_colors = {
        0: [1.0, 0.1, 0.1],  # Red for LOD 0 (highest detail)
        1: [0.1, 1.0, 0.1],  # Green for LOD 1 (medium detail)
        2: [0.1, 0.1, 1.0],  # Blue for LOD 2 (lowest detail)
    }
    
    # Store created layers for reference
    layers = {}
    
    # Process each mesh
    for mesh_idx, mesh_id in enumerate(load_meshes):
        try:
            # Read the manifest once
            manifest = mesh_loader.read_manifest(mesh_id)
            if not manifest or "num_lods" not in manifest:
                print(f"Could not load manifest for mesh {mesh_id}")
                continue
                
            num_lods = manifest["num_lods"]
            print(f"Mesh {mesh_id} has {num_lods} LOD levels")
            
            # Load each LOD level separately
            for lod in range(num_lods):
                # Skip LOD if it has no fragments
                if lod not in manifest["fragments"] or manifest["fragments_per_lod"][lod] == 0:
                    print(f"No fragments found for mesh {mesh_id} at LOD {lod}")
                    continue
                
                # Load the mesh at this LOD
                mesh = mesh_loader.load_lod_mesh(mesh_id, lod)
                if mesh is None:
                    print(f"Could not load mesh {mesh_id} at LOD {lod}")
                    continue
                
                # Apply offset based on LOD level if in separated mode
                if args.view_mode == "separated" and args.position_offset != 0:
                    offset_vector = np.array([lod * args.position_offset, 
                                            lod * args.position_offset, 
                                            lod * args.position_offset])
                    mesh.vertices = mesh.vertices + offset_vector
                
    # Add a helpful indicator to the layer name to show LOD level
    prefix_by_lod = {
        0: "🔴 LOD0", 
        1: "🟢 LOD1", 
        2: "🔵 LOD2"
    }
    prefix = prefix_by_lod.get(lod, f"LOD{lod}")
    layer_name = f"{prefix}: Mesh {mesh_id}"
                
                # Choose color based on LOD level or mesh ID
                if args.color_by_lod:
                    # Use LOD-specific color
                    color = lod_colors.get(lod, [0.5, 0.5, 0.5])  # Default to gray for unsupported LODs
                else:
                    # Use mesh-specific color with deterministic generation
                    import hashlib
                    color_seed = hashlib.md5(str(mesh_id).encode()).digest()
                    color = [
                        color_seed[0] / 255, 
                        color_seed[1] / 255, 
                        color_seed[2] / 255
                    ]
                
                # Add the surface
                try:
                    # Use a simpler approach without per-vertex colors
                    surface = viewer.add_surface(
                        data=(mesh.vertices, mesh.faces),
                        name=layer_name,
                        opacity=args.opacity,
                        blending='translucent',
                        colormap='gray',
                        contrast_limits=[0, 1]
                    )
                    
                    # Set the color manually for the entire surface
                    surface.color = color
                    
                    # Store the layer
                    layer_key = f"{mesh_id}_{lod}"
                    layers[layer_key] = surface
                    
                    print(f"Added mesh {mesh_id} LOD {lod} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                    
                except Exception as e:
                    print(f"Error adding surface for mesh {mesh_id} LOD {lod}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()
                
        except Exception as e:
            print(f"Error processing mesh {mesh_id}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    if not layers:
        print("No meshes could be loaded. Try using --debug for more information.")
        sys.exit(1)
        
    # Reset view to show all objects
    viewer.reset_view()
    
    print(f"\nNapari viewer launched with all LOD levels in '{args.view_mode}' mode.")
    print("LOD visualization colors:")
    print("- LOD 0 (highest detail): Red")
    print("- LOD 1 (medium detail): Green") 
    print("- LOD 2 (lowest detail): Blue")
    
    if args.view_mode == "aligned":
        print("\nMeshes are perfectly aligned to verify LOD quality.")
        print("You can toggle visibility of different LOD levels using the layer list on the left.")
    elif args.view_mode == "separated" and args.position_offset != 0:
        print(f"\nNOTE: LOD levels are offset by {args.position_offset} units for easier visual inspection.")
    
    # Start the napari event loop
    napari.run()

if __name__ == '__main__':
    main()
