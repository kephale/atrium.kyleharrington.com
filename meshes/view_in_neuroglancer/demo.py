#!/usr/bin/env python
# /// script
# title = "Neuroglancer Precomputed Mesh Viewer"
# description = "A Python script to view precomputed multiscale mesh data in Neuroglancer"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["mesh", "3D", "visualization", "neuroglancer", "precomputed"]
# documentation = "https://github.com/google/neuroglancer"
# requires-python = ">=3.8"
# dependencies = [
#     "neuroglancer",
#     "numpy",
#     "zarr>=3.0.0"
# ]
# ///

import argparse
import neuroglancer
import numpy as np
import os
import sys
import zarr


def parse_args():
    parser = argparse.ArgumentParser(description='View multiscale mesh in Neuroglancer')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to the zarr file containing the data')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Ensure the zarr path exists
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file {args.zarr_path} does not exist")
        sys.exit(1)
    
    # Set up server for Neuroglancer
    neuroglancer.set_server_bind_address('127.0.0.1')
    viewer = neuroglancer.Viewer()
    
    # Path where the precomputed mesh data is stored
    # This assumes that meshes are stored in the same directory as the zarr file
    mesh_dir = os.path.dirname(os.path.abspath(args.zarr_path))
    mesh_url = f"precomputed://{mesh_dir}"
    
    # Load zarr data to extract metadata if needed
    zarr_data = zarr.open(args.zarr_path, mode='r')
    
    # Add the mesh layer using precomputed format
    with viewer.txn() as s:
        # Add the mesh as a SegmentationLayer which can display precomputed meshes
        s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
            source=mesh_url
        )
        
        # Set default view options
        s.layers['multiscale_mesh'].visible = True
        s.layers['multiscale_mesh'].shader = """
            void main() {
                emitRGB(vec3(0.8, 0.2, 0.3));
            }
        """
        
        # Set initial camera position (adjust based on your data)
        s.navigation.position.voxel_coordinates = [50, 50, 50]
        s.navigation.zoom_factor = 100
    
    print(f"Neuroglancer URL: {viewer.get_viewer_url()}")
    
    # Keep the script running until user input
    print("Press Enter to exit...")
    input()


if __name__ == '__main__':
    main()
