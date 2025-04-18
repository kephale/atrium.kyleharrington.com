#!/usr/bin/env python
# Neuroglancer viewer for multiscale meshes

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
    # This assumes that the meshes are stored in the same directory as the zarr file
    # in a 'precomputed' directory structure
    mesh_dir = os.path.dirname(os.path.abspath(args.zarr_path))
    
    # Load zarr data to extract metadata if needed
    zarr_data = zarr.open(args.zarr_path, mode='r')
    
    # Add the mesh layer using precomputed format
    with viewer.txn() as s:
        s.layers['multiscale_mesh'] = neuroglancer.LocalVolume(
            volume_type='precomputed',
            mesh=mesh_dir,
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
    
    # Keep the script running
    print("Press Ctrl+C to exit")
    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
