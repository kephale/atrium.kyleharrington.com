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
#     "numpy"
# ]
# ///

import argparse
import neuroglancer
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser


def parse_args():
    parser = argparse.ArgumentParser(description='View multiscale mesh in Neuroglancer')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to the zarr directory containing the data')
    return parser.parse_args()


def serve_directory(directory, port=8000):
    """Start a simple HTTP server to serve the directory containing precomputed data"""
    os.chdir(directory)
    httpd = HTTPServer(('', port), SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    return httpd, port


def main():
    args = parse_args()
    
    # Ensure the zarr path exists
    zarr_path = os.path.abspath(args.zarr_path)
    if not os.path.exists(zarr_path) or not os.path.isdir(zarr_path):
        print(f"Error: Zarr directory {zarr_path} does not exist or is not a directory")
        sys.exit(1)
    
    # Get the parent directory containing both the zarr data and meshes
    data_dir = os.path.dirname(zarr_path)
    print(f"Data directory: {data_dir}")
    
    # Check if meshes directory exists
    mesh_dir = os.path.join(data_dir, "meshes")
    if not os.path.exists(mesh_dir):
        print(f"Warning: Mesh directory {mesh_dir} not found. Using data directory.")
        mesh_dir = data_dir
    else:
        print(f"Mesh directory found: {mesh_dir}")
    
    # Start a local HTTP server to serve the data directory
    http_server, http_port = serve_directory(data_dir)
    precomputed_url = f"http://localhost:{http_port}"
    print(f"HTTP server started at {precomputed_url}")
    
    # Set up Neuroglancer server
    neuroglancer.set_server_bind_address('127.0.0.1')
    viewer = neuroglancer.Viewer()
    
    # Add the mesh layer using precomputed format with proper URL
    with viewer.txn() as s:
        # Add the mesh as a SegmentationLayer with proper precomputed URL pointing to meshes
        s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
            source=f"precomputed://{precomputed_url}/meshes"
        )
        
        # Set default view options
        s.layers['multiscale_mesh'].visible = True
        
        # Set initial camera position (adjust based on your data)
        s.navigation.position.voxel_coordinates = [50, 50, 50]
        s.navigation.zoom_factor = 100
    
    # Print the Neuroglancer URL
    neuroglancer_url = viewer.get_viewer_url()
    print(f"Neuroglancer URL: {neuroglancer_url}")
    
    # Open the URL in a web browser
    print("Opening Neuroglancer in browser...")
    webbrowser.open(neuroglancer_url)
    
    # Keep the script running until user input
    print("Press Enter to exit...")
    input()
    
    # Shutdown HTTP server
    http_server.shutdown()


if __name__ == '__main__':
    main()
