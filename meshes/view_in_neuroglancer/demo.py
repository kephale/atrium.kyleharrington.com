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
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser
import time


def parse_args():
    parser = argparse.ArgumentParser(description='View multiscale mesh in Neuroglancer')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to the zarr file containing the data')
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
    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file {args.zarr_path} does not exist")
        sys.exit(1)
    
    # Get the directory containing the precomputed data
    mesh_dir = os.path.dirname(os.path.abspath(args.zarr_path))
    print(f"Serving precomputed data from: {mesh_dir}")
    
    # Start a local HTTP server to serve the precomputed data
    http_server, http_port = serve_directory(mesh_dir)
    precomputed_url = f"http://localhost:{http_port}"
    print(f"HTTP server started at {precomputed_url}")
    
    # Set up Neuroglancer server
    neuroglancer.set_server_bind_address('127.0.0.1')
    viewer = neuroglancer.Viewer()
    
    # Load zarr data to extract metadata if needed
    zarr_data = zarr.open(args.zarr_path, mode='r')
    
    # Add the mesh layer using precomputed format with proper URL
    with viewer.txn() as s:
        # Add the mesh as a SegmentationLayer with proper precomputed URL
        s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
            source=f"precomputed://{precomputed_url}"
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
