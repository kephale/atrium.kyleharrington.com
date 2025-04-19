#!/usr/bin/env python
# /// script
# title = "Neuroglancer Precomputed Mesh Viewer"
# description = "A Python script to view precomputed multiscale mesh data in Neuroglancer"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.2"
# keywords = ["mesh", "3D", "visualization", "neuroglancer", "precomputed"]
# documentation = "https://github.com/google/neuroglancer"
# requires-python = ">=3.8"
# dependencies = [
#     "neuroglancer",
#     "numpy",
#     "json"
# ]
# ///

import argparse
import neuroglancer
import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler with CORS headers"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        SimpleHTTPRequestHandler.end_headers(self)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def parse_args():
    parser = argparse.ArgumentParser(description='View multiscale mesh in Neuroglancer')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to the zarr directory containing the data')
    return parser.parse_args()


def serve_directory(directory, port=8000):
    """Start an HTTP server with CORS support to serve the directory"""
    os.chdir(directory)
    httpd = HTTPServer(('', port), CORSHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Serving directory: {directory}")
    print(f"Access server directly at: http://localhost:{port}")
    print(f"You can manually check mesh files at: http://localhost:{port}/meshes/")
    return httpd, port


def check_mesh_format(mesh_dir):
    """Check if the meshes directory follows the expected Neuroglancer format"""
    # Check if any files have corresponding .index files
    has_index_files = False
    has_mesh_files = False
    segment_ids = []
    
    # Check for non-info files (potential segment files)
    for filename in os.listdir(mesh_dir):
        if os.path.isfile(os.path.join(mesh_dir, filename)) and filename != "info":
            try:
                # See if it's a numeric file (segment ID)
                segment_id = int(filename)
                segment_ids.append(segment_id)
                has_mesh_files = True
                
                # Check if it has a corresponding .index file
                if os.path.exists(os.path.join(mesh_dir, f"{segment_id}.index")):
                    has_index_files = True
            except ValueError:
                # Not a numeric file, might be something else
                pass
    
    if not has_mesh_files:
        return "empty", []
    elif has_index_files:
        return "neuroglancer_full", segment_ids
    else:
        return "neuroglancer_partial", segment_ids


def ensure_info_file(mesh_dir, create_root_info=True):
    """Ensure that an info file exists in the mesh directory"""
    info_path = os.path.join(mesh_dir, "info")
    if not os.path.exists(info_path):
        # Create a basic info file for meshes
        info = {
            "type": "segmentation",
            "mesh": "mesh"
        }
        with open(info_path, "w") as f:
            json.dump(info, f)
        print(f"Created default info file at {info_path}")
    else:
        print(f"Info file already exists at {info_path}")
    
    # Create a copy of the info file at the root level for Neuroglancer to find
    if create_root_info:
        root_info_path = os.path.join(os.path.dirname(mesh_dir), "info")
        if not os.path.exists(root_info_path):
            # Copy the mesh info file to the root
            with open(info_path, "r") as src:
                info_content = json.load(src)
            
            with open(root_info_path, "w") as dest:
                json.dump(info_content, dest)
            print(f"Created root info file at {root_info_path}")
        else:
            print(f"Root info file already exists at {root_info_path}")


def main():
    args = parse_args()
    
    # Ensure the zarr path exists
    zarr_path = os.path.abspath(args.zarr_path)
    if not os.path.exists(zarr_path) or not os.path.isdir(zarr_path):
        print(f"Error: Zarr directory {zarr_path} does not exist or is not a directory")
        sys.exit(1)
    
    # The meshes directory is inside the zarr directory
    mesh_dir = os.path.join(zarr_path, "meshes")
    if not os.path.exists(mesh_dir):
        print(f"Error: Mesh directory {mesh_dir} not found. The meshes directory should be inside the zarr directory.")
        sys.exit(1)
        
    # Check if the mesh directory contains any files other than info
    mesh_files = [f for f in os.listdir(mesh_dir) if os.path.isfile(os.path.join(mesh_dir, f)) and f != "info"]
    if not mesh_files:
        print(f"Warning: Mesh directory {mesh_dir} exists but doesn't contain any mesh files.")
        print("If you just generated the meshes, ensure they were properly exported.")
        # Continue anyway, as there might be meshes that aren't detected this way
    
    print(f"Zarr path: {zarr_path}")
    print(f"Mesh directory found: {mesh_dir}")
    
    # Ensure there's an info file in the meshes directory and at the root
    ensure_info_file(mesh_dir, create_root_info=True)
    
    # Start a local HTTP server to serve the zarr directory
    http_server, http_port = serve_directory(zarr_path)
    precomputed_url = f"http://localhost:{http_port}"
    print(f"HTTP server started at {precomputed_url}")
    
    # Set up Neuroglancer server
    neuroglancer.set_server_bind_address('127.0.0.1')
    viewer = neuroglancer.Viewer()
    
    # Add the mesh layer using precomputed format with proper URL
    with viewer.txn() as s:
        # Check the mesh format type
        format_type, segment_ids = check_mesh_format(mesh_dir)
        
        # Configure the mesh layer based on format
        if format_type == "neuroglancer_full":
            # Standard neuroglancer format with .index files
            s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
                source=f"precomputed://{precomputed_url}"
            )
            print(f"Using standard Neuroglancer format with {len(segment_ids)} segment IDs")
            
        elif format_type == "neuroglancer_partial":
            # Try to detect the mesh format type from info file
            info_path = os.path.join(mesh_dir, "info")
            is_draco_format = False
            
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r") as f:
                        info_content = json.load(f)
                    
                    mesh_type = info_content.get("@type", "")
                    if mesh_type == "neuroglancer_multilod_draco":
                        is_draco_format = True
                        print("Detected neuroglancer_multilod_draco format")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading mesh info file: {e}")
            
            # Set the source URL based on the format
            if is_draco_format:
                mesh_url = f"precomputed://{precomputed_url}/meshes"
                print(f"Using meshes-specific URL: {mesh_url}")
                s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
                    source=mesh_url
                )
            else:
                # Try the root URL
                mesh_url = f"precomputed://{precomputed_url}"
                print(f"Using root URL with fallback: {mesh_url}")
                s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
                    source=mesh_url
                )
            
            print(f"Found {len(segment_ids)} segment IDs without .index files")
            
        else:  # empty format
            # Default to root URL
            s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
                source=f"precomputed://{precomputed_url}"
            )
            print("Warning: No segment IDs found in the meshes directory")
        
        # Set default view options
        s.layers['multiscale_mesh'].visible = True
        
        # Set the segments to show
        if segment_ids:
            s.layers['multiscale_mesh'].segments = set(segment_ids)
            print(f"Configured to show {len(segment_ids)} segments")
        
        # Set dimensions which is supported by Neuroglancer
        s.dimensions = neuroglancer.CoordinateSpace(
            names=['x', 'y', 'z'],
            units=['nm', 'nm', 'nm'],
            scales=[1, 1, 1]
        )
    
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
