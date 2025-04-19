#!/usr/bin/env python
# /// script
# title = "Neuroglancer Precomputed Mesh Viewer"
# description = "A Python script to view precomputed multiscale mesh data in Neuroglancer"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.1"
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
    """Custom HTTP request handler with CORS headers and special handling for mesh files"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        SimpleHTTPRequestHandler.end_headers(self)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # Check if the path is a direct request for an index file at root level
        # like /123.index where the actual file is in /meshes/123.index
        if self.path.endswith('.index') and '/' not in self.path[1:]:
            # Extract the segment ID
            segment_id = self.path[1:-6]  # Remove leading '/' and trailing '.index'
            # Construct the path to the meshes directory
            mesh_file_path = os.path.join('meshes', f"{segment_id}.index")
            
            # Check if the file exists in the meshes directory
            if os.path.exists(mesh_file_path):
                self.path = f"/meshes/{segment_id}.index"  # Redirect to the meshes directory
                return SimpleHTTPRequestHandler.do_GET(self)
        
        # For regular segment data requests like /123 (without .index)
        if '/' not in self.path[1:] and self.path[1:].isdigit():
            segment_id = self.path[1:]  # Remove leading '/'
            mesh_file_path = os.path.join('meshes', segment_id)
            
            # Check if the file exists in the meshes directory
            if os.path.exists(mesh_file_path):
                self.path = f"/meshes/{segment_id}"  # Redirect to the meshes directory
                return SimpleHTTPRequestHandler.do_GET(self)
        
        # Default handling for all other requests
        return SimpleHTTPRequestHandler.do_GET(self)


def parse_args():
    parser = argparse.ArgumentParser(description='View multiscale mesh in Neuroglancer')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to the zarr directory containing the data')
    return parser.parse_args()


def serve_directory(directory, port=8000):
    """Start an HTTP server with CORS support to serve the directory
    
    This server needs to handle the Neuroglancer mesh format properly,
    ensuring that .index files and mesh segment data are accessible with
    the correct URLs.
    """
    os.chdir(directory)
    httpd = HTTPServer(('', port), CORSHTTPRequestHandler)
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"Serving directory: {directory}")
    print(f"Access server directly at: http://localhost:{port}")
    print(f"You can manually check mesh files at: http://localhost:{port}/meshes/")
    return httpd, port


def ensure_info_file(mesh_dir, create_root_info=True):
    """Ensure that an info file exists in the mesh directory following the Neuroglancer precomputed format.
    
    The mesh info format is described at:
    https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-info-json-file-format
    """
    info_path = os.path.join(mesh_dir, "info")
    if not os.path.exists(info_path):
        # Create a basic info file for meshes that conforms to the Neuroglancer spec
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": 16,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "lod_scale_multiplier": 2.0
        }
        with open(info_path, "w") as f:
            json.dump(info, f)
        print(f"Created default info file at {info_path}")
    else:
        print(f"Info file already exists at {info_path}")
    
    # Create a segmentation info file at the root level for Neuroglancer to find
    if create_root_info:
        root_dir = os.path.dirname(mesh_dir)
        root_info_path = os.path.join(root_dir, "info")
        
        # Create the top-level info file with correct mesh directory reference
        if not os.path.exists(root_info_path):
            root_info = {
                "@type": "neuroglancer_scene",
                "dimensions": {
                    "x": [1, "nm"],
                    "y": [1, "nm"],
                    "z": [1, "nm"]
                },
                "position": [0, 0, 0],
                "crossSectionScale": 1,
                "projectionScale": 4096
            }
            
            with open(root_info_path, "w") as dest:
                json.dump(root_info, dest)
            print(f"Created root info file at {root_info_path}")
        else:
            print(f"Root info file already exists at {root_info_path}")
        
        # Check if a meshes/info file exists for meshes reference
        meshes_info_path = os.path.join(mesh_dir, "info")
        if not os.path.exists(meshes_info_path):
            # Ensure the meshes/info file exists with the correct format
            meshes_info = {
                "@type": "neuroglancer_multilod_draco",
                "vertex_quantization_bits": 16,
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                "lod_scale_multiplier": 2.0
            }
            with open(meshes_info_path, "w") as f:
                json.dump(meshes_info, f)
            print(f"Created meshes info file at {meshes_info_path}")


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
        # Explicitly specify the precomputed source URL to include the 'meshes/' directory
        source_url = f"precomputed://{precomputed_url}/meshes"
        print(f"Using mesh source URL: {source_url}")
        
        # Add the mesh as a SegmentationLayer with proper URL path
        s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
            source=source_url
        )
        
        # Set default view options
        s.layers['multiscale_mesh'].visible = True
        
        # Find all available segment IDs based on .index files
        mesh_dir = os.path.join(zarr_path, "meshes")
        segment_ids = []
        for filename in os.listdir(mesh_dir):
            # Look for files ending with .index where the base name is an integer
            if filename.endswith('.index'):
                try:
                    segment_id = int(filename[:-6])  # Remove '.index' suffix
                    segment_ids.append(segment_id)
                except ValueError:
                    pass
        
        # Set segments to show in the layer
        if segment_ids:
            s.layers['multiscale_mesh'].segments = set(segment_ids)
            print(f"Found segment IDs: {segment_ids}")
        else:
            print("Warning: No segment IDs found in the meshes directory")
        
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
