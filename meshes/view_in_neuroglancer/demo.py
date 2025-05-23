#!/usr/bin/env python
# /// script
# title = "Neuroglancer Precomputed Mesh Viewer"
# description = "A Python script to view precomputed multiscale mesh data in Neuroglancer"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.2.2"
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
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP request handler with CORS headers and improved handling for mesh files"""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Access-Control-Max-Age', '86400')
        
        # Add proper content type for mesh files
        if self.path.endswith('.index'):
            self.send_header('Content-Type', 'application/octet-stream')
        elif self.path.split('/')[-1].isdigit():  # Mesh data files
            self.send_header('Content-Type', 'application/octet-stream')
        
        SimpleHTTPRequestHandler.end_headers(self)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        # Improved path handling for mesh files to prevent 404 errors
        original_path = self.path
        
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
        
        # Handle explicit meshes/ requests properly
        if self.path.startswith('/meshes/'):
            # This is already the correct path, proceed normally
            return SimpleHTTPRequestHandler.do_GET(self)
        
        # Log requests for debugging
        if original_path != self.path:
            print(f"Redirected request: {original_path} -> {self.path}")
        
        # Default handling for all other requests
        return SimpleHTTPRequestHandler.do_GET(self)


def parse_args():
    parser = argparse.ArgumentParser(description='View multiscale mesh in Neuroglancer')
    parser.add_argument('--zarr-path', type=str, required=True,
                        help='Path to the zarr directory containing the data')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    return parser.parse_args()


def serve_directory(directory, port=8000):
    """Start an HTTP server with CORS support to serve the directory."""
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
    """Ensure that an info file exists with the correct specifications."""
    info_path = os.path.join(mesh_dir, "info")
    if not os.path.exists(info_path):
        # Create a properly configured info file for meshes with identity transform
        # to maintain coordinate system alignment with the image data
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": 16,  # Higher precision for better quality
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Identity transform
            "lod_scale_multiplier": 2.0
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Created properly configured info file at {info_path}")
    else:
        # Read existing info to verify it has appropriate settings
        try:
            with open(info_path, 'r') as f:
                existing_info = json.load(f)
                print(f"Existing info file found with type: {existing_info.get('@type', 'unknown')}")
                print(f"Quantization bits: {existing_info.get('vertex_quantization_bits', 'not specified')}")
                print(f"Transform: {existing_info.get('transform', 'not specified')}")
                
                # Check if transform is identity and warn if not
                transform = existing_info.get('transform', [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
                identity_transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
                if transform != identity_transform:
                    print(f"Warning: Non-identity transform detected: {transform}")
                    print("This may cause coordinate misalignment issues")
                else:
                    print("✓ Identity transform confirmed - good for coordinate alignment")
        except Exception as e:
            print(f"Warning: Error reading existing info file: {e}")
    
    # Create a segmentation info file at the root level with proper settings
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
                "projectionScale": 4096,
                # Coordinate system alignment settings
                "layers": [
                    {
                        "type": "segmentation",
                        "source": "precomputed://meshes",
                        "tab": "segments",
                        "name": "meshes",
                        # Add proper coordinate alignment settings - identity transform
                        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
                    }
                ]
            }
            
            with open(root_info_path, "w") as dest:
                json.dump(root_info, dest, indent=2)
            print(f"Created root info file at {root_info_path}")
        else:
            print(f"Root info file already exists at {root_info_path}")


def validate_mesh_files(mesh_dir, debug=False):
    """Validate that mesh files are properly formatted and accessible."""
    print(f"\nValidating mesh files in {mesh_dir}...")
    
    # Check for info file
    info_path = os.path.join(mesh_dir, "info")
    if not os.path.exists(info_path):
        print(f"ERROR: Missing info file at {info_path}")
        return False
    
    # Find all mesh files
    mesh_files = []
    index_files = []
    
    for filename in os.listdir(mesh_dir):
        filepath = os.path.join(mesh_dir, filename)
        if not os.path.isfile(filepath):
            continue
            
        if filename.endswith('.index'):
            try:
                mesh_id = int(filename[:-6])  # Remove .index suffix
                index_files.append(mesh_id)
            except ValueError:
                if debug:
                    print(f"Skipping non-numeric index file: {filename}")
        elif filename.isdigit():
            mesh_files.append(int(filename))
    
    # Check for paired files
    valid_meshes = set(mesh_files) & set(index_files)
    missing_data = set(index_files) - set(mesh_files)
    missing_index = set(mesh_files) - set(index_files)
    
    print(f"Found {len(valid_meshes)} complete mesh pairs")
    print(f"Valid mesh IDs: {sorted(list(valid_meshes))}")
    
    if missing_data:
        print(f"WARNING: Missing data files for indices: {sorted(list(missing_data))}")
    if missing_index:
        print(f"WARNING: Missing index files for data: {sorted(list(missing_index))}")
    
    if not valid_meshes:
        print("ERROR: No valid mesh pairs found!")
        return False
    
    # Validate file sizes
    for mesh_id in sorted(list(valid_meshes)):
        index_path = os.path.join(mesh_dir, f"{mesh_id}.index")
        data_path = os.path.join(mesh_dir, str(mesh_id))
        
        index_size = os.path.getsize(index_path)
        data_size = os.path.getsize(data_path)
        
        if debug:
            print(f"Mesh {mesh_id}: index={index_size} bytes, data={data_size} bytes")
        
        # Basic validation - files should not be empty
        if index_size == 0:
            print(f"ERROR: Empty index file for mesh {mesh_id}")
            return False
        if data_size == 0:
            print(f"ERROR: Empty data file for mesh {mesh_id}")
            return False
    
    print("✓ Mesh file validation completed successfully")
    return True


def read_zarr_metadata(zarr_path):
    """Read metadata from zarr store to understand coordinate system."""
    try:
        import zarr
        
        zarr_store = zarr.open(zarr_path, mode='r')
        
        # Get image shape info for coordinate reference
        image_shape = None
        if 'ome' in zarr_store.attrs:
            if 'multiscales' in zarr_store.attrs['ome']:
                multiscales = zarr_store.attrs['ome']['multiscales']
                if len(multiscales) > 0 and 'datasets' in multiscales[0]:
                    # Get the highest resolution dataset
                    first_dataset = multiscales[0]['datasets'][0]
                    dataset_path = first_dataset['path']
                    
                    # Try to get the array
                    if dataset_path in zarr_store:
                        array = zarr_store[dataset_path]
                        image_shape = array.shape
                        print(f"Found image data with shape: {image_shape}")
                        
                        # If 4D, assume (c, z, y, x) and extract spatial dimensions
                        if len(image_shape) == 4:
                            spatial_shape = image_shape[1:]  # Skip channel dimension
                            print(f"Spatial dimensions (z, y, x): {spatial_shape}")
                            return spatial_shape
                        elif len(image_shape) == 3:
                            print(f"3D dimensions (z, y, x): {image_shape}")
                            return image_shape
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not read zarr metadata: {e}")
        return None


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
        
    # Validate mesh files before starting the server
    if not validate_mesh_files(mesh_dir, debug=args.debug):
        print("Mesh file validation failed. Please check your mesh files.")
        sys.exit(1)
    
    print(f"Zarr path: {zarr_path}")
    print(f"Mesh directory found: {mesh_dir}")
    
    # Read zarr metadata to understand coordinate system
    image_shape = read_zarr_metadata(zarr_path)
    
    # Ensure there's an info file in the meshes directory and at the root
    ensure_info_file(mesh_dir, create_root_info=True)
    
    # Start a local HTTP server to serve the zarr directory
    http_server, http_port = serve_directory(zarr_path)
    precomputed_url = f"http://localhost:{http_port}"
    print(f"HTTP server started at {precomputed_url}")
    
    # Set up Neuroglancer server
    neuroglancer.set_server_bind_address('127.0.0.1')
    viewer = neuroglancer.Viewer()
    
    # Add the mesh layer using precomputed format with proper coordinate alignment
    with viewer.txn() as s:
        # Explicitly specify the precomputed source URL to include the 'meshes/' directory
        source_url = f"precomputed://{precomputed_url}/meshes"
        print(f"Using mesh source URL: {source_url}")
        
        # Add the mesh as a SegmentationLayer with proper URL path and coordinate settings
        s.layers['multiscale_mesh'] = neuroglancer.SegmentationLayer(
            source=source_url
        )
        
        # Set default view options
        s.layers['multiscale_mesh'].visible = True
        
        # Find all available segment IDs based on .index files
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
            print(f"Found segment IDs: {sorted(segment_ids)}")
            
            # Set proper display settings for better visualization
            s.layers['multiscale_mesh'].segment_default_color = '#ffffff'
            
            # Add some example segment colors for better visualization
            segment_colors = {}
            import random
            random.seed(42)  # For reproducible colors
            for seg_id in sorted(segment_ids):
                # Generate a distinct color for each segment
                r = random.randint(100, 255)
                g = random.randint(100, 255) 
                b = random.randint(100, 255)
                segment_colors[seg_id] = f"#{r:02x}{g:02x}{b:02x}"
            
            s.layers['multiscale_mesh'].segment_colors = segment_colors
            print(f"Applied colors to {len(segment_colors)} segments")
        else:
            print("Warning: No segment IDs found in the meshes directory")
        
        # Set dimensions with proper coordinate system alignment
        s.dimensions = neuroglancer.CoordinateSpace(
            names=['x', 'y', 'z'],
            units=['nm', 'nm', 'nm'],
            scales=[1, 1, 1]  # Identity scaling to maintain coordinate alignment
        )
        
        # Set proper viewing parameters for better mesh visualization
        s.projection_orientation = [0, 0, 0, 1]  # Default orientation
        s.projection_scale = 256  # Appropriate scale for viewing
        s.cross_section_scale = 1  # Identity cross-section scale
        
        # Set a good initial position based on image shape if available
        if image_shape is not None:
            # Center the view on the image center
            center_position = [dim // 2 for dim in image_shape]
            s.position = center_position
            print(f"Set initial position to image center: {center_position}")
        elif segment_ids and args.debug:
            # Fallback: rough center for a 192x192x192 volume
            s.position = [96, 96, 96]
            print(f"Set fallback position: [96, 96, 96]")
    
    # Print the Neuroglancer URL
    neuroglancer_url = viewer.get_viewer_url()
    print(f"Neuroglancer URL: {neuroglancer_url}")
    
    # Add instructions for users
    print("\n" + "="*80)
    print("MESH VIEWING INSTRUCTIONS:")
    print("="*80)
    print("1. In Neuroglancer, make sure the 'multiscale_mesh' layer is visible")
    print("2. Use the 'Segments' tab to control which meshes are displayed")
    print("3. Meshes should now align properly with the image coordinate system")
    print("4. Use the 3D view controls to rotate and zoom the meshes")
    print("5. Try adjusting the 'Rendering' settings if meshes don't appear")
    print("="*80)
    
    # Open the URL in a web browser
    print("Opening Neuroglancer in browser...")
    webbrowser.open(neuroglancer_url)
    
    # Keep the script running until user input
    print("\nPress Enter to exit...")
    input()
    
    # Shutdown HTTP server
    http_server.shutdown()


if __name__ == '__main__':
    main()
