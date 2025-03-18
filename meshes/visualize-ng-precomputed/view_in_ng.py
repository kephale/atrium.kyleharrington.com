# /// script
# title = "Neuroglancer Precomputed Mesh Viewer"
# description = "A Python script to view precomputed mesh data using Neuroglancer with webdriver"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["mesh", "3D", "visualization", "neuroglancer"]
# documentation = "https://github.com/google/neuroglancer"
# requires-python = ">=3.11"
# dependencies = [
#     "neuroglancer[webdriver]",
#     "numpy"
# ]
# ///

import argparse
import neuroglancer
import neuroglancer.cli
from pathlib import Path
import json
import webbrowser
import sys
import signal
import time
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

class PrecomputedMeshValidator:
    def __init__(self, precomputed_dir: Path):
        self.precomputed_dir = Path(precomputed_dir)
        
    def validate_directory_structure(self) -> Tuple[bool, str]:
        """Validate basic directory structure and info file."""
        if not self.precomputed_dir.exists():
            return False, f"Directory does not exist: {self.precomputed_dir}"
        
        print(f"Checking precomputed directory: {self.precomputed_dir}")
        if not os.path.isdir(self.precomputed_dir):
            return False, f"Path exists but is not a directory: {self.precomputed_dir}"
            
        info_path = self.precomputed_dir / "info"
        if not info_path.exists():
            # Try to provide helpful diagnostics about the directory content
            files = list(self.precomputed_dir.iterdir())
            if files:
                file_list = ", ".join([f.name for f in files[:10]])
                if len(files) > 10:
                    file_list += f", ... (and {len(files)-10} more)"
                return False, f"Missing info file at: {info_path}. Directory contains: {file_list}"
            else:
                return False, f"Missing info file at: {info_path}. Directory is empty."
            
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                
            # Validate required info file fields
            required_fields = ["data_type", "num_channels", "scales"]
            missing_fields = [field for field in required_fields if field not in info]
            if missing_fields:
                return False, f"Info file missing required fields: {missing_fields}"
                
            # Check for Neuroglancer compatibility - be more lenient about type
            if "@type" in info and info["@type"] != "neuroglancer_multilod_draco":
                print(f"Warning: Info file has @type '{info.get('@type')}', expected 'neuroglancer_multilod_draco'")
                # But continue anyway, as different @type values might still work
        except json.JSONDecodeError:
            return False, f"Invalid JSON in info file: {info_path}"
        except Exception as e:
            return False, f"Error reading info file: {str(e)}"
            
        return True, "Directory structure valid"
        
    def find_mesh_files(self) -> Dict[str, List[int]]:
        """Find and validate mesh files and their associated indexes."""
        result = {
            "complete_meshes": [],
            "missing_index": [],
            "missing_data": []
        }
        
        # Get all files in the directory for better diagnostics
        all_files = list(self.precomputed_dir.iterdir())
        print(f"Found {len(all_files)} total files in {self.precomputed_dir}")
        
        # Filter for numerical files (potential mesh data)
        numeric_files = [f for f in all_files if f.name.split('.')[0].isdigit()]
        print(f"Found {len(numeric_files)} numeric files")
        
        # Separate mesh data and index files
        mesh_files = {p.stem: p for p in numeric_files if not p.name.endswith('.index')}
        index_files = {p.stem: p for p in numeric_files if p.name.endswith('.index')}
        
        print(f"Found {len(mesh_files)} mesh data files and {len(index_files)} index files")
        
        # Check for complete mesh sets (both data and index)
        for mesh_id in mesh_files:
            if mesh_id in index_files:
                try:
                    result["complete_meshes"].append(int(mesh_id))
                except ValueError:
                    continue
            else:
                try:
                    result["missing_index"].append(int(mesh_id))
                except ValueError:
                    continue
                    
        for index_id in index_files:
            if index_id not in mesh_files:
                try:
                    result["missing_data"].append(int(index_id))
                except ValueError:
                    continue
        
        # Provide more diagnostics
        if result["complete_meshes"]:
            print(f"Complete meshes found: {sorted(result['complete_meshes'])}")
        
        return result
        
    def validate_mesh_data(self, mesh_id: int) -> Tuple[bool, str]:
        """Validate individual mesh data and index files."""
        mesh_path = self.precomputed_dir / str(mesh_id)
        index_path = self.precomputed_dir / f"{mesh_id}.index"
        
        if not mesh_path.exists():
            return False, f"Mesh data file missing: {mesh_path}"
        if not index_path.exists():
            return False, f"Mesh index file missing: {index_path}"
            
        try:
            mesh_size = mesh_path.stat().st_size
            if mesh_size == 0:
                return False, f"Mesh data file is empty: {mesh_path}"
                
            index_size = index_path.stat().st_size
            if index_size == 0:
                return False, f"Mesh index file is empty: {index_path}"
                
        except Exception as e:
            return False, f"Error checking mesh files: {str(e)}"
            
        return True, "Mesh data valid"

def verify_precomputed_mesh(directory: str) -> Dict:
    """Run all validations and return detailed results."""
    validator = PrecomputedMeshValidator(Path(directory))
    
    results = {
        "directory_valid": False,
        "directory_message": "",
        "mesh_files": {},
        "valid_meshes": [],
        "invalid_meshes": {}
    }
    
    # Check directory structure
    dir_valid, dir_message = validator.validate_directory_structure()
    results["directory_valid"] = dir_valid
    results["directory_message"] = dir_message
    
    if not dir_valid:
        return results
        
    # Find all mesh files
    results["mesh_files"] = validator.find_mesh_files()
    
    # Validate each complete mesh
    for mesh_id in results["mesh_files"]["complete_meshes"]:
        valid, message = validator.validate_mesh_data(mesh_id)
        if valid:
            results["valid_meshes"].append(mesh_id)
        else:
            results["invalid_meshes"][mesh_id] = message
            
    return results

def setup_viewer(precomputed_dir: Path):
    """Set up the Neuroglancer viewer with precomputed mesh data."""
    # Validate the precomputed data
    validation_results = verify_precomputed_mesh(str(precomputed_dir))
    
    if not validation_results["directory_valid"]:
        print(f"Error: {validation_results['directory_message']}")
        sys.exit(1)
        
    if not validation_results["valid_meshes"]:
        print("\nNo valid meshes found. Issues detected:")
        if validation_results["mesh_files"]["missing_index"]:
            print(f"- Meshes missing index files: {validation_results['mesh_files']['missing_index']}")
        if validation_results["mesh_files"]["missing_data"]:
            print(f"- Index files missing mesh data: {validation_results['mesh_files']['missing_data']}")
        if validation_results["invalid_meshes"]:
            print("\nInvalid meshes:")
            for mesh_id, message in validation_results["invalid_meshes"].items():
                print(f"- Mesh {mesh_id}: {message}")
        
        print("\nTip: If using 'minimal-ng-precomputed', ensure you've generated the meshes with sequential IDs.")
        print("     The meshes should be generated in the 'precomputed' directory.")
        sys.exit(1)
        
    print(f"\nFound {len(validation_results['valid_meshes'])} valid meshes")
    print(f"Valid mesh IDs: {sorted(validation_results['valid_meshes'])}")
    
    # Initialize viewer with validated meshes
    viewer = neuroglancer.Viewer()
    
    # Ensure the path is properly formatted
    absolute_path = str(precomputed_dir.absolute())
    print(f"Using data source: precomputed://{absolute_path}")
    
    with viewer.txn() as s:
        s.layers['meshes'] = neuroglancer.SegmentationLayer(
            source=f'precomputed://{absolute_path}',
            segments=validation_results["valid_meshes"]
        )
        s.layout = '3d'
        s.show_axis_lines = True
        
        # Set some reasonable defaults for the view
        s.perspective_zoom = 1024
        s.perspective_orientation = [0.5, 0.5, 0.5, 0.5]
                
    return viewer

def main():
    parser = argparse.ArgumentParser(description="View precomputed mesh data in Neuroglancer")
    neuroglancer.cli.add_server_arguments(parser)
    parser.add_argument("--mesh-dir", type=str, required=True,
                      help="Directory containing precomputed mesh data")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup local server to allow access to local files
    neuroglancer.set_server_bind_address('127.0.0.1')
    neuroglancer.set_static_content_source(neuroglancer.LocalStaticContentSource())
    
    if args.debug:
        neuroglancer.set_server_bind_address('127.0.0.1')
        neuroglancer.set_static_content_source(neuroglancer.LocalStaticContentSource())
        
    neuroglancer.cli.handle_server_arguments(args)
    
    try:
        viewer = setup_viewer(Path(args.mesh_dir))
        
        print("\nViewer URL:")
        url = viewer.get_viewer_url()
        print(url)
        
        print("\nViewer State:")
        try:
            state_json = viewer.state.to_json()
            # Handle both dict and string formats
            if isinstance(state_json, dict):
                print(json.dumps(state_json, indent=2))
            else:
                print(json.dumps(json.loads(state_json), indent=2))
        except Exception as e:
            print(f"Could not display viewer state: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
        
        try:
            webbrowser.open(url)
            print("Opened viewer in web browser")
        except Exception as e:
            print(f"Could not open browser automatically: {str(e)}")
            print("Please copy and paste the URL into your browser manually.")
        
        print("\nControls:")
        print("- Right mouse button: Rotate")
        print("- Left mouse button: Pan")
        print("- Mouse wheel: Zoom")
        print("- 'r' key: Reset view")
        print("- 'h' key: Show help")
        print("- In the 'meshes' layer's settings (gear icon), you can change which meshes are visible")
        print("- You can toggle visibility of individual meshes in the segmentation tab")
        
        print("\nServer is running. Press Ctrl+C to exit...")
        try:
            signal.signal(signal.SIGINT, lambda signal, frame: neuroglancer.stop_web_server())
            while True:
                sleep_time = 1000000  # Keep main thread alive
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nShutting down viewer server...")
            neuroglancer.stop_web_server()
        except Exception as e:
            print(f"\nError during viewer execution: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
