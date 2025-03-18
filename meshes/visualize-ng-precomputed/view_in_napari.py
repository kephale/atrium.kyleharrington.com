# /// script
# title = "Napari Precomputed Mesh Viewer"
# description = "A Python script to view precomputed mesh data in napari with each resolution as a separate layer"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["mesh", "3D", "visualization", "napari"]
# documentation = "https://napari.org/stable/api/napari.html"
# requires-python = ">=3.11"
# dependencies = [
#     "napari",
#     "numpy",
#     "PyQt5"
# ]
# ///

import argparse
import json
import sys
import os
from pathlib import Path
import numpy as np
import struct
from typing import Dict, List, Tuple, Optional
import napari

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

def read_mesh_manifest(precomputed_dir: Path, mesh_id: int) -> Dict:
    """Read the binary manifest file for a mesh."""
    index_path = precomputed_dir / f"{mesh_id}.index"
    manifest = {}
    
    try:
        with open(index_path, "rb") as f:
            # Read header information
            manifest["chunk_shape"] = np.frombuffer(f.read(12), dtype=np.float32)  # 3x float32
            manifest["grid_origin"] = np.frombuffer(f.read(12), dtype=np.float32)  # 3x float32
            manifest["num_lods"] = np.frombuffer(f.read(4), dtype=np.uint32)[0]    # 1x uint32
            
            # Read LOD scales
            manifest["lod_scales"] = np.frombuffer(f.read(4 * manifest["num_lods"]), dtype=np.float32)
            
            # Read vertex offsets
            manifest["vertex_offsets"] = np.frombuffer(f.read(12 * manifest["num_lods"]), 
                                           dtype=np.float32).reshape(manifest["num_lods"], 3)
            
            # Read fragment counts per LOD
            manifest["fragments_per_lod"] = np.frombuffer(f.read(4 * manifest["num_lods"]), dtype=np.uint32)
            
            # For each LOD, read fragment positions and sizes
            manifest["fragments"] = {}
            for lod in range(manifest["num_lods"]):
                num_fragments = manifest["fragments_per_lod"][lod]
                if num_fragments > 0:
                    positions = np.frombuffer(f.read(12 * num_fragments), 
                                            dtype=np.uint32).reshape(num_fragments, 3)
                    sizes = np.frombuffer(f.read(4 * num_fragments), dtype=np.uint32)
                    
                    manifest["fragments"][lod] = {
                        "positions": positions,
                        "sizes": sizes
                    }
        
        return manifest
    except Exception as e:
        print(f"Error reading mesh manifest for ID {mesh_id}: {str(e)}")
        return {}

def verify_precomputed_mesh(directory: str) -> Dict:
    """Run all validations and return detailed results."""
    validator = PrecomputedMeshValidator(Path(directory))
    
    results = {
        "directory_valid": False,
        "directory_message": "",
        "mesh_files": {},
        "valid_meshes": []
    }
    
    # Check directory structure
    dir_valid, dir_message = validator.validate_directory_structure()
    results["directory_valid"] = dir_valid
    results["directory_message"] = dir_message
    
    if not dir_valid:
        return results
        
    # Find all mesh files
    results["mesh_files"] = validator.find_mesh_files()
    results["valid_meshes"] = results["mesh_files"]["complete_meshes"]
            
    return results

def create_proxy_mesh(positions: np.ndarray, scale: float, grid_origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple proxy mesh representation for a fragment."""
    if len(positions) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    
    # Calculate min and max positions in world coordinates
    min_pos = grid_origin + positions.min(axis=0) * scale
    max_pos = grid_origin + (positions.max(axis=0) + 1) * scale  # +1 because positions are indices
    
    # Create a box representing the volume
    vertices = np.array([
        [min_pos[0], min_pos[1], min_pos[2]],
        [max_pos[0], min_pos[1], min_pos[2]],
        [min_pos[0], max_pos[1], min_pos[2]],
        [max_pos[0], max_pos[1], min_pos[2]],
        [min_pos[0], min_pos[1], max_pos[2]],
        [max_pos[0], min_pos[1], max_pos[2]],
        [min_pos[0], max_pos[1], max_pos[2]],
        [max_pos[0], max_pos[1], max_pos[2]],
    ])
    
    # Define the edges of the box
    edges = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Connecting edges
    ])
    
    return vertices, edges

def main():
    parser = argparse.ArgumentParser(description="View precomputed mesh data in napari")
    parser.add_argument("--mesh-dir", type=str, required=True,
                       help="Directory containing precomputed mesh data")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    try:
        # Validate the precomputed data
        precomputed_dir = Path(args.mesh_dir)
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
            
            print("\nTip: If using 'minimal-ng-precomputed', ensure you've generated the meshes with sequential IDs.")
            print("     The meshes should be generated in the 'precomputed' directory.")
            sys.exit(1)
            
        print(f"\nFound {len(validation_results['valid_meshes'])} valid meshes")
        print(f"Valid mesh IDs: {sorted(validation_results['valid_meshes'])}")
        
        # Initialize napari viewer
        viewer = napari.Viewer(ndisplay=3)
        
        mesh_colors = {}  # Store a consistent color for each mesh ID
        
        # Process each mesh
        for mesh_id in sorted(validation_results["valid_meshes"]):
            print(f"\nLoading mesh {mesh_id}")
            
            # Generate a consistent color for this mesh
            import hashlib
            color_seed = hashlib.md5(str(mesh_id).encode()).digest()
            mesh_colors[mesh_id] = (
                color_seed[0] / 255, 
                color_seed[1] / 255, 
                color_seed[2] / 255
            )
            
            # Read the manifest
            manifest = read_mesh_manifest(precomputed_dir, mesh_id)
            if not manifest or "num_lods" not in manifest:
                print(f"Could not load manifest for mesh {mesh_id}")
                continue
                
            print(f"Mesh {mesh_id} has {manifest['num_lods']} LOD levels")
            
            # Create a layer for each LOD
            for lod in range(manifest["num_lods"]):
                if lod not in manifest["fragments"] or manifest["fragments_per_lod"][lod] == 0:
                    print(f"  LOD {lod}: No fragments")
                    continue
                    
                fragments = manifest["fragments"][lod]
                print(f"  LOD {lod}: {len(fragments['positions'])} fragments, scale={manifest['lod_scales'][lod]}")
                
                # Create a proxy representation for all fragments in this LOD
                vertices, edges = create_proxy_mesh(
                    fragments["positions"], 
                    manifest["lod_scales"][lod],
                    manifest["grid_origin"]
                )
                
                # Add as shapes layer (box outlines)
                lines = []
                for i in range(len(edges)):
                    lines.append(np.array([vertices[edges[i, 0]], vertices[edges[i, 1]]]))
                
                # Convert color tuple to string format '#RRGGBB'
                color_hex = f'#{int(mesh_colors[mesh_id][0]*255):02x}{int(mesh_colors[mesh_id][1]*255):02x}{int(mesh_colors[mesh_id][2]*255):02x}'
                
                viewer.add_shapes(
                    lines,
                    shape_type='line',
                    edge_width=2,
                    edge_color=color_hex,
                    name=f"Mesh {mesh_id} - LOD {lod}"
                )
        
        print("\nNapari viewer launched. Each layer represents a different mesh LOD level.")
        napari.run()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
