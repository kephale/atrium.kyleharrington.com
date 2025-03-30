# /// script
# title = "Napari Precomputed Mesh Viewer"
# description = "A Python script to view precomputed mesh data in napari with proper multiscale mesh rendering"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.2.0"
# keywords = ["mesh", "3D", "visualization", "napari", "neuroglancer"]
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
from pathlib import Path
import numpy as np
import struct
from typing import Dict, List, Tuple, Optional, Any
import napari
import trimesh
import time

class PrecomputedMeshLoader:
    """Handles loading and parsing of Neuroglancer Precomputed meshes."""
    
    def __init__(self, precomputed_dir: Path, debug: bool = False):
        self.precomputed_dir = Path(precomputed_dir)
        self.debug = debug
        self.transform = None
        self.vertex_quantization_bits = None
        self._load_info()
        
    def _log(self, *args, **kwargs):
        """Debug logging helper."""
        if self.debug:
            print(*args, **kwargs)
            
    def _load_info(self):
        """Load the info file for the Precomputed dataset."""
        info_path = self.precomputed_dir / "info"
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
                
            self._log(f"Loaded info file: {info_path}")
            self._log(f"Info content: {json.dumps(info, indent=2)}")
            
            # Extract transformation matrix and other parameters
            if "transform" in info:
                self.transform = np.array(info["transform"]).reshape(3, 4)
                self._log(f"Found transform: {self.transform}")
            
            if "vertex_quantization_bits" in info:
                self.vertex_quantization_bits = info["vertex_quantization_bits"]
                self._log(f"Found vertex_quantization_bits: {self.vertex_quantization_bits}")
                
            # Store the whole info object for reference
            self.info = info
                
        except Exception as e:
            print(f"Error loading info file: {e}")
            self.info = {}
            
    def validate_directory(self) -> Tuple[bool, str, List[int]]:
        """Validate the Precomputed directory and return valid mesh IDs."""
        if not self.precomputed_dir.exists():
            return False, f"Directory does not exist: {self.precomputed_dir}", []
            
        if not os.path.isdir(self.precomputed_dir):
            return False, f"Path exists but is not a directory: {self.precomputed_dir}", []
            
        info_path = self.precomputed_dir / "info"
        if not info_path.exists():
            # Try to provide helpful diagnostics about the directory content
            files = list(self.precomputed_dir.iterdir())
            if files:
                file_list = ", ".join([f.name for f in files[:10]])
                if len(files) > 10:
                    file_list += f", ... (and {len(files)-10} more)"
                return False, f"Missing info file at: {info_path}. Directory contains: {file_list}", []
            else:
                return False, f"Missing info file at: {info_path}. Directory is empty.", []
                
        # Find valid mesh files
        valid_meshes, missing_index, missing_data = self._find_meshes()
        
        if not valid_meshes:
            message = "No valid meshes found."
            if missing_index:
                message += f" Missing index files for: {missing_index[:5]}"
                if len(missing_index) > 5:
                    message += f"... (and {len(missing_index)-5} more)"
            if missing_data:
                message += f" Missing data files for: {missing_data[:5]}"
                if len(missing_data) > 5:
                    message += f"... (and {len(missing_data)-5} more)"
            return False, message, []
            
        return True, f"Found {len(valid_meshes)} valid meshes", valid_meshes
    
    def _find_meshes(self) -> Tuple[List[int], List[int], List[int]]:
        """Find all valid meshes in the directory."""
        all_files = list(self.precomputed_dir.iterdir())
        self._log(f"Found {len(all_files)} files in directory")
        
        # Find potential mesh files (those with numeric names)
        numeric_files = [f for f in all_files if f.name.split('.')[0].isdigit()]
        
        # Separate mesh data and index files
        mesh_files = {int(p.stem): p for p in numeric_files if not p.name.endswith('.index')}
        index_files = {int(p.stem): p for p in numeric_files if p.name.endswith('.index')}
        
        self._log(f"Found {len(mesh_files)} mesh data files and {len(index_files)} index files")
        
        # Find complete mesh sets
        valid_meshes = sorted(set(mesh_files.keys()) & set(index_files.keys()))
        missing_index = sorted(set(mesh_files.keys()) - set(index_files.keys()))
        missing_data = sorted(set(index_files.keys()) - set(mesh_files.keys()))
        
        self._log(f"Valid meshes: {valid_meshes}")
        self._log(f"Meshes missing index: {missing_index}")
        self._log(f"Index files missing data: {missing_data}")
        
        return valid_meshes, missing_index, missing_data
        
    def read_manifest(self, mesh_id: int) -> Dict:
        """Read the binary manifest file for a mesh."""
        index_path = self.precomputed_dir / f"{mesh_id}.index"
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
                total_fragments = 0
                
                for lod in range(manifest["num_lods"]):
                    num_fragments = manifest["fragments_per_lod"][lod]
                    total_fragments += num_fragments
                    
                    if num_fragments > 0:
                        positions = np.frombuffer(f.read(12 * num_fragments), 
                                                dtype=np.uint32).reshape(num_fragments, 3)
                        sizes = np.frombuffer(f.read(4 * num_fragments), dtype=np.uint32)
                        
                        # Calculate offsets for each fragment in the data file
                        offsets = np.zeros(num_fragments, dtype=np.uint64)
                        if lod > 0:
                            # For LOD > 0, we need to sum up all previous LOD fragment sizes
                            for prev_lod in range(lod):
                                if prev_lod in manifest["fragments"]:
                                    offsets += np.sum(manifest["fragments"][prev_lod]["sizes"])
                            
                        # For fragments within this LOD, add cumulative offsets
                        if num_fragments > 1:
                            # First fragment starts at current offset
                            # Subsequent fragments start after the preceding ones
                            offsets[1:] += np.cumsum(sizes[:-1])
                        
                        manifest["fragments"][lod] = {
                            "positions": positions,
                            "sizes": sizes,
                            "offsets": offsets
                        }
            
            self._log(f"Manifest for mesh {mesh_id}: {manifest['num_lods']} LODs, {total_fragments} total fragments")
            return manifest
            
        except Exception as e:
            print(f"Error reading manifest for mesh {mesh_id}: {str(e)}")
            return {}
            
    def decode_draco_mesh(self, encoded_data: bytes) -> Optional[trimesh.Trimesh]:
        """Decode a Draco-encoded mesh to a trimesh.Trimesh object."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.drc', delete=False) as tmp_drc:
                tmp_drc.write(encoded_data)
                tmp_drc_path = tmp_drc.name
                
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_obj:
                tmp_obj_path = tmp_obj.name
            
            # Run draco_decoder to convert to OBJ format
            cmd = [
                "draco_decoder",
                "-i", tmp_drc_path,
                "-o", tmp_obj_path
            ]
            
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                self._log(f"Draco decoding failed: {process.stderr}")
                return None
                
            # Load as a trimesh
            mesh = trimesh.load(tmp_obj_path, file_type='obj', force='mesh')
            
            # Clean up temporary files
            os.unlink(tmp_drc_path)
            os.unlink(tmp_obj_path)
            
            return mesh
            
        except Exception as e:
            self._log(f"Error decoding Draco mesh: {str(e)}")
            return None
            
    def load_fragment(self, mesh_id: int, lod: int, fragment_idx: int, 
                     manifest: Dict) -> Optional[trimesh.Trimesh]:
        """Load a specific mesh fragment."""
        if lod not in manifest["fragments"]:
            self._log(f"LOD {lod} not found in manifest")
            return None
            
        fragments = manifest["fragments"][lod]
        if fragment_idx >= len(fragments["positions"]):
            self._log(f"Fragment index {fragment_idx} out of range for LOD {lod}")
            return None
            
        # Get fragment information
        position = fragments["positions"][fragment_idx]
        size = fragments["sizes"][fragment_idx]
        offset = fragments["offsets"][fragment_idx]
        
        # Read the fragment data
        data_path = self.precomputed_dir / str(mesh_id)
        
        try:
            with open(data_path, "rb") as f:
                f.seek(offset)
                encoded_data = f.read(size)
                
            # Decode the Draco mesh
            mesh = self.decode_draco_mesh(encoded_data)
            
            if mesh is None:
                return None
                
            # Apply transformations to convert from local coordinates to world coordinates
            scale = manifest["lod_scales"][lod]
            grid_origin = manifest["grid_origin"]
            
            # Rescale vertices to world coordinates 
            if self.vertex_quantization_bits:
                # Denormalize from quantized space
                normalization = (2**self.vertex_quantization_bits - 1)
                mesh.vertices = mesh.vertices / normalization * scale
            
            # Add grid origin and fragment position offsets
            box_offset = position * scale
            mesh.vertices = mesh.vertices + grid_origin + box_offset
            
            # Apply global transform if available
            if self.transform is not None:
                rotation = self.transform[:, :3]  # 3x3 rotation matrix
                translation = self.transform[:, 3]  # 3x1 translation vector
                
                mesh.vertices = np.dot(mesh.vertices, rotation.T) + translation
                
            return mesh
            
        except Exception as e:
            self._log(f"Error loading fragment: {str(e)}")
            return None
            
    def load_lod_mesh(self, mesh_id: int, lod: int, max_fragments: int = None) -> Optional[trimesh.Trimesh]:
        """Load all fragments for a specific LOD level and combine them."""
        manifest = self.read_manifest(mesh_id)
        if not manifest or "num_lods" not in manifest:
            print(f"Could not load manifest for mesh {mesh_id}")
            return None
            
        if lod not in manifest["fragments"] or manifest["fragments_per_lod"][lod] == 0:
            self._log(f"No fragments found for mesh {mesh_id} at LOD {lod}")
            return None
            
        fragments = manifest["fragments"][lod]
        num_fragments = len(fragments["positions"])
        
        if max_fragments is not None and max_fragments < num_fragments:
            # If limiting fragments, prioritize larger ones
            sizes = fragments["sizes"]
            indices = np.argsort(sizes)[-max_fragments:]
            self._log(f"Loading {len(indices)} of {num_fragments} fragments for LOD {lod}")
        else:
            indices = range(num_fragments)
            self._log(f"Loading all {num_fragments} fragments for LOD {lod}")
            
        # Load and combine fragments
        meshes = []
        for idx in indices:
            fragment = self.load_fragment(mesh_id, lod, idx, manifest)
            if fragment is not None:
                meshes.append(fragment)
                
        if not meshes:
            self._log(f"No valid fragments loaded for mesh {mesh_id} at LOD {lod}")
            return None
            
        # Combine all fragments into a single mesh
        if len(meshes) == 1:
            return meshes[0]
        else:
            try:
                combined = trimesh.util.concatenate(meshes)
                self._log(f"Combined {len(meshes)} fragments into a mesh with {len(combined.vertices)} vertices")
                return combined
            except Exception as e:
                self._log(f"Error combining fragments: {str(e)}")
                # Fall back to returning the first mesh
                return meshes[0] if meshes else None


def create_simple_proxy_mesh(manifest, mesh_id, lod):
    """Create a simple proxy mesh to represent a fragment without loading the full mesh."""
    if lod not in manifest["fragments"]:
        return None
        
    fragments = manifest["fragments"][lod]
    positions = fragments["positions"]
    
    if len(positions) == 0:
        return None
    
    # Get the grid origin and scale for this LOD
    grid_origin = manifest["grid_origin"]
    scale = manifest["lod_scales"][lod]
    
    # Create a simple box representing the extents of all fragments
    min_pos = np.min(positions, axis=0) * scale + grid_origin
    max_pos = np.max(positions, axis=0) * scale + grid_origin + scale
    
    # Create 8 corners of the bounding box
    vertices = np.array([
        [min_pos[0], min_pos[1], min_pos[2]],  # 0: front bottom left
        [max_pos[0], min_pos[1], min_pos[2]],  # 1: front bottom right
        [max_pos[0], max_pos[1], min_pos[2]],  # 2: front top right
        [min_pos[0], max_pos[1], min_pos[2]],  # 3: front top left
        [min_pos[0], min_pos[1], max_pos[2]],  # 4: back bottom left
        [max_pos[0], min_pos[1], max_pos[2]],  # 5: back bottom right
        [max_pos[0], max_pos[1], max_pos[2]],  # 6: back top right
        [min_pos[0], max_pos[1], max_pos[2]],  # 7: back top left
    ])
    
    # Define the faces of the box (12 triangles forming 6 sides)
    faces = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Back face
        [5, 4, 7], [5, 7, 6],
        # Left face
        [4, 0, 3], [4, 3, 7],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [0, 4, 5], [0, 5, 1]
    ])
    
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description="View precomputed mesh data in napari with dynamic LOD switching")
    parser.add_argument("--mesh-dir", type=str, required=True,
                       help="Directory containing precomputed mesh data")
    parser.add_argument("--proxy-mode", action="store_true",
                       help="Use simplified proxy meshes instead of loading full meshes")
    parser.add_argument("--num-meshes", type=int, default=5,
                       help="Number of meshes to load (default: 5, use 0 for all)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    try:
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
            print("To load more meshes, use --num-meshes parameter")
        else:
            load_meshes = valid_meshes
            print(f"Loading all {len(valid_meshes)} meshes")
            print(f"Valid mesh IDs: {valid_meshes}")
        
        # Initialize napari viewer with 3D mode
        viewer = napari.Viewer(ndisplay=3)
        
        # Store created layers for reference
        layers = {}
        
        # Process each mesh
        for mesh_id in load_meshes:
            try:
                # Read the manifest once
                manifest = mesh_loader.read_manifest(mesh_id)
                if not manifest or "num_lods" not in manifest:
                    print(f"Could not load manifest for mesh {mesh_id}")
                    continue
                    
                num_lods = manifest["num_lods"]
                print(f"Mesh {mesh_id} has {num_lods} LOD levels")
                
                # Generate a deterministic color for this mesh
                import hashlib
                color_seed = hashlib.md5(str(mesh_id).encode()).digest()
                base_color = np.array([
                    color_seed[0] / 255, 
                    color_seed[1] / 255, 
                    color_seed[2] / 255
                ])
                
                # Choose the highest LOD level available (lowest detail)
                highest_available_lod = -1
                for lod in range(num_lods-1, -1, -1):
                    if lod in manifest["fragments"] and manifest["fragments_per_lod"][lod] > 0:
                        highest_available_lod = lod
                        break
                        
                if highest_available_lod == -1:
                    print(f"No valid LOD levels found for mesh {mesh_id}")
                    continue
                
                # Create the mesh
                lod = highest_available_lod  # Use the highest LOD (lowest detail) for better performance
                
                if args.proxy_mode:
                    # Create a proxy mesh (simple bounding box) for better performance
                    mesh_data = create_simple_proxy_mesh(manifest, mesh_id, lod)
                    if mesh_data is None:
                        print(f"Could not create proxy mesh for mesh {mesh_id} at LOD {lod}")
                        continue
                        
                    vertices, faces = mesh_data
                else:
                    # Load the actual mesh - this can be slow for complex meshes
                    mesh = mesh_loader.load_lod_mesh(mesh_id, lod)
                    if mesh is None:
                        print(f"Could not load mesh {mesh_id} at LOD {lod}")
                        continue
                        
                    vertices, faces = mesh.vertices, mesh.faces
                
                # Create a vertex color array with the correct shape (num_vertices, 3)
                # The key fix here is to ensure we have one color per vertex
                values = np.tile(base_color, (len(vertices), 1))
                
                # Create a unique name for this layer
                layer_name = f"Mesh {mesh_id} - LOD {lod}"
                
                # Add the surface layer with specified properties
                surface = viewer.add_surface(
                    data=(vertices, faces, values),  # Include values for per-vertex coloring
                    name=layer_name,
                    opacity=0.7,
                    blending='translucent'
                )
                
                # Store the layer for future reference
                layers[mesh_id] = surface
                
                print(f"Added mesh {mesh_id} with {len(vertices)} vertices and {len(faces)} faces")
                
            except Exception as e:
                import traceback
                print(f"Error processing mesh {mesh_id}: {e}")
                if args.debug:
                    traceback.print_exc()
        
        if not layers:
            print("No meshes could be loaded. Try using --debug for more information.")
            sys.exit(1)
            
        # Reset view to show all objects
        viewer.reset_view()
        
        print("\nNapari viewer launched.")
        print("Use mouse to navigate: right-click and drag to rotate, middle-click to pan, scroll to zoom.")
        
        if args.proxy_mode:
            print("NOTE: Running in proxy mode with simplified meshes. Use --proxy-mode=False for full detail.")
            
        # Start the napari event loop
        napari.run()
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
