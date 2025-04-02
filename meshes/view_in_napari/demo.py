#!/usr/bin/env python
# /// script
# title = "Napari Precomputed Mesh Viewer"
# description = "A Python script to view precomputed mesh data in napari with proper multiscale mesh rendering that automatically scales meshes to match image data"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.4.3"
# keywords = ["mesh", "3D", "visualization", "napari", "neuroglancer"]
# documentation = "https://napari.org/stable/api/napari.html"
# requires-python = ">=3.8"
# dependencies = [
#     "napari",
#     "numpy",
#     "PyQt5", 
#     "draco",
#     "trimesh",
#     "zarr>=3.0.0"
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
import zarr

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
                
            print(f"Loaded info file: {info_path}")
            
            # Extract transformation matrix and other parameters
            if "transform" in info:
                self.transform = np.array(info["transform"]).reshape(3, 4)
                print(f"Found transform: {self.transform}")
            
            if "vertex_quantization_bits" in info:
                self.vertex_quantization_bits = info["vertex_quantization_bits"]
                print(f"Found vertex_quantization_bits: {self.vertex_quantization_bits}")
                
            # Store the whole info object for reference
            self.info = info
                
        except Exception as e:
            print(f"Error loading info file: {e}")
            self.info = {}
            
    def get_valid_mesh_ids(self) -> List[int]:
        """Get list of valid mesh IDs."""
        valid_meshes, _, _ = self._find_meshes()
        return valid_meshes
            
    def _find_meshes(self) -> Tuple[List[int], List[int], List[int]]:
        """Find all valid meshes in the directory."""
        all_files = list(self.precomputed_dir.iterdir())
        print(f"Found {len(all_files)} files in directory")
        
        # Find potential mesh files (those with numeric names)
        numeric_files = [f for f in all_files if f.name.split('.')[0].isdigit()]
        
        # Separate mesh data and index files
        mesh_files = {int(p.stem): p for p in numeric_files if not p.name.endswith('.index')}
        index_files = {int(p.stem): p for p in numeric_files if p.name.endswith('.index')}
        
        print(f"Found {len(mesh_files)} mesh data files and {len(index_files)} index files")
        
        # Find complete mesh sets
        valid_meshes = sorted(set(mesh_files.keys()) & set(index_files.keys()))
        missing_index = sorted(set(mesh_files.keys()) - set(index_files.keys()))
        missing_data = sorted(set(index_files.keys()) - set(mesh_files.keys()))
        
        print(f"Valid meshes: {valid_meshes[:10]}..." if len(valid_meshes) > 10 else f"Valid meshes: {valid_meshes}")
        
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
                        
                        # Start offset is the sum of all previous LOD fragments sizes
                        base_offset = 0
                        if lod > 0:
                            # For LOD > 0, we need to sum up all previous LOD fragment sizes
                            for prev_lod in range(lod):
                                if prev_lod in manifest["fragments"]:
                                    base_offset += np.sum(manifest["fragments"][prev_lod]["sizes"])
                        
                        # All offsets start at the base offset for this LOD
                        offsets[:] = base_offset
                        
                        # For fragments within this LOD, add cumulative offsets
                        if num_fragments > 1:
                            # First fragment starts at current base_offset
                            # Subsequent fragments start after the preceding ones
                            offsets[1:] += np.cumsum(sizes[:-1])
                        
                        manifest["fragments"][lod] = {
                            "positions": positions,
                            "sizes": sizes,
                            "offsets": offsets
                        }
            
            print(f"Manifest for mesh {mesh_id}: {manifest['num_lods']} LODs, {total_fragments} total fragments")
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
            print(f"Loading {len(indices)} of {num_fragments} fragments for LOD {lod}")
        else:
            indices = range(num_fragments)
            print(f"Loading all {num_fragments} fragments for LOD {lod}")
            
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
                print(f"Combined {len(meshes)} fragments into a mesh with {len(combined.vertices)} vertices")
                return combined
            except Exception as e:
                self._log(f"Error combining fragments: {str(e)}")
                # Fall back to returning the first mesh
                return meshes[0] if meshes else None

class ZarrLoader:
    """Handles loading and parsing of OME-NGFF Zarr data."""
    
    def __init__(self, zarr_path: str, debug: bool = False):
        self.zarr_path = Path(zarr_path)
        self.debug = debug
        self.root = None
        self.multiscales = None
        self.labels = None
        self.mesh_dir = None
        
    def _log(self, *args, **kwargs):
        """Debug logging helper."""
        if self.debug:
            print(*args, **kwargs)
            
    def open(self) -> bool:
        """Open and validate the Zarr store."""
        try:
            # Open the zarr store
            self.root = zarr.open(str(self.zarr_path))
            print(f"Opened Zarr store at {self.zarr_path}")
            
            # Check for multiscales metadata
            if hasattr(self.root, 'attrs') and 'ome' in self.root.attrs:
                if 'multiscales' in self.root.attrs['ome']:
                    self.multiscales = self.root.attrs['ome']['multiscales']
                    print(f"Found multiscales metadata: {len(self.multiscales)} dataset(s)")
                else:
                    print("No multiscales found in OME metadata")
            
            # Check for labels
            if 'labels' in self.root:
                self.labels = self.root['labels']
                print(f"Found labels group with keys: {list(self.labels.keys())}")
            
            # Check for mesh directory
            mesh_dir = self.zarr_path / 'meshes'
            if mesh_dir.exists() and mesh_dir.is_dir():
                self.mesh_dir = mesh_dir
                print(f"Found meshes directory: {mesh_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error opening Zarr store: {str(e)}")
            return False
    
    def load_image_data(self, resolution_level: int = 0) -> Optional[np.ndarray]:
        """Load image data at the specified resolution level."""
        if self.multiscales is None or len(self.multiscales) == 0:
            print("No multiscales metadata found")
            return None
        
        try:
            # Get path to the dataset from multiscales metadata
            multiscale = self.multiscales[0]  # Use the first multiscale group
            if 'datasets' not in multiscale or len(multiscale['datasets']) == 0:
                print("No datasets defined in multiscales metadata")
                return None
            
            # Check if requested resolution level exists
            if resolution_level >= len(multiscale['datasets']):
                print(f"Resolution level {resolution_level} not available, using highest level")
                resolution_level = len(multiscale['datasets']) - 1
            
            # Get dataset path
            dataset_path = multiscale['datasets'][resolution_level]['path']
            print(f"Loading image data from {dataset_path}")
            
            # Check if path exists as a direct group or needs to be interpreted as a relative path
            if dataset_path in self.root:
                dataset = self.root[dataset_path]
            else:
                # Try to resolve relative path
                dataset = self.root
                for part in dataset_path.split('/'):
                    if part in dataset:
                        dataset = dataset[part]
                    else:
                        print(f"Could not find path component '{part}' in dataset")
                        return None
            
            # Load the data
            data = dataset[:]
            
            # If data is 4D with a channel dimension of 1, squeeze it
            if data.ndim == 4 and data.shape[0] == 1:
                data = data.squeeze(0)
            
            print(f"Loaded image data with shape {data.shape} and dtype {data.dtype}")
            return data
            
        except Exception as e:
            print(f"Error loading image data: {str(e)}")
            return None
    
    def load_labels_data(self, resolution_level: int = 0) -> Optional[np.ndarray]:
        """Load labels data at the specified resolution level."""
        if self.labels is None:
            print("No labels group found")
            return None
        
        try:
            # Check for segmentation data
            if 'segmentation' not in self.labels:
                print("No segmentation data found in labels group")
                return None
            
            segmentation = self.labels['segmentation']
            
            # Get metadata for multiscale organization
            if hasattr(segmentation, 'attrs') and 'ome' in segmentation.attrs:
                if 'multiscales' in segmentation.attrs['ome']:
                    multiscales = segmentation.attrs['ome']['multiscales']
                    
                    # Check if requested resolution level exists
                    if len(multiscales) > 0 and 'datasets' in multiscales[0]:
                        datasets = multiscales[0]['datasets']
                        
                        if resolution_level >= len(datasets):
                            print(f"Resolution level {resolution_level} not available for labels, using highest level")
                            resolution_level = len(datasets) - 1
                        
                        # Get dataset path
                        dataset_path = datasets[resolution_level]['path']
                        print(f"Loading labels data from {dataset_path}")
                        
                        # Open and load dataset
                        dataset = segmentation[dataset_path]
                        data = dataset[:]
                        
                        # If data is 4D with a channel dimension of 1, squeeze it
                        if data.ndim == 4 and data.shape[0] == 1:
                            data = data.squeeze(0)
                        
                        print(f"Loaded labels data with shape {data.shape} and dtype {data.dtype}")
                        return data
                        
            # Fallback: try to load data directly from numeric keys
            if str(resolution_level) in segmentation:
                data = segmentation[str(resolution_level)][:]
                
                # If data is 4D with a channel dimension of 1, squeeze it
                if data.ndim == 4 and data.shape[0] == 1:
                    data = data.squeeze(0)
                    
                print(f"Loaded labels data with shape {data.shape} and dtype {data.dtype}")
                return data
            
            print(f"Could not find labels data at resolution level {resolution_level}")
            return None
            
        except Exception as e:
            print(f"Error loading labels data: {str(e)}")
            return None
    
    def get_mesh_dir(self) -> Optional[Path]:
        """Return the mesh directory if it exists."""
        return self.mesh_dir
    
    def get_scale_transform(self, resolution_level: int = 0) -> Optional[np.ndarray]:
        """Get scale transform for the specified resolution level."""
        if self.multiscales is None or len(self.multiscales) == 0:
            return None
        
        try:
            multiscale = self.multiscales[0]
            
            # Global transform from multiscales metadata
            global_transform = np.eye(4)
            if 'coordinateTransformations' in multiscale:
                for transform in multiscale['coordinateTransformations']:
                    if transform['type'] == 'scale':
                        # Update scale - typically [1, z, y, x] but we only need the spatial dimensions
                        scale = np.array(transform['scale'])
                        if len(scale) == 4:  # [channel, z, y, x]
                            global_transform[0, 0] = scale[1]  # z
                            global_transform[1, 1] = scale[2]  # y
                            global_transform[2, 2] = scale[3]  # x
                        elif len(scale) == 3:  # [z, y, x]
                            global_transform[0, 0] = scale[0]  # z
                            global_transform[1, 1] = scale[1]  # y
                            global_transform[2, 2] = scale[2]  # x
            
            # Dataset-specific transform
            if 'datasets' in multiscale and resolution_level < len(multiscale['datasets']):
                dataset = multiscale['datasets'][resolution_level]
                if 'coordinateTransformations' in dataset:
                    dataset_transform = np.eye(4)
                    for transform in dataset['coordinateTransformations']:
                        if transform['type'] == 'scale':
                            # Update scale
                            scale = np.array(transform['scale'])
                            if len(scale) == 4:  # [channel, z, y, x]
                                dataset_transform[0, 0] = scale[1]  # z
                                dataset_transform[1, 1] = scale[2]  # y
                                dataset_transform[2, 2] = scale[3]  # x
                            elif len(scale) == 3:  # [z, y, x]
                                dataset_transform[0, 0] = scale[0]  # z
                                dataset_transform[1, 1] = scale[1]  # y
                                dataset_transform[2, 2] = scale[2]  # x
                    
                    # Combine global and dataset transforms
                    combined_transform = np.dot(global_transform, dataset_transform)
                    return combined_transform
            
            return global_transform
            
        except Exception as e:
            print(f"Error getting scale transform: {str(e)}")
            return None

def load_mesh(mesh_loader, mesh_id, lod=None, max_fragments=None, load_all_lods=False):
    """
    Load a mesh or multiple LOD levels of a mesh.
    
    Args:
        mesh_loader: The PrecomputedMeshLoader instance
        mesh_id: ID of the mesh to load
        lod: Specific LOD level to load (ignored if load_all_lods=True)
        max_fragments: Maximum number of fragments to load per LOD
        load_all_lods: If True, load all available LOD levels and return a dictionary
                       mapping LOD levels to (vertices, faces) tuples
    
    Returns:
        If load_all_lods=True: Dictionary of {lod: (vertices, faces)} for each LOD level
        If load_all_lods=False: Tuple of (vertices, faces) for a single LOD level
        None if no valid mesh data could be loaded
    """

    """Load a mesh and return vertices and faces."""
    try:
        # Read the manifest
        manifest = mesh_loader.read_manifest(mesh_id)
        if not manifest or "num_lods" not in manifest:
            print(f"Could not load manifest for mesh {mesh_id}")
            return None
        
        num_lods = manifest["num_lods"]
        
        # If load_all_lods is True, load all available LODs and return a dictionary
        if load_all_lods:
            meshes = {}
            for l in range(num_lods):
                if l in manifest["fragments"] and manifest["fragments_per_lod"][l] > 0:
                    mesh = mesh_loader.load_lod_mesh(mesh_id, l, max_fragments)
                    if mesh is not None:
                        meshes[l] = (mesh.vertices, mesh.faces)
                        print(f"Loaded LOD {l} for mesh {mesh_id} with {len(mesh.vertices)} vertices")
            
            if not meshes:
                print(f"No valid LOD levels found for mesh {mesh_id}")
                return None
                
            return meshes
        
        # Otherwise load a single LOD
        else:
            # Determine which LOD to use
            if lod is not None and lod < num_lods:
                # Use the specified LOD
                target_lod = lod
            else:
                # Choose the lowest LOD (highest detail) that has fragments
                target_lod = None
                for l in range(num_lods):
                    if l in manifest["fragments"] and manifest["fragments_per_lod"][l] > 0:
                        target_lod = l
                        break
                        
                # If no valid LOD found, try the highest one
                if target_lod is None:
                    for l in range(num_lods-1, -1, -1):
                        if l in manifest["fragments"] and manifest["fragments_per_lod"][l] > 0:
                            target_lod = l
                            break
            
            if target_lod is None:
                print(f"No valid LOD levels found for mesh {mesh_id}")
                return None
                
            # Load the mesh
            mesh = mesh_loader.load_lod_mesh(mesh_id, target_lod, max_fragments)
            if mesh is None:
                return None
                
            return mesh.vertices, mesh.faces
        
    except Exception as e:
        print(f"Error loading mesh {mesh_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Zarr and Precomputed Mesh Viewer for Napari with automatic multiscale mesh support")
    parser.add_argument("--zarr-path", type=str, required=True,
                       help="Path to the Zarr store containing OME-NGFF data with meshes")
    parser.add_argument("--num-meshes", type=int, default=0,
                       help="Number of meshes to load (default: 0 for all)")
    parser.add_argument("--resolution", type=int, default=0,
                       help="Image resolution level to load (default: 0 for highest resolution)")
    parser.add_argument("--skip-images", action="store_true",
                       help="Skip loading image data")
    parser.add_argument("--skip-labels", action="store_true",
                       help="Skip loading label data")
    parser.add_argument("--skip-meshes", action="store_true",
                       help="Skip loading meshes")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--mesh-scale", type=float, nargs=3, default=None,
                       help="Override mesh scale (z, y, x)")
    parser.add_argument("--load-all-lods", action="store_true", default=True,
                       help="Load all LOD levels for each mesh (default: True)")
    parser.add_argument("--one-lod-per-mesh", action="store_true",
                       help="Load only a single LOD level per mesh (overrides --load-all-lods)")
    
    args = parser.parse_args()
    
    try:
        # Initialize the Zarr loader
        zarr_loader = ZarrLoader(args.zarr_path, debug=args.debug)
        
        # Open the Zarr store
        if not zarr_loader.open():
            print(f"Error: Could not open Zarr store at {args.zarr_path}")
            sys.exit(1)
        
        # Initialize napari viewer with 3D mode
        viewer = napari.Viewer(ndisplay=3)
        
        # Get scale transform for all layers to ensure consistency
        transform = zarr_loader.get_scale_transform(args.resolution)
        scale = np.ones(3)
        if transform is not None:
            scale = np.array([transform[0, 0], transform[1, 1], transform[2, 2]])
            print(f"Using scale from metadata: {scale}")
        
        # Override scale if specified
        if args.mesh_scale is not None:
            scale = np.array(args.mesh_scale)
            print(f"Using override scale: {scale}")
        
        # Load image data
        if not args.skip_images:
            image_data = zarr_loader.load_image_data(args.resolution)
            if image_data is not None:
                # Add image layer to napari
                viewer.add_image(
                    image_data,
                    name="Image",
                    scale=scale,
                    colormap='gray',
                    blending='additive'
                )
                print(f"Added image layer with shape {image_data.shape}, scale {scale}")
            else:
                print("No image data found or could not be loaded")
        
        # Load labels data
        if not args.skip_labels:
            labels_data = zarr_loader.load_labels_data(args.resolution)
            if labels_data is not None:
                # Add labels layer to napari
                viewer.add_labels(
                    labels_data,
                    name="Segmentation",
                    scale=scale,
                    opacity=0.5,
                    blending='translucent'
                )
                print(f"Added labels layer with shape {labels_data.shape}, scale {scale}")
            else:
                print("No labels data found or could not be loaded")
        
        # Check for meshes directory
        if not args.skip_meshes:
            mesh_dir = zarr_loader.get_mesh_dir()
            if mesh_dir is not None:
                print(f"Loading meshes from {mesh_dir}")
                
                # Initialize the mesh loader
                mesh_loader = PrecomputedMeshLoader(mesh_dir, debug=args.debug)
                
                # Get valid mesh IDs
                valid_meshes = mesh_loader.get_valid_mesh_ids()
                
                if valid_meshes:
                    print(f"\nFound {len(valid_meshes)} valid meshes")
                    
                        # Determine how many to load
                    if args.num_meshes > 0 and args.num_meshes < len(valid_meshes):
                        load_meshes = valid_meshes[:args.num_meshes]
                        print(f"Loading first {args.num_meshes} of {len(valid_meshes)} meshes")
                    else:
                        load_meshes = valid_meshes
                        print(f"Loading all {len(valid_meshes)} meshes")
                        
                    # Print out information about the meshing mode
                    if args.load_all_lods and not args.one_lod_per_mesh:
                        print("Using multi-LOD mode: Loading all available mesh LODs with scale matching the image")
                    else:
                        print("Using single-LOD mode: Loading only one LOD level per mesh")
                    
                    # Load each mesh
                    for mesh_id in load_meshes:
                        # Determine whether to load all LODs or a single one
                        load_all_lods = args.load_all_lods and not args.one_lod_per_mesh
                        
                        result = load_mesh(mesh_loader, mesh_id, load_all_lods=load_all_lods)
                        if result:
                            if load_all_lods and isinstance(result, dict):
                                # Add each LOD as a separate layer, using scale matching the resolution level
                                for lod, (vertices, faces) in result.items():
                                    # Calculate scale for this LOD
                                    mesh_scale = scale.copy()
                                    
                                    # Get the manifest to extract lod_scales
                                    manifest = mesh_loader.read_manifest(mesh_id)
                                    if manifest and "lod_scales" in manifest:
                                        # Detect the transform matrix from the mesh info file
                                        mesh_transform_scale = None
                                        if hasattr(mesh_loader, 'transform') and mesh_loader.transform is not None:
                                            # Extract the scaling component from the transform matrix
                                            # The transform is typically a 3x4 matrix where the diagonal elements
                                            # represent scaling in x, y, z directions
                                            diagonal = np.array([mesh_loader.transform[i, i] for i in range(3)])
                                            if np.all(diagonal > 0):  # Ensure valid scale factors
                                                mesh_transform_scale = np.mean(diagonal)
                                                print(f"Detected mesh transform scale: {mesh_transform_scale}")
                                        
                                        # If we detect a scaling difference between the mesh and image coordinate systems,
                                        # apply a compensation factor
                                        if mesh_transform_scale is not None and mesh_transform_scale != 1.0:
                                            # Apply the detected scale factor uniformly to all LOD levels
                                            mesh_scale = scale.copy() / mesh_transform_scale
                                            print(f"Applying uniform scale adjustment of 1/{mesh_transform_scale}")
                                        else:
                                            mesh_scale = scale.copy()
                                            print("Using default 1 scale adjustment for all LOD levels")
                                    
                                    viewer.add_surface(
                                        data=(vertices, faces),
                                        name=f"Mesh {mesh_id} (LOD {lod})",
                                        colormap='turbo',
                                        opacity=0.7,
                                        scale=mesh_scale
                                    )
                                    print(f"Added mesh {mesh_id} LOD {lod} with {len(vertices)} vertices and {len(faces)} faces, scale {mesh_scale}")
                            else:
                                # Single LOD case
                                vertices, faces = result
                                
                                # Add surface layer with the same scale as images and labels
                                viewer.add_surface(
                                    data=(vertices, faces),
                                    name=f"Mesh {mesh_id}",
                                    colormap='turbo',
                                    opacity=0.7,
                                    scale=scale
                                )
                                print(f"Added mesh {mesh_id} with {len(vertices)} vertices and {len(faces)} faces, scale {scale}")
                else:
                    print("No valid meshes found")
            else:
                print("No meshes directory found")
        
        # Reset view to show all objects
        viewer.reset_view()
        
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