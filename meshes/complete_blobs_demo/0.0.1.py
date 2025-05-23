#!/usr/bin/env python
# /// script
# title = "3D Blob Meshing with NGFF Export"
# description = "A Python script to generate and visualize 3D meshes from scikit-image blobs, with export to OME-NGFF Zarr format with Neuroglancer Precomputed meshes - FIXED coordinate alignment"
# author = "Kyle Harrington (modified)"
# license = "MIT"
# version = "0.2.2"
# keywords = ["mesh", "3D", "visualization", "scikit-image", "zmesh", "neuroglancer", "OME-NGFF", "zarr"]
# documentation = "https://atrium.kyleharrington.com/meshes/generation/generate_blobs/index.html"
# classifiers = [
# "Development Status :: 3 - Alpha",
# "Intended Audience :: Science/Research",
# "License :: OSI Approved :: MIT License",
# "Programming Language :: Python :: 3.8",
# "Topic :: Scientific/Engineering :: Visualization",
# "Topic :: Scientific/Engineering :: Image Processing",
# ]
# requires-python = ">=3.8"
# dependencies = [
# "zmesh>=1.0.0",
# "scikit-image",
# "numpy",
# "trimesh",
# "pyfqmr",
# "shapely",
# "mapbox-earcut",
# "scipy",
# "zarr>3.0.0",
# "numcodecs"
# ]
# ///

# Install Draco first
# # Ubuntu/Debian
# sudo apt-get install draco-tools

# # macOS
# brew install draco

# # Or build from source:
# # https://github.com/google/draco

from zmesh import Mesher
import numpy as np
from skimage import data
from skimage.measure import label, find_contours
import json
from pathlib import Path
import trimesh
import pyfqmr
import subprocess
import tempfile
import os
import struct
from typing import List, Tuple, Dict, NamedTuple, Optional
import math
import zarr
import numcodecs
from scipy import ndimage
import shutil

from zarr.codecs.bytes import BytesCodec
from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding

class Fragment(NamedTuple):
    """Represents a mesh fragment with its position and encoded data."""
    position: np.ndarray  # 3D position of fragment in grid
    draco_bytes: bytes    # Draco-encoded mesh data
    size: int            # Size of encoded data in bytes
    lod: int            # Level of detail for this fragment

class NeuroglancerMeshWriter:
    def __init__(self, output_dir: str, box_size: int = 64, 
                 vertex_quantization_bits: int = 10,
                 transform: Optional[List[float]] = None,
                 clean_output: bool = False,
                 data_type: str = "uint64"):
        """Initialize the mesh writer with output directory and parameters.
        
        Args:
            output_dir: Base output directory
            box_size: Size of the smallest (LOD 0) chunks
            vertex_quantization_bits: Number of bits for vertex quantization (10 or 16)
            transform: Optional 4x3 homogeneous transform matrix (12 values)
            clean_output: If True, remove existing directory before starting
        """
        if vertex_quantization_bits not in (10, 16):
            raise ValueError("vertex_quantization_bits must be 10 or 16")
            
        self.output_dir = Path(output_dir)
        self.box_size = box_size
        self.vertex_quantization_bits = vertex_quantization_bits
        # Use identity transform by default to maintain coordinate system alignment
        self.transform = transform or [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        self.lod_scale_multiplier = 2.0
        self.data_type = data_type
        
        # Clean output directory if requested
        if clean_output and self.output_dir.exists():
            print(f"Cleaning existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def write_info_file(self):
        """Write the Neuroglancer info JSON file according to the specification at:
        https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-info-json-file-format
        """
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": self.vertex_quantization_bits,
            "transform": self.transform,
            "lod_scale_multiplier": self.lod_scale_multiplier
        }
        
        with open(self.output_dir / "info", "w") as f:
            json.dump(info, f)
            
        print(f"Created info file: {self.output_dir / 'info'}")

    def write_binary_manifest(self, mesh_id: int, fragments_by_lod: Dict[int, List[Fragment]], 
                            grid_origin: np.ndarray, num_lods: int):
        """Write the binary manifest file following the Neuroglancer precomputed mesh format."""
        # Ensure mesh ID is written as a base-10 string representation, as required by the spec
        mesh_id_str = str(mesh_id)
        manifest_path = self.output_dir / f"{mesh_id_str}.index"
        print(f"\nWriting manifest for mesh {mesh_id} to {manifest_path}")
        print(f"Number of LODs: {num_lods}")
        print(f"Grid origin: {grid_origin}")
        
        fragments_per_lod = [len(fragments_by_lod.get(lod, [])) 
                            for lod in range(num_lods)]
        print(f"Fragments per LOD: {fragments_per_lod}")
        
        try:
            with open(manifest_path, "wb") as f:
                # Write chunk shape (3x float32le)
                chunk_shape = np.array([self.box_size] * 3, dtype=np.float32)
                f.write(chunk_shape.tobytes())
                
                # Write grid origin (3x float32le)  
                grid_origin = np.array(grid_origin, dtype=np.float32)
                f.write(grid_origin.tobytes())
                
                # Write num_lods (uint32le)
                f.write(struct.pack("<I", num_lods))
                
                # Write lod_scales (num_lods x float32le)
                lod_scales = np.array([self.box_size * (2 ** i) for i in range(num_lods)], 
                                    dtype=np.float32)
                f.write(lod_scales.tobytes())
                
                # Write vertex_offsets ([num_lods, 3] array of float32le)
                vertex_offsets = np.zeros((num_lods, 3), dtype=np.float32)
                f.write(vertex_offsets.tobytes())
                
                # Write num_fragments_per_lod (num_lods x uint32le)
                fragments_per_lod_arr = np.array(fragments_per_lod, dtype=np.uint32)
                f.write(fragments_per_lod_arr.tobytes())
                
                # Write fragment data for each LOD
                for lod in range(num_lods):
                    fragments = fragments_by_lod.get(lod, [])
                    if not fragments:
                        continue
                    
                    # Sort fragments by Z-order
                    fragments = sorted(fragments, 
                                    key=lambda f: self._compute_z_order(f.position))
                    
                    # Write fragment positions ([num_fragments, 3] array of uint32le)
                    positions = np.array([f.position for f in fragments], dtype=np.uint32)
                    f.write(positions.tobytes())
                    
                    # Write fragment sizes (num_fragments x uint32le)
                    sizes = np.array([f.size for f in fragments], dtype=np.uint32)
                    f.write(sizes.tobytes())
            
            print(f"Successfully wrote manifest file: {manifest_path}")
            
        except Exception as e:
            print(f"Error writing manifest for mesh {mesh_id}: {str(e)}")
            raise

    def write_fragment_data(self, mesh_id: int, fragments_by_lod: Dict[int, List[Fragment]]):
        """Write the fragment data file following the Neuroglancer precomputed mesh format."""
        # Ensure mesh ID is written as a base-10 string representation, as required by the spec
        mesh_id_str = str(mesh_id)
        data_path = self.output_dir / mesh_id_str
        print(f"\nWriting fragment data for mesh {mesh_id} to {data_path}")
        
        try:
            with open(data_path, "wb") as f:
                total_fragments = 0
                total_bytes = 0
                
                for lod in sorted(fragments_by_lod.keys()):
                    lod_fragments = fragments_by_lod[lod]
                    
                    for fragment in sorted(lod_fragments, 
                                        key=lambda f: self._compute_z_order(f.position)):
                        f.write(fragment.draco_bytes)
                        total_fragments += 1
                        total_bytes += len(fragment.draco_bytes)
                
                print(f"Successfully wrote {total_fragments} fragments, {total_bytes} bytes")
                
        except Exception as e:
            print(f"Error writing fragment data for mesh {mesh_id}: {str(e)}")
            raise

    def process_mesh(self, mesh_id: int, mesh: trimesh.Trimesh, num_lods: int = 3):
        """Process a single mesh with CORRECTED coordinate alignment."""
        print(f"\nProcessing mesh {mesh_id}")
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"Mesh bounds: {mesh.vertices.min(axis=0)} to {mesh.vertices.max(axis=0)}")
        
        # Ensure mesh is watertight before processing
        if not mesh.is_watertight:
            print(f"Making mesh watertight before processing")
            mesh.fill_holes()
            mesh.fix_normals()
            
        # Apply moderate smoothing to improve quality
        print(f"Applying pre-processing smoothing")
        mesh = mesh.smoothed(lamb=0.5, iterations=3)
        
        # Generate LODs
        lod_meshes = self.generate_lods(mesh, num_lods)
        print(f"Generated {len(lod_meshes)} LOD levels")
        
        # CRITICAL FIX: Calculate grid origin properly for coordinate alignment
        # The mesh vertices are already in the correct image coordinate system
        # Grid origin should be 0,0,0 or aligned to mesh bounds without additional offsets
        mesh_min = mesh.vertices.min(axis=0)
        grid_origin = np.floor(mesh_min / self.box_size) * self.box_size
        print(f"Grid origin: {grid_origin}")
        print(f"Mesh minimum: {mesh_min}")
        
        # Generate fragments for each LOD
        fragments_by_lod = {}
        for lod, lod_mesh in enumerate(lod_meshes):
            print(f"\nProcessing LOD {lod}")
            print(f"LOD mesh: {len(lod_mesh.vertices)} vertices, {len(lod_mesh.faces)} faces")
            
            # Keep vertices in original coordinate system - no coordinate transformations
            fragments = self.generate_fragments(lod_mesh, lod, grid_origin)
            
            if fragments:
                fragments_by_lod[lod] = fragments
                print(f"Generated {len(fragments)} fragments for LOD {lod}")
            else:
                print(f"No fragments generated for LOD {lod}")
        
        if not fragments_by_lod:
            print(f"Warning: No fragments generated for any LOD level")
            return
        
        try:
            # Write manifest and fragment data
            self.write_binary_manifest(mesh_id, fragments_by_lod, grid_origin, num_lods)
            self.write_fragment_data(mesh_id, fragments_by_lod)
            
            # Verify the files exist and are accessible
            mesh_id_str = str(mesh_id)
            index_path = self.output_dir / f"{mesh_id_str}.index"
            data_path = self.output_dir / mesh_id_str
            
            if index_path.exists() and data_path.exists():
                print(f"Successfully processed mesh {mesh_id}")
                print(f"  Index file: {index_path} ({index_path.stat().st_size} bytes)")
                print(f"  Data file: {data_path} ({data_path.stat().st_size} bytes)")
            else:
                if not index_path.exists():
                    print(f"ERROR: Index file {index_path} does not exist!")
                if not data_path.exists():
                    print(f"ERROR: Data file {data_path} does not exist!")
        except Exception as e:
            print(f"Error processing mesh {mesh_id}: {str(e)}")
            raise

    def _compute_z_order(self, pos: np.ndarray) -> int:
        """Compute Z-order curve index for a 3D position."""
        x, y, z = pos
        answer = 0
        for i in range(21):  # Support up to 21 bits per dimension
            answer |= ((x & (1 << i)) << (3 * i)) | \
                     ((y & (1 << i)) << (3 * i + 1)) | \
                     ((z & (1 << i)) << (3 * i + 2))
        return answer

    def generate_fragments(self, mesh: trimesh.Trimesh, lod: int, grid_origin: np.ndarray,
                            enforce_grid_partition: bool = True) -> List[Fragment]:
        """Generate mesh fragments for a given LOD level with corrected coordinate handling."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return []
            
        print(f"LOD {lod}: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        current_box_size = self.box_size * (2 ** lod)
        vertices = mesh.vertices
        
        # Calculate fragment bounds relative to grid origin
        relative_vertices = vertices - grid_origin
        start_fragment = np.floor(relative_vertices.min(axis=0) / current_box_size).astype(int)
        end_fragment = np.ceil(relative_vertices.max(axis=0) / current_box_size).astype(int)
        
        # Increased overlap factor to reduce gaps between fragments
        overlap_factor = 0.5 * current_box_size  # 50% overlap
        
        fragments = []
        fragment_count = 0
        for x in range(start_fragment[0], end_fragment[0]):
            for y in range(start_fragment[1], end_fragment[1]):
                for z in range(start_fragment[2], end_fragment[2]):
                    pos = np.array([x, y, z])
                    
                    # Calculate bounds in world coordinates relative to grid origin
                    bounds_min = grid_origin + pos * current_box_size - overlap_factor
                    bounds_max = bounds_min + current_box_size + (2 * overlap_factor)
                    
                    # Use expanded bounds for selecting vertices to ensure overlap
                    mask = np.all((vertices >= bounds_min) & (vertices < bounds_max), axis=1)
                    vertex_indices = np.where(mask)[0]
                    
                    if len(vertex_indices) == 0:
                        continue
                        
                    fragment_vertices = vertices[vertex_indices]
                    
                    # Remap faces to use new vertex indices
                    vertex_map = {old: new for new, old in enumerate(vertex_indices)}
                    face_mask = np.all(np.isin(mesh.faces, vertex_indices), axis=1)
                    fragment_faces = mesh.faces[face_mask]
                    
                    if len(fragment_faces) == 0:
                        continue
                        
                    fragment_faces = np.array([[vertex_map[v] for v in face] 
                                            for face in fragment_faces])
                    
                    # Create fragment mesh
                    fragment_mesh = trimesh.Trimesh(
                        vertices=fragment_vertices,
                        faces=fragment_faces
                    )
                    
                    # For LOD > 0, enforce 2x2x2 grid partitioning
                    if enforce_grid_partition and lod > 0:
                        fragment_mesh = self._enforce_grid_partition(
                            fragment_mesh,
                            bounds_min,
                            current_box_size
                        )
                    
                    if len(fragment_mesh.vertices) == 0 or len(fragment_mesh.faces) == 0:
                        continue
                    
                    try:
                        # Encode using Draco
                        draco_bytes = self._encode_mesh_draco(
                            fragment_mesh.vertices,
                            fragment_mesh.faces,
                            bounds_min,
                            current_box_size
                        )
                        
                        if len(draco_bytes) > 12:
                            fragments.append(Fragment(
                                position=pos,
                                draco_bytes=draco_bytes,
                                size=len(draco_bytes),
                                lod=lod
                            ))
                            fragment_count += 1
                            
                    except Exception as e:
                        print(f"Error processing fragment at {pos}: {str(e)}")
                        continue
        
        print(f"LOD {lod}: Generated {fragment_count} fragments")
        return fragments

    def _encode_mesh_draco(self, vertices: np.ndarray, faces: np.ndarray,
                        bounds_min: np.ndarray, box_size: float) -> bytes:
        """Encode a mesh using Google's Draco encoder with proper coordinate normalization."""
        vertices = vertices.copy()
        
        # Calculate actual bounds of the vertices for more precise quantization
        vertex_min = vertices.min(axis=0)
        vertex_max = vertices.max(axis=0)
        vertex_range = vertex_max - vertex_min
        
        # Use a smaller padding to maintain precision
        padding = np.maximum(vertex_range * 0.001, 0.01)  # 0.1% padding or minimum 0.01 units
        
        # Normalize vertices to [0, 1] range within the padded bounds
        padded_min = vertex_min - padding
        padded_range = vertex_range + 2 * padding
        
        # Avoid division by zero
        padded_range = np.maximum(padded_range, 1e-10)
        
        normalized_vertices = (vertices - padded_min) / padded_range
        normalized_vertices *= (2**self.vertex_quantization_bits - 1)
        normalized_vertices = normalized_vertices.astype(np.int32)
        
        # Create temporary files for the mesh
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = os.path.join(temp_dir, "temp.obj")
            drc_path = os.path.join(temp_dir, "temp.drc")
            
            # Write simple OBJ file
            with open(obj_path, "w") as f:
                for v in normalized_vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces + 1:  # OBJ indices are 1-based
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            
            # Get the appropriate draco_encoder path
            if os.path.exists("/opt/homebrew/bin/draco_encoder"):
                draco_path = "/opt/homebrew/bin/draco_encoder"
            elif os.path.exists("/usr/bin/draco_encoder"):
                draco_path = "/usr/bin/draco_encoder" 
            elif os.path.exists("/usr/local/bin/draco_encoder"):
                draco_path = "/usr/local/bin/draco_encoder"
            else:
                raise RuntimeError("Cannot find draco_encoder. Please install it.")
                
            # Run draco_encoder with improved settings
            cmd = [
                draco_path,
                "-i", obj_path,
                "-o", drc_path,
                "-qp", str(self.vertex_quantization_bits),
                "-qt", "10",
                "-qn", "10",
                "-qtx", "10",
                "-cl", "7",
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                with open(drc_path, "rb") as f:
                    return f.read()
                    
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Draco encoding failed: {e.stdout} {e.stderr}")

    def _enforce_grid_partition(self, mesh: trimesh.Trimesh, 
                            bounds_min: np.ndarray,
                            box_size: float) -> trimesh.Trimesh:
        """Enforce grid partitioning for better fragment alignment."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return mesh
            
        try:
            # Calculate grid planes
            mid_points = bounds_min + np.array([
                [box_size/2, 0, 0],
                [0, box_size/2, 0],
                [0, 0, box_size/2]
            ])
            
            normals = np.eye(3)
            
            # Split mesh along each plane with enhanced capping
            result_mesh = mesh
            for point, normal in zip(mid_points, normals):
                try:
                    new_mesh = result_mesh.slice_plane(point, normal, cap=True)
                    if new_mesh is not None and len(new_mesh.vertices) > 0:
                        new_mesh.fill_holes()
                        new_mesh.fix_normals()
                        result_mesh = new_mesh
                except ValueError:
                    continue
                    
            if len(result_mesh.vertices) == 0:
                return mesh
                
            return result_mesh
            
        except Exception:
            return mesh

    def decimate_mesh(self, mesh: trimesh.Trimesh, target_ratio: float) -> trimesh.Trimesh:
        """Decimate a mesh to a target ratio of original faces."""
        # Skip tiny meshes
        if len(mesh.faces) < 20:
            return mesh
            
        try:
            # Make a copy to avoid modifying the original
            mesh_copy = mesh.copy()
            
            # Fix any potential issues with the mesh before simplification
            if not mesh_copy.is_watertight:
                mesh_copy.fill_holes()
                
            # Update faces 
            mesh_copy.update_faces(mesh_copy.unique_faces())
            mesh_copy.update_faces(mesh_copy.nondegenerate_faces())
            mesh_copy.fix_normals()
            
            # Set up simplifier
            simplifier = pyfqmr.Simplify()
            simplifier.setMesh(mesh_copy.vertices, mesh_copy.faces)
            
            # Calculate target face count
            target_count = max(int(len(mesh_copy.faces) * target_ratio), 12)
            
            # Simplify with conservative settings for better quality
            simplifier.simplify_mesh(target_count=target_count, 
                                   aggressiveness=2,
                                   preserve_border=True,
                                   verbose=False)
                                   
            vertices, faces, _ = simplifier.getMesh()
            
            # Check if we got valid results
            if len(vertices) < 3 or len(faces) < 1:
                return mesh
                
            # Create a new trimesh with the decimated mesh
            result = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Post-process the simplified mesh
            result.update_faces(result.nondegenerate_faces())
            result.fix_normals()
            
            return result
            
        except Exception as e:
            print(f"Error during mesh decimation: {str(e)}")
            return mesh

    def generate_lods(self, mesh: trimesh.Trimesh, num_lods: int) -> List[trimesh.Trimesh]:
        """Generate levels of detail for a mesh."""
        lods = [mesh]  # LOD 0 is the highest detail
        current_mesh = mesh
        
        for i in range(1, num_lods):
            target_ratio = 0.4  # Aggressive reduction for clearer LOD differences
            decimated = self.decimate_mesh(current_mesh, target_ratio)
            
            # Ensure each LOD has fewer vertices than the previous
            if len(decimated.vertices) > 0.8 * len(current_mesh.vertices):
                decimated = self.decimate_mesh(current_mesh, 0.25)
                
            print(f"LOD {i}: {len(current_mesh.vertices)} → {len(decimated.vertices)} vertices" + 
                 f" ({len(decimated.vertices)/len(current_mesh.vertices):.2f}x)")
                 
            lods.append(decimated)
            current_mesh = decimated
            
        return lods

def create_ome_zarr_group(root_dir, name, blob_data, labels_data=None, mesh_writer=None):
    """Create an OME-NGFF Zarr store following the 0.5 specification."""
    # Create path
    zarr_path = os.path.join(root_dir, f"{name}.zarr")
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    
    # Create 3D multiscale image (z, y, x) with a channel dimension
    z, y, x = blob_data.shape
    blob_data_with_channel = blob_data.reshape(1, z, y, x)
    
    print(f"Creating Zarr v3 group at {zarr_path}")
    root = zarr.open(zarr_path, mode='w')
    
    # Set base OME-NGFF attributes
    root.attrs.put({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5"
            }
        }
    })
    
    # Create multiscales metadata with proper coordinate system specification
    multiscales_metadata = {
        "ome": {
            "version": "0.5",
            "multiscales": [{
                "name": "3d_blobs",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"}
                ],
                "datasets": [],
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0, 1.0]  # Identity transform
                    }
                ],
                "type": "gaussian",
                "metadata": {
                    "description": "3D blobs with mesh",
                    "method": "skimage.data.binary_blobs",
                    "version": "0.19.3"
                }
            }],
            "omero": {
                "name": "3D Blobs",
                "channels": [{
                    "active": True,
                    "coefficient": 1,
                    "color": "FFFFFF",
                    "family": "linear",
                    "inverted": False,
                    "label": "Blobs",
                    "window": {
                        "end": 1,
                        "max": 1,
                        "min": 0,
                        "start": 0
                    }
                }],
                "rdefs": {
                    "defaultT": 0,
                    "defaultZ": z // 2,
                    "model": "greyscale"
                }
            }
        }
    }
    
    # Create pyramid levels
    pyramid_levels = 3
    datasets_info = []
    
    print("Creating resolution levels in zarr store...")
    current_data = blob_data_with_channel.astype(np.uint8)
    
    for i in range(pyramid_levels):
        # Create group for this resolution level
        level_group = zarr.open(os.path.join(zarr_path, str(i)), mode='w')
        
        level_group.attrs.put({
            "zarr_format": 3,
            "node_type": "group"
        })
        
        # Create a 4D array (c, z, y, x)
        chunks = (1, min(16, current_data.shape[1]), 
                  min(64, current_data.shape[2]), 
                  min(64, current_data.shape[3]))
        
        # Create zarr array
        array = zarr.create(
            store=os.path.join(zarr_path, str(i), "0"),
            shape=current_data.shape,
            chunks=chunks,
            dtype=current_data.dtype,
            codecs=[BytesCodec()],
            chunk_key_encoding=DefaultChunkKeyEncoding(separator="/")
        )
        
        # Set array metadata 
        array.attrs.put({
            "zarr_format": 3,
            "node_type": "array",
            "dimension_names": ["c", "z", "y", "x"]
        })
        
        # Write data
        array[:] = current_data
        
        # Add dataset info to multiscales metadata
        scale_factor = 2 ** i
        datasets_info.append({
            "path": f"{i}/0",
            "coordinateTransformations": [{
                "type": "scale",
                "scale": [1.0, scale_factor, scale_factor, scale_factor]
            }]
        })
        
        # Downsample for next level
        if i < pyramid_levels - 1:
            current_data = current_data[:, ::2, ::2, ::2]
    
    # Update multiscales metadata
    multiscales_metadata["ome"]["multiscales"][0]["datasets"] = datasets_info
    root.attrs.put(multiscales_metadata)
    
    # Add labels if provided
    if labels_data is not None:
        print("Creating labels group...")
        
        # Create labels group
        labels_group = zarr.open(os.path.join(zarr_path, "labels"), mode='w')
        labels_group.attrs.put({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "labels": ["segmentation/0"]
                }
            }
        })
        
        # Create segmentation group
        seg_group = zarr.open(os.path.join(zarr_path, "labels", "segmentation"), mode='w')
        
        # Add a channel dimension to labels data
        labels_with_channel = labels_data.reshape(1, *labels_data.shape)
        
        # Create label multiscales metadata
        seg_group.attrs.put({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "ome": {
                    "version": "0.5",
                    "multiscales": [{
                        "name": "blob_segmentation",
                        "axes": [
                            {"name": "c", "type": "channel"},
                            {"name": "z", "type": "space", "unit": "micrometer"},
                            {"name": "y", "type": "space", "unit": "micrometer"},
                            {"name": "x", "type": "space", "unit": "micrometer"}
                        ],
                        "datasets": [],
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0, 1.0, 1.0]  # Identity transform
                            }
                        ]
                    }],
                    "image-label": {
                        "version": "0.5",
                        "colors": [],
                        "properties": [],
                        "source": {
                            "image": "../../"
                        }
                    }
                }
            }
        })
        
        # Create label pyramid levels
        current_labels = labels_with_channel
        label_datasets_info = []
        
        for i in range(pyramid_levels):
            chunks = (1, min(16, current_labels.shape[1]), 
                      min(64, current_labels.shape[2]), 
                      min(64, current_labels.shape[3]))
            
            label_array = zarr.create(
                store=os.path.join(zarr_path, "labels", "segmentation", str(i)),
                shape=current_labels.shape,
                chunks=chunks,
                dtype=current_labels.dtype,
                codecs=[BytesCodec()],
                chunk_key_encoding=DefaultChunkKeyEncoding(separator="/")
            )
            
            label_array.attrs.put({
                "zarr_format": 3,
                "node_type": "array",
                "dimension_names": ["c", "z", "y", "x"]
            })
            
            # Write data
            label_array[:] = current_labels
            
            scale_factor = 2 ** i
            label_datasets_info.append({
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [1.0, scale_factor, scale_factor, scale_factor]
                }]
            })
            
            # Downsample for next level
            if i < pyramid_levels - 1:
                current_labels = current_labels[:, ::2, ::2, ::2]
        
        # Update multiscales metadata for labels
        seg_attrs = seg_group.attrs.asdict()
        seg_attrs["attributes"]["ome"]["multiscales"][0]["datasets"] = label_datasets_info
        
        # Add color information for labels
        unique_labels = np.unique(labels_data)
        colors = []
        properties = []
        
        import random
        for label in unique_labels:
            if label == 0:  # background
                rgba = [0, 0, 0, 0]  # transparent
            else:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                rgba = [r, g, b, 200]  # semi-transparent
            
            colors.append({
                "label-value": int(label),
                "rgba": rgba
            })
            
            if label > 0:
                voxel_count = np.sum(labels_data == label)
                properties.append({
                    "label-value": int(label),
                    "voxel-count": int(voxel_count),
                    "name": f"Blob {int(label)}"
                })
        
        seg_attrs["attributes"]["ome"]["image-label"]["colors"] = colors
        seg_attrs["attributes"]["ome"]["image-label"]["properties"] = properties
        seg_group.attrs.put(seg_attrs)
    
    # Create mesh directory
    mesh_dir = os.path.join(zarr_path, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Create zarr.json for the mesh group according to RFC-8
    mesh_metadata = {
        "zarr_format": 3,
        "node_type": "external",
        "attributes": {
            "ome": {
                "version": "0.5",
                "mesh": {
                    "version": "0.1",
                    "type": "neuroglancer_multilod_draco",
                    "source": {
                        "image": "../",
                        "labels": "../labels/segmentation"
                    }
                }
            }
        }
    }
    
    # Write the mesh metadata file
    with open(os.path.join(mesh_dir, "zarr.json"), "w") as f:
        json.dump(mesh_metadata, f, indent=2)
    
    # Copy mesh data if provided
    if mesh_writer is not None:
        mesh_source_dir = mesh_writer.output_dir
        print(f"\nCopying mesh files from {mesh_source_dir} to {mesh_dir}")
        
        source_files = os.listdir(mesh_source_dir)
        
        for item in source_files:
            src_path = os.path.join(mesh_source_dir, item)
            dst_path = os.path.join(mesh_dir, item)
            
            if not os.path.isfile(src_path):
                continue
                
            shutil.copy2(src_path, dst_path)
            if os.path.exists(dst_path):
                print(f"  ✓ Copied {item} ({os.path.getsize(dst_path)} bytes)")
    
    return zarr_path

def main():
    """Main function with corrected coordinate alignment."""
    # Create output directory for the mesh data
    temp_mesh_dir = Path("temp_precomputed")
    mesh_writer = NeuroglancerMeshWriter(
        temp_mesh_dir,
        box_size=32,
        vertex_quantization_bits=16,
        transform=[1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0],  # Identity transform
        clean_output=True
    )
    print(f"Creating meshes in: {temp_mesh_dir.absolute()}")
    
    # Generate sample data
    volume_size = 192
    print(f"Generating 3D blobs with size {volume_size}")
    blobs = data.binary_blobs(length=volume_size, n_dim=3, 
                             volume_fraction=0.2,
                             rng=42)
    
    # Apply smoothing
    print("Applying morphological operations and smoothing...")
    blobs = ndimage.binary_closing(blobs, iterations=3)
    blobs = ndimage.binary_opening(blobs, iterations=2)
    blobs = ndimage.gaussian_filter(blobs.astype(float), sigma=1.2) > 0.5
    
    # Label the connected components
    print("Labeling connected components...")
    labels = label(blobs)
    num_labels = labels.max()
    print(f"Found {num_labels} separate blob objects")
    print(f"Labels bounds: {labels.min()} to {labels.max()}")
    print(f"Image shape: {blobs.shape}")
    
    # Filter out very small blobs
    sizes = ndimage.sum(blobs, labels, range(1, num_labels+1))
    mask = sizes > 100
    filtered_labels = np.zeros_like(labels)
    for i, keep in enumerate(mask, 1):
        if keep:
            filtered_labels[labels == i] = i
    
    num_filtered_labels = len(np.unique(filtered_labels)) - 1
    print(f"After filtering small blobs: {num_filtered_labels} objects remain")
    labels = filtered_labels
    
    # Print coordinate system information for debugging
    print(f"\nCOORDINATE SYSTEM DEBUG INFO:")
    print(f"Image data shape: {blobs.shape}")
    print(f"Image bounds: (0,0,0) to {blobs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Non-zero labels: {np.unique(labels)[1:]}")  # Skip background
    
    # Sample a few label positions for debugging
    for label_id in np.unique(labels)[1:6]:  # Check first 5 non-background labels
        if label_id > 0:
            positions = np.where(labels == label_id)
            min_pos = [pos.min() for pos in positions]
            max_pos = [pos.max() for pos in positions]
            center_pos = [(min_pos[i] + max_pos[i])/2 for i in range(3)]
            print(f"Label {label_id}: center at {center_pos}, bounds {min_pos} to {max_pos}")
    
    # Create meshes - this is where coordinate alignment matters most
    print("Generating meshes...")
    mesher = Mesher((1.0, 1.0, 1.0))  # Unit voxel spacing matches image coordinates
    mesher.mesh(labels, close=True)
    
    # Process each mesh
    mesh_id = 1
    mesh_object_count = 0
    
    object_ids = list(mesher.ids())
    print(f"Processing {len(object_ids)} meshes...")
    
    for obj_id in object_ids:
        mesh = mesher.get(obj_id, normals=True)
        
        # Skip meshes with too few vertices or faces
        if len(mesh.vertices) < 10 or len(mesh.faces) < 10:
            print(f"Skipping small mesh {obj_id}")
            mesher.erase(obj_id)
            continue
            
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        
        # CRITICAL DEBUG: Print mesh coordinate information
        print(f"\nMesh {mesh_id} (object {obj_id}) coordinate analysis:")
        print(f"  Mesh vertices bounds: {trimesh_mesh.vertices.min(axis=0)} to {trimesh_mesh.vertices.max(axis=0)}")
        print(f"  Mesh center: {(trimesh_mesh.vertices.min(axis=0) + trimesh_mesh.vertices.max(axis=0))/2}")
        
        # Verify mesh is in same coordinate system as labels
        mesh_center = (trimesh_mesh.vertices.min(axis=0) + trimesh_mesh.vertices.max(axis=0))/2
        
        # Find the corresponding label center for comparison
        label_positions = np.where(labels == obj_id)
        if len(label_positions[0]) > 0:
            label_center = [np.mean(pos) for pos in label_positions]
            print(f"  Corresponding label center: {label_center}")
            print(f"  Coordinate difference: {np.array(mesh_center) - np.array(label_center)}")
        
        # Apply smoothing
        print(f"Smoothing mesh {mesh_id} ({len(trimesh_mesh.vertices)} vertices)...")
        trimesh_mesh = trimesh_mesh.smoothed(lamb=0.75, iterations=10)
        
        # Fix mesh issues
        if not trimesh_mesh.is_watertight:
            print(f"Fixing non-watertight mesh {mesh_id}...")
            trimesh_mesh.fill_holes()
        
        trimesh_mesh.fix_normals()
        
        # CRITICAL: Do NOT apply any coordinate transformations here
        # The mesh vertices are already in the correct image coordinate system
        print(f"Final mesh bounds: {trimesh_mesh.vertices.min(axis=0)} to {trimesh_mesh.vertices.max(axis=0)}")
        
        # Process the mesh with multiple LODs
        mesh_writer.process_mesh(mesh_id, trimesh_mesh, num_lods=3)
        mesh_id += 1
        mesh_object_count += 1
        mesher.erase(obj_id)
    
    # Write mesh info file
    mesh_writer.write_info_file()
    
    print(f"Generated {mesh_object_count} Neuroglancer Precomputed meshes")
    
    # Create the NGFF zarr group
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    zarr_path = create_ome_zarr_group(
        output_dir, 
        "blob_with_meshes", 
        blobs, 
        labels_data=labels,
        mesh_writer=mesh_writer
    )
    
    print(f"Created NGFF zarr with meshes at: {zarr_path}")
    
    # Clean up temporary mesh directory
    if temp_mesh_dir.exists():
        shutil.rmtree(temp_mesh_dir)

if __name__ == "__main__":
    main()
