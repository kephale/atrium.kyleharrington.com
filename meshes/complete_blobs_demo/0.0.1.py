#!/usr/bin/env python
# /// script
# title = "3D Blob Meshing with NGFF Export"
# description = "A Python script to generate and visualize 3D meshes from scikit-image blobs, with export to OME-NGFF Zarr format with Neuroglancer Precomputed meshes"
# author = "Kyle Harrington (modified)"
# license = "MIT"
# version = "0.2.0"
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
        """Write the binary manifest file following the Neuroglancer precomputed mesh format.
        
        The manifest file format is described at:
        https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-manifest-file-format
        """
        """Write the binary manifest file with debug logging."""
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
                print(f"Wrote chunk shape: {chunk_shape}")
                
                # Write grid origin (3x float32le)
                grid_origin = np.array(grid_origin, dtype=np.float32)
                f.write(grid_origin.tobytes())
                print(f"Wrote grid origin: {grid_origin}")
                
                # Write num_lods (uint32le)
                f.write(struct.pack("<I", num_lods))
                print(f"Wrote num_lods: {num_lods}")
                
                # Write lod_scales (num_lods x float32le)
                lod_scales = np.array([self.box_size * (2 ** i) for i in range(num_lods)], 
                                    dtype=np.float32)
                f.write(lod_scales.tobytes())
                print(f"Wrote lod_scales: {lod_scales}")
                
                # Write vertex_offsets ([num_lods, 3] array of float32le)
                vertex_offsets = np.zeros((num_lods, 3), dtype=np.float32)
                f.write(vertex_offsets.tobytes())
                print(f"Wrote vertex_offsets")
                
                # Write num_fragments_per_lod (num_lods x uint32le)
                fragments_per_lod_arr = np.array(fragments_per_lod, dtype=np.uint32)
                f.write(fragments_per_lod_arr.tobytes())
                print(f"Wrote fragments_per_lod: {fragments_per_lod_arr}")
                
                # Write fragment data for each LOD
                for lod in range(num_lods):
                    fragments = fragments_by_lod.get(lod, [])
                    if not fragments:
                        print(f"No fragments for LOD {lod}")
                        continue
                    
                    print(f"Writing {len(fragments)} fragments for LOD {lod}")
                    
                    # Sort fragments by Z-order
                    fragments = sorted(fragments, 
                                    key=lambda f: self._compute_z_order(f.position))
                    
                    # Write fragment positions ([num_fragments, 3] array of uint32le)
                    positions = np.array([f.position for f in fragments], dtype=np.uint32)
                    f.write(positions.tobytes())
                    print(f"Wrote fragment positions for LOD {lod}")
                    
                    # Write fragment sizes (num_fragments x uint32le)
                    sizes = np.array([f.size for f in fragments], dtype=np.uint32)
                    f.write(sizes.tobytes())
                    print(f"Wrote fragment sizes for LOD {lod}")
            
            print(f"Successfully wrote manifest file: {manifest_path}")
            # Verify file exists and has content
            print(f"File exists: {manifest_path.exists()}")
            print(f"File size: {manifest_path.stat().st_size} bytes")
            
        except Exception as e:
            print(f"Error writing manifest for mesh {mesh_id}: {str(e)}")
            raise

    def write_fragment_data(self, mesh_id: int, fragments_by_lod: Dict[int, List[Fragment]]):
        """Write the fragment data file following the Neuroglancer precomputed mesh format.
        
        The fragment data file format is described at:
        https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md#multi-resolution-mesh-fragment-data-file-format
        """
        """Write the fragment data file with debug logging."""
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
                    print(f"LOD {lod}: Writing {len(lod_fragments)} fragments")
                    
                    for fragment in sorted(lod_fragments, 
                                        key=lambda f: self._compute_z_order(f.position)):
                        f.write(fragment.draco_bytes)
                        total_fragments += 1
                        total_bytes += len(fragment.draco_bytes)
                
                print(f"Successfully wrote {total_fragments} fragments")
                print(f"Total bytes written: {total_bytes}")
                
            # Verify file exists and has content
            print(f"File exists: {data_path.exists()}")
            print(f"File size: {data_path.stat().st_size} bytes")
            
        except Exception as e:
            print(f"Error writing fragment data for mesh {mesh_id}: {str(e)}")
            raise

    def process_mesh(self, mesh_id: int, mesh: trimesh.Trimesh, num_lods: int = 3):
        """Process a single mesh with additional debug logging."""
        print(f"\nProcessing mesh {mesh_id}")
        print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
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
        
        # Calculate grid origin
        grid_origin = (mesh.vertices.min(axis=0) // self.box_size - 1) * self.box_size
        print(f"Grid origin: {grid_origin}")
        
        # Generate fragments for each LOD
        fragments_by_lod = {}
        for lod, lod_mesh in enumerate(lod_meshes):
            print(f"\nProcessing LOD {lod}")
            print(f"LOD mesh: {len(lod_mesh.vertices)} vertices, {len(lod_mesh.faces)} faces")
            
            # Apply coordinate offset - mesh vertices need to be in the same coordinate system as the voxel data
            # Offset should be 0 instead of the previous (-1,-1,-1) to align correctly with the image/labels
            lod_mesh.vertices = lod_mesh.vertices - grid_origin
            fragments = self.generate_fragments(lod_mesh, lod)
            
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

    def generate_fragments(self, mesh: trimesh.Trimesh, lod: int,
                            enforce_grid_partition: bool = True) -> List[Fragment]:
        """Generate mesh fragments for a given LOD level."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return []
            
        print(f"LOD {lod}: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        current_box_size = self.box_size * (2 ** lod)
        vertices = mesh.vertices
        
        # Calculate fragment bounds - ensure we include a margin around the actual mesh to avoid gaps
        start_fragment = np.maximum(
            vertices.min(axis=0) // current_box_size - 1,
            np.array([0, 0, 0])).astype(int)
        end_fragment = (vertices.max(axis=0) // current_box_size + 1).astype(int)
        
        # Add an overlap factor for better fragment connectivity
        overlap_factor = 0.2 * current_box_size  # 20% overlap between adjacent fragments (increased from 5%)
        
        fragments = []
        fragment_count = 0
        for x in range(start_fragment[0], end_fragment[0]):
            for y in range(start_fragment[1], end_fragment[1]):
                for z in range(start_fragment[2], end_fragment[2]):
                    pos = np.array([x, y, z])
                    
                    # Extract vertices and faces for this fragment with overlap to reduce gaps
                    bounds_min = pos * current_box_size - overlap_factor
                    bounds_max = bounds_min + current_box_size + (2 * overlap_factor)
                    
                    # Use expanded bounds for selecting vertices to ensure overlap between adjacent fragments
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
        """Encode a mesh using Google's Draco encoder with enhanced quality settings."""
        # Normalize vertices to quantization range
        vertices = vertices.copy()
        
        # Improved quantization to prevent vertex snapping at boundaries
        # Add a slight padding to avoid precision issues at the boundaries
        padding = box_size * 0.001  # 0.1% padding
        bounds_min = bounds_min - padding
        box_size = box_size + 2 * padding
        
        vertices -= bounds_min
        vertices /= box_size
        vertices *= (2**self.vertex_quantization_bits - 1)
        vertices = vertices.astype(np.int32)
        
        # Create temporary files for the mesh
        with tempfile.TemporaryDirectory() as temp_dir:
            obj_path = os.path.join(temp_dir, "temp.obj")
            drc_path = os.path.join(temp_dir, "temp.drc")
            
            # Write simple OBJ file
            with open(obj_path, "w") as f:
                for v in vertices:
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
                
            # Run draco_encoder
            cmd = [
                draco_path,
                "-i", obj_path,
                "-o", drc_path,
                "-qp", str(self.vertex_quantization_bits),
                "-qt", "8",  # Higher quality tangents
                "-qn", "8",  # Higher quality normals
                "-qtx", "8",  # Higher quality texture coordinates
                "-cl", "10",  # Maximum compression level
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
        """Enforce grid partitioning for better fragment alignment and reduced gaps."""
        """Enforce 2x2x2 grid partitioning for LOD > 0 meshes."""
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
            
            # Create a slightly expanded version for splitting to avoid gaps
            # We'll use a small expansion factor to ensure overlapping fragments connect properly
            expansion_factor = 0.01  # 1% expansion for better watertight results
            
            # Split mesh along each plane with enhanced capping
            result_mesh = mesh
            for point, normal in zip(mid_points, normals):
                try:
                    # Use a slightly expanded mesh for slicing to ensure watertight results
                    new_mesh = result_mesh.slice_plane(point, normal, cap=True)
                    if new_mesh is not None and len(new_mesh.vertices) > 0:
                        # Apply post-processing to ensure the mesh is clean after slicing
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
        # Skip tiny meshes as they can't be simplified well
        if len(mesh.faces) < 20:
            return mesh
            
        # Handle meshes with potential issues
        try:
            # Make a copy to avoid modifying the original
            mesh_copy = mesh.copy()
            
            # Try to fix any potential issues with the mesh before simplification
            if not mesh_copy.is_watertight:
                mesh_copy.fill_holes()
                
            # Update faces 
            mesh_copy.update_faces(mesh_copy.unique_faces())
            mesh_copy.update_faces(mesh_copy.nondegenerate_faces())
            mesh_copy.fix_normals()
            
            # Set up simplifier
            simplifier = pyfqmr.Simplify()
            simplifier.setMesh(mesh_copy.vertices, mesh_copy.faces)
            
            # Calculate target face count, ensuring we keep at least 10 faces
            target_count = max(int(len(mesh_copy.faces) * target_ratio), 10)
            
            # Simplify with more conservative settings for better quality
            simplifier.simplify_mesh(target_count=target_count, 
                                   aggressiveness=3,
                                   preserve_border=True,
                                   verbose=False)
                                   
            vertices, faces, _ = simplifier.getMesh()
            
            # Check if we got valid results
            if len(vertices) < 3 or len(faces) < 1:
                print(f"Warning: Simplification produced invalid mesh, using original")
                return mesh
                
            # Create a new trimesh with the decimated mesh
            result = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Post-process the simplified mesh
            result.update_faces(result.nondegenerate_faces())
            result.fix_normals()
            
            return result
            
        except Exception as e:
            print(f"Error during mesh decimation: {str(e)}")
            return mesh  # Return original mesh on error

    def generate_lods(self, mesh: trimesh.Trimesh, num_lods: int) -> List[trimesh.Trimesh]:
        """Generate levels of detail for a mesh."""
        lods = [mesh]  # LOD 0 is the highest detail
        current_mesh = mesh
        
        for i in range(1, num_lods):
            # Progressive decimation from previous LOD level for smoother transition
            target_ratio = 0.5  # Each level is about half the detail of previous level
            decimated = self.decimate_mesh(current_mesh, target_ratio)
            
            # Ensure each LOD has at least 10% fewer vertices than the previous
            if len(decimated.vertices) > 0.9 * len(current_mesh.vertices):
                # If decimation didn't reduce enough, force more reduction
                decimated = self.decimate_mesh(current_mesh, 0.3)
                
            # Add diagnostic info    
            print(f"LOD {i}: Original {len(current_mesh.vertices)} vertices → {len(decimated.vertices)} vertices" + 
                 f" ({len(decimated.vertices)/len(current_mesh.vertices):.2f}x)")
                 
            lods.append(decimated)
            current_mesh = decimated  # Use this LOD as base for next level
            
        return lods

def create_ome_zarr_group(root_dir, name, blob_data, labels_data=None, mesh_writer=None):
    """Create an OME-NGFF Zarr store with Neuroglancer-compatible meshes"""
    """
    Create an OME-NGFF Zarr store following the 0.5 specification.
    
    Args:
        root_dir: Root directory for the Zarr store
        name: Name of the Zarr store 
        blob_data: 3D numpy array of the blob data
        labels_data: Optional segmentation labels
    """
    # Create path
    zarr_path = os.path.join(root_dir, f"{name}.zarr")
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)
    
    # We'll create a 3D multiscale image (z, y, x) with a channel dimension
    # First, add a channel dimension to blob data - binary is a single channel
    z, y, x = blob_data.shape
    blob_data_with_channel = blob_data.reshape(1, z, y, x)
    
    # Create the root group with zarr v3 format
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
    
    # Create multiscales metadata
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
                        "scale": [1.0, 1.0, 1.0, 1.0]  # channel, z, y, x
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
    
    # Now we'll create the pyramid levels
    pyramid_levels = 3
    datasets_info = []
    
    # compressor = numcodecs.blosc.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.blosc.SHUFFLE)
    
    # Create original resolution level
    print("Creating resolution levels in zarr store...")
    current_data = blob_data_with_channel.astype(np.uint8)
    
    for i in range(pyramid_levels):
        # Create a group for this resolution level
        level_group = zarr.open(os.path.join(zarr_path, str(i)), mode='w')
        
        # Set array metadata
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
        
        # Downsample for next level by 2x in each spatial dimension
        if i < pyramid_levels - 1:
            # Simple downsample by taking every other voxel
            current_data = current_data[:, ::2, ::2, ::2]
    
    # Update multiscales metadata with dataset info
    multiscales_metadata["ome"]["multiscales"][0]["datasets"] = datasets_info
    
    # Add multiscales metadata to root group
    root.attrs.put(multiscales_metadata)
    
    # If we have labels, add them too
    if labels_data is not None:
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
                                "scale": [1.0, 1.0, 1.0, 1.0]
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
            # Create array for this level
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
            
            # Set array metadata
            label_array.attrs.put({
                "zarr_format": 3,
                "node_type": "array",
                "dimension_names": ["c", "z", "y", "x"]
            })
            
            # Write data
            label_array[:] = current_labels
            
            # Add dataset info
            scale_factor = 2 ** i
            label_datasets_info.append({
                "path": str(i),
                "coordinateTransformations": [{
                    "type": "scale",
                    "scale": [1.0, scale_factor, scale_factor, scale_factor]
                }]
            })
            
            # Downsample for next level if needed
            if i < pyramid_levels - 1:
                current_labels = current_labels[:, ::2, ::2, ::2]
        
        # Update multiscales metadata for labels
        seg_attrs = seg_group.attrs.asdict()
        seg_attrs["attributes"]["ome"]["multiscales"][0]["datasets"] = label_datasets_info
        
        # Add color information for labels
        # Get unique labels and create color mapping
        unique_labels = np.unique(labels_data)
        colors = []
        properties = []
        
        import random
        for label in unique_labels:
            if label == 0:  # background
                rgba = [0, 0, 0, 0]  # transparent
            else:
                # Generate a random color for each object
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                rgba = [r, g, b, 200]  # semi-transparent
            
            colors.append({
                "label-value": int(label),
                "rgba": rgba
            })
            
            # Calculate properties for this label
            if label > 0:
                # Count voxels for this label
                voxel_count = np.sum(labels_data == label)
                properties.append({
                    "label-value": int(label),
                    "voxel-count": int(voxel_count),
                    "name": f"Blob {int(label)}"
                })
        
        # Update image-label metadata with colors and properties
        seg_attrs["attributes"]["ome"]["image-label"]["colors"] = colors
        seg_attrs["attributes"]["ome"]["image-label"]["properties"] = properties
        seg_group.attrs.put(seg_attrs)
    
    # Create a directory for mesh data within the zarr store
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
    
    # If mesh_writer is provided, copy mesh data into the zarr store
    if mesh_writer is not None:
        mesh_source_dir = mesh_writer.output_dir
        print(f"\nCopying mesh files from {mesh_source_dir} to {mesh_dir}")
        
        # First check what files exist in the source directory
        source_files = os.listdir(mesh_source_dir)
        index_files = [f for f in source_files if f.endswith('.index')]
        data_files = [f for f in source_files if f.isdigit() and os.path.isfile(os.path.join(mesh_source_dir, f))]
        
        print(f"Source directory contains {len(index_files)} index files and {len(data_files)} data files")
        print(f"Index files: {index_files[:5]}... (showing first 5)" if len(index_files) > 5 else f"Index files: {index_files}")
        print(f"Data files: {data_files[:5]}... (showing first 5)" if len(data_files) > 5 else f"Data files: {data_files}")
        
        # Manually copy each index and data file to ensure they're correctly placed
        for item in source_files:
            src_path = os.path.join(mesh_source_dir, item)
            dst_path = os.path.join(mesh_dir, item)
            
            if not os.path.isfile(src_path):
                continue  # Skip directories
                
            # Special handling for info file
            if item == "info":
                print(f"Copying info file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                continue
                
            # Copy index files
            if item.endswith('.index'):
                print(f"Copying index file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                # Check if copied successfully
                if os.path.exists(dst_path):
                    print(f"  ✓ Successfully copied {item} ({os.path.getsize(dst_path)} bytes)")
                else:
                    print(f"  ✗ Failed to copy {item}!")
                continue
                
            # Copy data files (numeric filenames)
            if item.isdigit():
                print(f"Copying data file: {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)
                # Check if copied successfully
                if os.path.exists(dst_path):
                    print(f"  ✓ Successfully copied {item} ({os.path.getsize(dst_path)} bytes)")
                else:
                    print(f"  ✗ Failed to copy {item}!")
                continue
                
            # Copy any other files
            print(f"Copying other file: {src_path} -> {dst_path}")
            shutil.copy2(src_path, dst_path)
                
        # Verify all necessary files were copied
        print(f"\nVerifying mesh files in destination directory {mesh_dir}:")
        dest_files = os.listdir(mesh_dir)
        dest_index_files = [f for f in dest_files if f.endswith('.index')]
        dest_data_files = [f for f in dest_files if f.isdigit() and os.path.isfile(os.path.join(mesh_dir, f))]
        
        print(f"Destination directory contains {len(dest_index_files)} index files and {len(dest_data_files)} data files")
        
        # Check for any missing index files
        for idx_file in index_files:
            if idx_file not in dest_files:
                print(f"WARNING: Index file {idx_file} not copied to destination!")
                
        # Check for any missing data files
        for data_file in data_files:
            if data_file not in dest_files:
                print(f"WARNING: Data file {data_file} not copied to destination!")
                
        # Check for paired files
        for idx_file in dest_index_files:
            base_name = idx_file[:-6]  # Remove .index suffix
            if base_name not in dest_data_files:
                print(f"WARNING: Missing data file for index {idx_file}")
                
        for data_file in dest_data_files:
            if f"{data_file}.index" not in dest_index_files:
                print(f"WARNING: Missing index file for data {data_file}")
    
    return zarr_path

def main():
    # Create output directory for the mesh data
    temp_mesh_dir = Path("temp_precomputed")
    mesh_writer = NeuroglancerMeshWriter(
        temp_mesh_dir,
        box_size=32,
        vertex_quantization_bits=16,
        transform=[1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0],
        clean_output=True
    )
    print(f"Creating meshes in: {temp_mesh_dir.absolute()}")
    
    # Generate sample data using binary blobs
    volume_size = 192
    print(f"Generating 3D blobs with size {volume_size}")
    blobs = data.binary_blobs(length=volume_size, n_dim=3, 
                             volume_fraction=0.2,
                             rng=42)
    
    # Apply smoothing to the binary volume
    print("Applying morphological operations and smoothing...")
    blobs = ndimage.binary_closing(blobs, iterations=3)
    blobs = ndimage.binary_opening(blobs, iterations=2)
    blobs = ndimage.gaussian_filter(blobs.astype(float), sigma=1.2) > 0.5
    
    # Label the connected components
    print("Labeling connected components...")
    labels = label(blobs)
    num_labels = labels.max()
    print(f"Found {num_labels} separate blob objects")
    
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
    
    # Create initial meshes
    print("Generating meshes...")
    mesher = Mesher((1.0, 1.0, 1.0))
    mesher.mesh(labels, close=True)
    
    # Process each mesh
    mesh_id = 1
    mesh_object_count = 0
    
    # Get all object IDs
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
        
        # Apply smoothing
        print(f"Smoothing mesh {mesh_id} ({len(trimesh_mesh.vertices)} vertices)...")
        trimesh_mesh = trimesh_mesh.smoothed(lamb=0.75, iterations=10)
        
        # Fix any mesh issues
        if not trimesh_mesh.is_watertight:
            print(f"Fixing non-watertight mesh {mesh_id}...")
            trimesh_mesh.fill_holes()
        
        trimesh_mesh.fix_normals()
        
        # Ensure proper alignment with the image coordinate system
        print(f"Applying proper coordinate alignment for mesh")
        # We need to ensure the mesh coordinates are in the same space as the image/labels data
        # No offset is needed as the vertices are already in the correct coordinate space
        # Explicitly stating this to maintain clarity across renderers
        trimesh_mesh.vertices = trimesh_mesh.vertices.copy()
        
        # Process the mesh with multiple LODs
        mesh_writer.process_mesh(mesh_id, trimesh_mesh, num_lods=3)
        mesh_id += 1
        mesh_object_count += 1
        mesher.erase(obj_id)
    
    # Write mesh info file
    mesh_writer.write_info_file()
    
    print(f"Generated {mesh_object_count} Neuroglancer Precomputed meshes")
    
    # Now create the NGFF zarr group with the meshes included
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
