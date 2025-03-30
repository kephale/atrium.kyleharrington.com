# /// script
# title = "3D Blob Meshing (install draco)"
# description = "A Python script to generate and visualize 3D meshes from scikit-image blobs, with export to OBJ, PLY, and Neuroglancer Precomputed formats."
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["mesh", "3D", "visualization", "scikit-image", "zmesh", "neuroglancer"]
# documentation = "https://github.com/seung-lab/zmesh#readme"
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
# "scipy"
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
from skimage.measure import label
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
                 clean_output: bool = False):
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
        self.lod_scale_multiplier = 1.0
        
        # Clean output directory if requested
        if clean_output and self.output_dir.exists():
            import shutil
            print(f"Cleaning existing output directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        
        # Create directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def write_info_file(self):
        """Write the Neuroglancer info JSON file."""
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": self.vertex_quantization_bits,
            "transform": self.transform,
            "lod_scale_multiplier": self.lod_scale_multiplier,
            # Add required fields for viewer compatibility
            "data_type": "uint64",  # Standard for segmentation data
            "num_channels": 1,      # Single channel for mesh data
            "type": "segmentation",  # Important for Neuroglancer to recognize as meshes
            "scales": [
                {
                    "chunk_sizes": [[64, 64, 64]],
                    "encoding": "compressed_segmentation",
                    "key": "64_64_64",
                    "resolution": [1, 1, 1],
                    "size": [256, 256, 256],  # Match the size of our data
                    "voxel_offset": [0, 0, 0]
                }
            ]
        }
        
        with open(self.output_dir / "info", "w") as f:
            json.dump(info, f)
            
        print(f"Created info file: {self.output_dir / 'info'}")
        print(f"Info file content: {json.dumps(info, indent=2)}")

    def write_binary_manifest(self, mesh_id: int, fragments_by_lod: Dict[int, List[Fragment]], 
                            grid_origin: np.ndarray, num_lods: int):
        """Write the binary manifest file with debug logging."""
        manifest_path = self.output_dir / f"{mesh_id}.index"
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
        """Write the fragment data file with debug logging."""
        data_path = self.output_dir / str(mesh_id)
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
            print(f"Successfully processed mesh {mesh_id}")
        except Exception as e:
            print(f"Error processing mesh {mesh_id}: {str(e)}")
            raise

    def _compute_z_order(self, pos: np.ndarray) -> int:
        """Compute Z-order curve index for a 3D position."""
        x, y, z = pos
        answer = 0
        for i in range(21):  # Support up to 21 bits per dimension
            answer |= ((x & (1 << i)) << (2 * i)) | \
                     ((y & (1 << i)) << (2 * i + 1)) | \
                     ((z & (1 << i)) << (2 * i + 2))
        return answer

    def generate_fragments(self, mesh: trimesh.Trimesh, lod: int,
                            enforce_grid_partition: bool = False) -> List[Fragment]:  # Changed default to False for better LOD compatibility
        """Generate mesh fragments for a given LOD level."""
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return []
            
        print(f"LOD {lod}: Processing mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        current_box_size = self.box_size * (2 ** lod)
        vertices = mesh.vertices
        
        # Calculate fragment bounds
        start_fragment = np.maximum(
            vertices.min(axis=0) // current_box_size - 1,
            np.array([0, 0, 0])).astype(int)
        end_fragment = (vertices.max(axis=0) // current_box_size + 1).astype(int)
        
        fragments = []
        fragment_count = 0
        for x in range(start_fragment[0], end_fragment[0]):
            for y in range(start_fragment[1], end_fragment[1]):
                for z in range(start_fragment[2], end_fragment[2]):
                    pos = np.array([x, y, z])
                    
                    # Extract vertices and faces for this fragment
                    bounds_min = pos * current_box_size
                    bounds_max = bounds_min + current_box_size
                    
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
                        
                        # Debug message to help diagnose LOD issues
                        print(f"LOD {lod}: Generated fragment at position {pos} with {len(fragment_mesh.vertices)} vertices, {len(fragment_mesh.faces)} faces, {len(draco_bytes)} bytes")
                        
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
        """Encode a mesh using Google's Draco encoder."""
        # Normalize vertices to quantization range
        vertices = vertices.copy()
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
            
            # Run draco_encoder
            cmd = [
                "/opt/homebrew/bin/draco_encoder",
                "-i", obj_path,
                "-o", drc_path,
                "-qp", str(self.vertex_quantization_bits),
                "-qt", "2",
                "-qn", "2",
                "-qtx", "2",
                "-cl", "10",
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
            
            # Split mesh along each plane
            result_mesh = mesh
            for point, normal in zip(mid_points, normals):
                try:
                    new_mesh = result_mesh.slice_plane(point, normal, cap=True)
                    if new_mesh is not None and len(new_mesh.vertices) > 0:
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
                
            # Use updated methods instead of deprecated ones
            mesh_copy.update_faces(mesh_copy.unique_faces())
            mesh_copy.update_faces(mesh_copy.nondegenerate_faces())
            mesh_copy.fix_normals()
            
            # Set up simplifier
            simplifier = pyfqmr.Simplify()
            simplifier.setMesh(mesh_copy.vertices, mesh_copy.faces)
            
            # Calculate target face count, ensuring we keep at least 10 faces
            target_count = max(int(len(mesh_copy.faces) * target_ratio), 10)
            
            # Simplify with more conservative settings for better quality
            # Remove the preserve_normal parameter which isn't supported
            simplifier.simplify_mesh(target_count=target_count, 
                                   aggressiveness=3,  # Less aggressive (was 5)
                                   preserve_border=True,
                                   verbose=False)
                                   
            vertices, faces, _ = simplifier.getMesh()
            
            # Check if we got valid results
            if len(vertices) < 3 or len(faces) < 1:
                print(f"Warning: Simplification produced invalid mesh, using original")
                return mesh
                
            # Create a new trimesh with the decimated mesh
            result = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Post-process the simplified mesh using updated methods
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

def main():
    # Create output directory and writer
    output_dir = Path("precomputed")
    writer = NeuroglancerMeshWriter(
        output_dir,
        box_size=32,  # Smaller box size for better detail
        vertex_quantization_bits=16,  # Higher vertex precision
        transform=[4.0, 0, 0, 0, 0, 4.0, 0, 0, 0, 0, 4.0, 0],  # Scale up meshes 
        clean_output=True  # Clean existing output to avoid issues
    )
    print(f"Creating meshes in: {output_dir.absolute()}")
    
    # Generate sample data using binary blobs
    # Increase volume size for better resolution
    volume_size = 192  # Larger size for more detailed blobs
    print(f"Generating 3D blobs with size {volume_size}")
    blobs = data.binary_blobs(length=volume_size, n_dim=3, 
                             volume_fraction=0.2,  # Less crowded
                             rng=42)
    
    # Apply smoothing to the binary volume to reduce voxelization artifacts
    from scipy import ndimage
    print("Applying morphological operations and smoothing...")
    # More aggressive morphological operations to separate blobs better
    blobs = ndimage.binary_closing(blobs, iterations=3)  # Increased from 2
    blobs = ndimage.binary_opening(blobs, iterations=2)  # Increased from 1
    # Apply a stronger Gaussian filter to smooth edges better
    blobs = ndimage.gaussian_filter(blobs.astype(float), sigma=1.2) > 0.5  # Increased from 0.8
    
    # Label the connected components
    print("Labeling connected components...")
    labels = label(blobs)
    num_labels = labels.max()
    print(f"Found {num_labels} separate blob objects")
    
    # Filter out very small blobs that might cause rendering issues
    from scipy import ndimage
    sizes = ndimage.sum(blobs, labels, range(1, num_labels+1))
    mask = sizes > 100  # Only keep blobs with more than 100 voxels
    filtered_labels = np.zeros_like(labels)
    for i, keep in enumerate(mask, 1):
        if keep:
            filtered_labels[labels == i] = i
    
    num_filtered_labels = len(np.unique(filtered_labels)) - 1  # Subtract 1 for background
    print(f"After filtering small blobs: {num_filtered_labels} objects remain")
    labels = filtered_labels
    
    # Create initial meshes
    # Use isotropic voxel scaling for better quality
    print("Generating meshes...")
    mesher = Mesher((1.0, 1.0, 1.0))
    mesher.mesh(labels, close=True)
    
    # Process each mesh - map to sequential IDs starting from 1
    # This ensures we have a clean sequence of mesh IDs that can be found by the viewer
    mesh_id = 1
    mesh_object_count = 0
    
    # Get all object IDs first to count them
    object_ids = list(mesher.ids())
    print(f"Processing {len(object_ids)} meshes...")
    
    for obj_id in object_ids:
        mesh = mesher.get(obj_id, normals=True)
        
        # Skip meshes with too few vertices or faces
        if len(mesh.vertices) < 10 or len(mesh.faces) < 10:
            print(f"Skipping small mesh {obj_id} with only {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            mesher.erase(obj_id)
            continue
            
        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        
        # Apply more iterations of Laplacian smoothing for smoother blobs
        print(f"Smoothing mesh {mesh_id} ({len(trimesh_mesh.vertices)} vertices)...")
        trimesh_mesh = trimesh_mesh.smoothed(lamb=0.75, iterations=10)  # More aggressive smoothing
        
        # Fix any mesh issues
        if not trimesh_mesh.is_watertight:
            print(f"Fixing non-watertight mesh {mesh_id}...")
            trimesh_mesh.fill_holes()
            
        # Ensure the mesh has the correct winding and normals
        trimesh_mesh.fix_normals()
        
        # Process the mesh with multiple LODs
        writer.process_mesh(mesh_id, trimesh_mesh, num_lods=3)
        mesh_id += 1
        mesh_object_count += 1
        mesher.erase(obj_id)
    
    # Write info file
    writer.write_info_file()
    
    print(f"Generated {mesh_object_count} Neuroglancer Precomputed meshes in {output_dir}")
    print("\nTo view the meshes, run:")
    print("uv run meshes/visualize-ng-precomputed/view_in_ng.py --mesh-dir precomputed")

if __name__ == "__main__":
    main()