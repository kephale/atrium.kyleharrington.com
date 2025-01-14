# /// script
# title = "3D Blob Meshing"
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
# ]
# ///
from zmesh import Mesher
import numpy as np
from skimage import data
from skimage.measure import label
import os

# Create output directories
os.makedirs('obj', exist_ok=True)
os.makedirs('ply', exist_ok=True)
os.makedirs('precomputed', exist_ok=True)

# Load the 3D blobs data from scikit-image
# This creates a 3D binary image of spherical blobs
blobs = data.binary_blobs(length=256, n_dim=3, volume_fraction=0.2)

# Run connected components to get unique labels for each blob
labels = label(blobs)

# Create mesher with isotropic resolution (1,1,1)
# since binary_blobs creates isotropic data
mesher = Mesher((1,1,1))

# Initial marching cubes pass
mesher.mesh(labels, close=True) # close=True since we want complete spheres

meshes = []
for obj_id in mesher.ids():
    mesh = mesher.get(
        obj_id,
        normals=True, # Enable normals for better visualization
        reduction_factor=50, # Less aggressive reduction for spherical shapes
        max_error=2, # Smaller error tolerance to maintain spherical shape
        voxel_centered=True, # Center vertices in voxels for smoother appearance
    )
    meshes.append((obj_id, mesh))
    mesher.erase(obj_id)

mesher.clear()

# Save meshes in different formats
for obj_id, mesh in meshes:
    # Save as OBJ
    with open(f'obj/blob_{obj_id}.obj', 'wb') as f:
        f.write(mesh.to_obj())
    
    # Save as PLY
    with open(f'ply/blob_{obj_id}.ply', 'wb') as f:
        f.write(mesh.to_ply())
    
    # Save as Neuroglancer Precomputed format
    # Neuroglancer expects specific filename format: "{segment_id}:0"
    with open(f'precomputed/{obj_id}:0', 'wb') as f:
        f.write(mesh.to_precomputed())

print(f"Generated {len(meshes)} blob meshes")

# Basic mesh statistics
for obj_id, mesh in meshes:
    print(f"\nBlob {obj_id}:")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Has normals: {mesh.normals is not None}")