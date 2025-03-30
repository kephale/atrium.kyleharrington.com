# /// script
# title = "Adaptive LOD Mesh Viewer"
# description = "A Python script that adaptively switches between LOD levels based on camera distance"
# author = "Kyle Harrington"
# license = "MIT"
# version = "0.1.0"
# keywords = ["mesh", "3D", "visualization", "napari", "neuroglancer", "LOD", "adaptive"]
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
import numpy as np
import trimesh
import napari
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import the PrecomputedMeshLoader class
from view_in_napari import PrecomputedMeshLoader

class AdaptiveLODViewer:
    """
    A viewer that adaptively switches between different LOD levels based on camera distance.
    
    This viewer loads all LOD levels for each mesh but only shows the appropriate
    level based on the camera distance to the mesh.
    """
    
    def __init__(self, mesh_dir: str, num_meshes: int = 3, debug: bool = False):
        self.mesh_dir = Path(mesh_dir)
        self.num_meshes = num_meshes
        self.debug = debug
        
        # Initialize the mesh loader
        self.mesh_loader = PrecomputedMeshLoader(mesh_dir, debug=debug)
        
        # Initialize structures for storing mesh data and layers
        self.meshes = {}  # Dict to store loaded meshes by mesh_id and LOD
        self.layers = {}  # Dict to store napari layers by mesh_id and LOD
        self.visible_lods = {}  # Currently visible LOD for each mesh
        
        # Create the napari viewer
        self.viewer = napari.Viewer(ndisplay=3)
        
        # LOD distance thresholds - adjust these to control when LOD levels switch
        self.lod_thresholds = [50.0, 120.0]  # Distances at which to switch LOD levels
        
        # Set up camera change event callback
        self.viewer.camera.events.connect(self.on_camera_change)
        
    def load_meshes(self):
        """Load meshes from the precomputed directory."""
        # Validate the directory and get valid mesh IDs
        valid, message, valid_meshes = self.mesh_loader.validate_directory()
        
        if not valid:
            print(f"Error: {message}")
            print("\nTip: If using 'minimal-ng-precomputed', ensure you've generated the meshes first:")
            print("     uv run meshes/minimal-ng-precomputed/0.0.1.py")
            return False
            
        print(f"\nFound {len(valid_meshes)} valid meshes in {self.mesh_dir}")
        
        # Decide how many meshes to load
        if self.num_meshes > 0 and self.num_meshes < len(valid_meshes):
            load_meshes = valid_meshes[:self.num_meshes]
            print(f"Loading first {self.num_meshes} of {len(valid_meshes)} meshes")
            print(f"Loading mesh IDs: {load_meshes}")
        else:
            load_meshes = valid_meshes
            print(f"Loading all {len(valid_meshes)} meshes")
        
        # Process each mesh
        for mesh_idx, mesh_id in enumerate(load_meshes):
            try:
                # Read the manifest once
                manifest = self.mesh_loader.read_manifest(mesh_id)
                if not manifest or "num_lods" not in manifest:
                    print(f"Could not load manifest for mesh {mesh_id}")
                    continue
                    
                num_lods = manifest["num_lods"]
                print(f"Mesh {mesh_id} has {num_lods} LOD levels")
                
                # Initialize storage for this mesh
                self.meshes[mesh_id] = {}
                self.layers[mesh_id] = {}
                
                # Load all LOD levels for this mesh
                max_lod_loaded = -1
                
                for lod in range(num_lods):
                    # Skip LOD if it has no fragments
                    if lod not in manifest["fragments"] or manifest["fragments_per_lod"][lod] == 0:
                        print(f"No fragments found for mesh {mesh_id} at LOD {lod}")
                        continue
                    
                    # Load the mesh at this LOD
                    mesh = self.mesh_loader.load_lod_mesh(mesh_id, lod)
                    if mesh is None:
                        print(f"Could not load mesh {mesh_id} at LOD {lod}")
                        continue
                    
                    # Store the mesh
                    self.meshes[mesh_id][lod] = mesh
                    max_lod_loaded = max(max_lod_loaded, lod)
                    
                    print(f"Loaded mesh {mesh_id} LOD {lod} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
                
                # If we successfully loaded at least one LOD, add layers for all LODs
                if max_lod_loaded >= 0:
                    self.add_mesh_layers(mesh_id)
                    # Initially show the highest LOD (lowest detail)
                    self.visible_lods[mesh_id] = max_lod_loaded
                    self.update_lod_visibility(mesh_id, max_lod_loaded)
                
            except Exception as e:
                print(f"Error processing mesh {mesh_id}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
        
        if not self.layers:
            print("No meshes could be loaded. Try using --debug for more information.")
            return False
            
        # Reset view to show all objects
        self.viewer.reset_view()
        return True
        
    def add_mesh_layers(self, mesh_id: int):
        """Add napari layers for all LOD levels of a mesh."""
        # Define colors for different LOD levels
        lod_colors = {
            0: [1.0, 0.4, 0.4],  # Red for LOD 0 (highest detail)
            1: [0.4, 1.0, 0.4],  # Green for LOD 1 (medium detail)
            2: [0.4, 0.4, 1.0],  # Blue for LOD 2 (lowest detail)
        }
        
        # For each LOD level of this mesh
        for lod, mesh in self.meshes[mesh_id].items():
            # Create a unique name for this layer
            layer_name = f"Mesh {mesh_id} - LOD {lod}"
            
            # Choose color based on LOD level
            color = lod_colors.get(lod, [0.7, 0.7, 0.7])  # Default to gray for unsupported LODs
            
            try:
                # Add the surface layer
                surface = self.viewer.add_surface(
                    data=(mesh.vertices, mesh.faces),
                    name=layer_name,
                    opacity=0.9,
                    blending='translucent',
                    colormap='gray',
                    contrast_limits=[0, 1],
                    visible=False  # Initially hide all layers
                )
                
                # Set the color for this layer
                surface.color = color
                
                # Store the layer
                self.layers[mesh_id][lod] = surface
                
            except Exception as e:
                print(f"Error adding surface for mesh {mesh_id} LOD {lod}: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
    
    def update_lod_visibility(self, mesh_id: int, active_lod: int):
        """Update which LOD level is visible for a specific mesh."""
        # Hide all LOD levels for this mesh
        for lod, layer in self.layers[mesh_id].items():
            layer.visible = (lod == active_lod)
        
        # Update the visible LOD tracker
        self.visible_lods[mesh_id] = active_lod
        
    def on_camera_change(self, event):
        """Called when the camera position or orientation changes."""
        # Get current camera position
        camera_position = np.array(self.viewer.camera.center)
        
        # For each mesh, determine the appropriate LOD level based on distance
        for mesh_id in self.meshes:
            # Calculate approximate distance to the mesh
            # This could be improved with a more accurate distance calculation to mesh surface
            mesh_position = self.get_mesh_center(mesh_id)
            distance = np.linalg.norm(camera_position - mesh_position)
            
            # Determine appropriate LOD level based on distance
            appropriate_lod = self.get_appropriate_lod(distance)
            
            # If the appropriate LOD is different from the currently visible one, update
            current_lod = self.visible_lods.get(mesh_id)
            if appropriate_lod != current_lod and appropriate_lod in self.meshes[mesh_id]:
                print(f"Switching mesh {mesh_id} from LOD {current_lod} to LOD {appropriate_lod} (distance: {distance:.1f})")
                self.update_lod_visibility(mesh_id, appropriate_lod)
    
    def get_mesh_center(self, mesh_id: int) -> np.ndarray:
        """Calculate the center of a mesh (using any available LOD level)."""
        # Use the first available LOD level to calculate center
        for lod in sorted(self.meshes[mesh_id].keys()):  # Prefer higher detail LODs
            mesh = self.meshes[mesh_id][lod]
            return mesh.vertices.mean(axis=0)
        
        # Fallback if no mesh data found
        return np.array([0, 0, 0])
    
    def get_appropriate_lod(self, distance: float) -> int:
        """Determine appropriate LOD level based on distance."""
        if distance < self.lod_thresholds[0]:
            return 0  # Closest = highest detail
        elif distance < self.lod_thresholds[1]:
            return 1  # Medium distance = medium detail
        else:
            return 2  # Furthest = lowest detail
    
    def start(self):
        """Start the napari viewer."""
        print("\nAdaptive LOD viewer launched!")
        print("The viewer will automatically switch between LOD levels based on camera distance:")
        print(f"- Distance < {self.lod_thresholds[0]:.1f}: LOD 0 (highest detail)")
        print(f"- Distance {self.lod_thresholds[0]:.1f} to {self.lod_thresholds[1]:.1f}: LOD 1 (medium detail)")
        print(f"- Distance > {self.lod_thresholds[1]:.1f}: LOD 2 (lowest detail)")
        print("\nControls:")
        print("- Right-click and drag to rotate")
        print("- Middle-click and drag to pan")
        print("- Scroll to zoom")
        
        # Start the napari event loop
        napari.run()

def main():
    parser = argparse.ArgumentParser(description="Adaptive LOD viewer that switches detail based on camera distance")
    parser.add_argument("--mesh-dir", type=str, required=True,
                      help="Directory containing precomputed mesh data")
    parser.add_argument("--num-meshes", type=int, default=3,
                      help="Number of meshes to load (default: 3)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Create and start the adaptive LOD viewer
    viewer = AdaptiveLODViewer(
        mesh_dir=args.mesh_dir,
        num_meshes=args.num_meshes,
        debug=args.debug
    )
    
    # Load meshes
    if viewer.load_meshes():
        # Start the viewer if meshes were loaded successfully
        viewer.start()
    else:
        print("Error loading meshes. Exiting.")
        sys.exit(1)

if __name__ == '__main__':
    main()
