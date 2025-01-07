# /// script
# title = "Napari PNS Browser"
# description = "A napari-based viewer for browsing PNS files with widget controls"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
# keywords = ["napari", "viewer", "visualization", "pns", "structural biology"]
# repository = "https://github.com/kephale/napari-pns-browser"
# documentation = "https://github.com/kephale/napari-pns-browser#readme"
# classifiers = [
#   "Development Status :: 4 - Beta",
#   "Intended Audience :: Science/Research", 
#   "License :: OSI Approved :: MIT License",
#   "Programming Language :: Python :: 3.9",
#   "Topic :: Scientific/Engineering :: Bio-Informatics",
#   "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.9"
# dependencies = [
#   "numpy<2.0",
#   "napari[all]>=0.4.18",
#   "typer",
#   "magicgui>=0.7.0",
#   "qtpy",
#   "vispy>=0.10.0",
#   "napari-screen-recorder @ git+https://github.com/kephale/napari-screen-recorder.git"
# ]
# ///

import os
from pathlib import Path
import typer
import napari
import numpy as np
from magicgui import magicgui
from typing import Optional, List

def read_pns_file(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and faces from a PNS file."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        # Read header (number of vertices and faces)
        n_vertices, n_faces = map(int, f.readline().strip().split())
        
        # Read vertices
        for _ in range(n_vertices):
            x, y, z = map(float, f.readline().strip().split())
            vertices.append([x, y, z])
        
        # Read faces
        for _ in range(n_faces):
            parts = f.readline().strip().split()
            # Skip the first number (should be 3 for triangles)
            faces.append([int(idx)-1 for idx in parts[1:]])  # Convert to 0-based indexing
            
    return np.array(vertices), np.array(faces)

class PNSBrowser:
    def __init__(self, directory: Path):
        self.directory = directory
        self.pns_files = sorted(list(directory.glob("*.pns")))
        self.current_index = 0
        self.viewer = napari.Viewer(ndisplay=3)
        self.surface_layer = None
        
        # Create file selection widget
        @magicgui(
            auto_call=True,
            layout="vertical",
            filename={"choices": [f.stem for f in self.pns_files]},
            next_button={"widget_type": "PushButton", "text": "Next"},
            prev_button={"widget_type": "PushButton", "text": "Previous"},
        )
        def browser_widget(
            filename: str = self.pns_files[0].stem if self.pns_files else "",
            next_button=False,
            prev_button=False,
        ):
            if next_button:
                self.next_file()
            elif prev_button:
                self.prev_file()
            else:
                self.show_file(filename)
        
        self.widget = browser_widget
        self.viewer.window.add_dock_widget(self.widget, area="right")
        
        # Load initial file
        if self.pns_files:
            self.show_file(self.pns_files[0].stem)
    
    def show_file(self, filename: str):
        """Display a specific PNS file."""
        filepath = self.directory / f"{filename}.pns"
        if not filepath.exists():
            return
        
        vertices, faces = read_pns_file(filepath)
        
        # Remove existing surface layer if it exists
        if self.surface_layer is not None:
            self.viewer.layers.remove(self.surface_layer)
        
        # Add new surface layer
        self.surface_layer = self.viewer.add_surface(
            (vertices, faces),
            name=filename,
            opacity=0.7,
            shading='smooth',
            blending='translucent'
        )
        
        # Set text properties carefully to avoid numpy array copy issues
        text_params = {
            'visible': True,
            'size': 12,
            'color': 'white'
        }
        try:
            self.surface_layer.text = text_params
        except Exception as e:
            print(f"Note: Could not set text properties: {e}")
        
        # Update camera if this is the first load
        if self.current_index == 0:
            self.viewer.reset_view()
    
    def next_file(self):
        """Show next PNS file."""
        if not self.pns_files:
            return
            
        self.current_index = (self.current_index + 1) % len(self.pns_files)
        filename = self.pns_files[self.current_index].stem
        self.widget.filename.value = filename
        self.show_file(filename)
    
    def prev_file(self):
        """Show previous PNS file."""
        if not self.pns_files:
            return
            
        self.current_index = (self.current_index - 1) % len(self.pns_files)
        filename = self.pns_files[self.current_index].stem
        self.widget.filename.value = filename
        self.show_file(filename)

app = typer.Typer(help="Browse PNS files using napari")

@app.command()
def browse(
    directory: str = typer.Option(
        ...,
        help="Directory containing PNS files"
    ),
):
    """
    Launch napari viewer to browse PNS files in the specified directory.
    
    The viewer includes widget controls for:
    - Selecting specific files from a dropdown
    - Navigating through files with Next/Previous buttons
    - 3D visualization controls
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        typer.echo(f"Error: Directory {directory} does not exist", err=True)
        raise typer.Exit(code=1)
        
    browser = PNSBrowser(directory_path)
    napari.run()

if __name__ == "__main__":
    app()