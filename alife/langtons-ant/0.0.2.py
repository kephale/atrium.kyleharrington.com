# /// script
# title = "Langton's Ant Simulation"
# description = "Simulates Langton's Ant using napari for visualization."
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.0.2"
# keywords = ["Langton's Ant", "automata", "simulation", "napari", "visualization"]
# repository = "https://github.com/solutions.computational.life/langtons-ant"
# documentation = "https://github.com/solutions.computational.life/langtons-ant#readme"
# homepage = "https://solutions.computational.life"
# classifiers = [
#     "Development Status :: 3 - Alpha",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.10",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.10"
# dependencies = [
#     "napari[all]",
#     "numpy<2",
#     "qtpy",
#     "typer",
# ]
# ///

import typer
import napari
import numpy as np
from qtpy.QtCore import QTimer

app = typer.Typer(help="Langton's Ant Simulation with Napari visualization")

class LangtonsAntSimulation:
    def __init__(self, size=200):
        self.size = size
        self.ant_position = [size // 2, size // 2]
        self.ant_direction = 0  # 0=up, 1=right, 2=down, 3=left
        self.ant_color = [255, 0, 0]  # Red color for the ant

        # Create a grid with two states (0=white, 1=black)
        self.grid = np.zeros((size, size), dtype=np.uint8)

        # Directions: up, right, down, left (dy, dx)
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def update_ant(self):
        # Determine the current cell state
        current_cell_state = self.grid[self.ant_position[0], self.ant_position[1]]

        # Flip the cell state
        self.grid[self.ant_position[0], self.ant_position[1]] = 1 - current_cell_state

        # Turn the ant: right on white (0), left on black (1)
        if current_cell_state == 0:
            self.ant_direction = (self.ant_direction + 1) % 4
        else:
            self.ant_direction = (self.ant_direction - 1) % 4

        # Move the ant forward
        self.ant_position[0] = (self.ant_position[0] + self.directions[self.ant_direction][0]) % self.size
        self.ant_position[1] = (self.ant_position[1] + self.directions[self.ant_direction][1]) % self.size

    def update_layer(self, layer):
        self.update_ant()
        
        # Create an RGB image to highlight the ant
        rgb_grid = np.stack([self.grid * 255] * 3, axis=-1)  # Convert grid to RGB
        rgb_grid[self.ant_position[0], self.ant_position[1]] = self.ant_color  # Color the ant
        layer.data = rgb_grid
        layer.refresh()

@app.command()
def run(
    size: int = typer.Option(200, help="Size of the simulation grid."),
    interval: int = typer.Option(50, help="Update interval in milliseconds.")
):
    """
    Start the Langton's Ant simulation.
    """
    # Initialize the simulation
    simulation = LangtonsAntSimulation(size=size)

    # Initialize Napari viewer
    viewer = napari.Viewer()
    layer = viewer.add_image(
        np.stack([simulation.grid] * 3, axis=-1), 
        name="Langton's Ant"
    )

    # Timer for updating the image layer
    timer = QTimer()
    timer.timeout.connect(lambda: simulation.update_layer(layer))
    timer.start(interval)

    napari.run()

if __name__ == "__main__":
    app()
