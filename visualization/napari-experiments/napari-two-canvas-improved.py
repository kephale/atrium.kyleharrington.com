# /// script
# title = "napari"
# description = "A simple Python script to launch napari."
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.5.4"
# keywords = ["napari", "viewer", "visualization"]
# repository = "https://github.com/napari/napari"
# documentation = "https://github.com/napari/napari#readme"
# homepage = "https://napari.org"
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Developers",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.12"
# dependencies = [
#     "napari[all]>=0.5.5",
#     "typer",
# ]
# ///


import typer
import napari
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMainWindow

app = typer.Typer(help="A simple CLI to launch the napari viewer with optional preloaded files and layers.")


@app.command()
def launch_with_multiple_canvases():
    """
    Launch the Napari viewer with multiple canvases integrated into a single window.
    """
    # Create sample image data
    image1 = np.random.random((512, 512))
    image2 = np.random.random((512, 512))

    # Start the main Napari viewer
    main_viewer = napari.Viewer()
    main_viewer.add_image(image1, name="Image 1")

    # Access the main viewer's Qt main window
    main_window = main_viewer.window._qt_window

    # Create a container widget with a horizontal layout
    container = QWidget()
    layout = QHBoxLayout(container)

    # Get the main viewer's canvas and add it to the container
    main_canvas = main_window.centralWidget()
    layout.addWidget(main_canvas)

    # Create a secondary viewer and add its canvas to the container
    secondary_viewer = napari.Viewer(show=False)
    secondary_viewer.add_image(image2, name="Image 2")
    secondary_canvas = secondary_viewer.window.qt_viewer
    layout.addWidget(secondary_canvas)

    # Set the container as the central widget for the main viewer
    main_window.setCentralWidget(container)

    # Show the main Napari viewer window
    napari.run()


if __name__ == "__main__":
    app()
