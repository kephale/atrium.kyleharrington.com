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
import os
import numpy as np
from qtpy.QtWidgets import QMainWindow, QApplication, QHBoxLayout, QWidget
from typing import Optional, List

app = typer.Typer(help="A simple CLI to launch the napari viewer with optional preloaded files and layers.")


@app.command()
def launch(
    files: Optional[List[str]] = typer.Option(None, help="Paths to files to load into the viewer."),
    display_mode: str = typer.Option("2d", help="Set viewer mode: '2d' or '3d'."),
    show_dual: bool = typer.Option(False, help="Display two sample images side by side in different widgets."),
):
    """
    Launch the napari viewer with optional files or dual viewer mode.
    """
    if show_dual:
        # Create sample image data
        image1 = np.random.random((512, 512))
        image2 = np.random.random((512, 512))

        # Start the QApplication (if not already running)
        app = QApplication.instance()
        if not app:
            app = QApplication([])

        # Create a main window with two Napari viewers
        main_window = QMainWindow()
        central_widget = QWidget()
        main_window.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Create two Napari viewers
        viewer1 = napari.Viewer(ndisplay=3 if display_mode == "3d" else 2)
        viewer2 = napari.Viewer(ndisplay=3 if display_mode == "3d" else 2)

        # Add the viewers to the layout
        layout.addWidget(viewer1.window.qt_viewer)
        layout.addWidget(viewer2.window.qt_viewer)

        # Add images to the viewers
        viewer1.add_image(image1, name="Image 1")
        viewer2.add_image(image2, name="Image 2")

        # Show the main window
        main_window.show()

        typer.echo("Launching dual Napari viewers with sample images...")
        napari.run()

    else:
        # Default single viewer behavior
        viewer = napari.Viewer(ndisplay=3 if display_mode == "3d" else 2)

        if files:
            for file_path in files:
                if not os.path.exists(file_path):
                    typer.echo(f"Error: File {file_path} does not exist.", err=True)
                    continue

                try:
                    viewer.open(file_path)
                    typer.echo(f"Loaded file: {file_path}")
                except Exception as e:
                    typer.echo(f"Error loading {file_path}: {e}", err=True)

        typer.echo("Launching napari viewer...")
        napari.run()


if __name__ == "__main__":
    app()
