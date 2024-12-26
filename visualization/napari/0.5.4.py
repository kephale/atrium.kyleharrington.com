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
#     "napari[all]==0.5.4",
#     "typer",
# ]
# ///

import typer
import napari
import os
from typing import Optional, List

app = typer.Typer(help="A simple CLI to launch the napari viewer with optional preloaded files and layers.")


@app.command()
def launch(
    files: Optional[List[str]] = typer.Option(None, help="Paths to files to load into the viewer."),
    display_mode: str = typer.Option("2d", help="Set viewer mode: '2d' or '3d'."),
):
    """
    Launch the napari viewer with optional files.
    """
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
