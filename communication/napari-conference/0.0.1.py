# /// script
# title = "napari-webcam-filters"
# description = "A Python script to launch napari with real-time webcam filters for video conferencing. Follow pyvirtualcam install instructions first."
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["napari", "webcam", "filters", "video-conferencing", "virtual-camera"]
# repository = "https://github.com/kephale/napari-webcam-filters"
# documentation = "https://github.com/kephale/napari-webcam-filters#readme"
# homepage = "https://napari.org"
# classifiers = [
#     "Development Status :: 3 - Alpha",
#     "Intended Audience :: End Users/Desktop",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.8",
#     "Topic :: Multimedia :: Video",
#     "Topic :: Scientific/Engineering :: Image Processing",
# ]
# requires-python = ">=3.8"
# dependencies = [
#     "napari[all]",
#     "opencv-python",
#     "pyvirtualcam",
#     "numpy",
#     "magicgui",
#     "napari-conference @ git+https://github.com/kephale/napari-conference"
# ]
# ///

import sys
import site

import typer
import napari
from dataclasses import dataclass
from typing import Optional
import logging

from napari_conference import conference_widget, WebcamProcessor, ImageFilters, WebcamState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="A CLI to launch napari with webcam filters for video conferencing.")

@dataclass
class LaunchConfig:
    """Configuration for launching the webcam application."""
    window_width: int = 800
    window_height: int = 600
    layer_name: str = "Webcam Feed"
    default_filter: str = "None"
    default_trails: float = 0.1

@app.command()
def launch(
    width: int = typer.Option(800, help="Window width"),
    height: int = typer.Option(600, help="Window height"),
    layer_name: str = typer.Option("Webcam Feed", help="Name of the webcam layer"),
    filter_name: str = typer.Option("None", help="Initial filter to apply"),
    trails: float = typer.Option(0.1, help="Motion trails parameter (0-1)")
):
    """
    Launch napari with the webcam filters plugin.
    
    The application provides real-time filters and effects for video conferencing,
    including virtual camera output capabilities.
    """
    config = LaunchConfig(
        window_width=width,
        window_height=height,
        layer_name=layer_name,
        default_filter=filter_name,
        default_trails=trails
    )
    
    logger.info(f"Launching napari with window size {config.window_width}x{config.window_height}")
    
    # Initialize viewer and state
    viewer = napari.Viewer()
    viewer.window.resize(config.window_width, config.window_height)
    state = WebcamState()
    
    # Create and add the widget
    widget = conference_widget()
    widget.layer_name.value = config.layer_name
    widget.dropdown.value = config.default_filter
    widget.trails_param.value = config.default_trails
    
    viewer.window.add_dock_widget(widget, name="Webcam Filters")
    
    logger.info("Starting napari event loop")
    napari.run()

def main():
    """Entry point for the application."""
    try:
        app()
    except Exception as e:
        logger.error(f"Error running application: {e}")
        raise

if __name__ == "__main__":
    main()