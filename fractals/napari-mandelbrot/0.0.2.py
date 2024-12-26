# /// script
# title = "Mandelbrot Set"
# description = "Visualize the Mandelbrot set interactively using napari."
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.0.2"
# keywords = ["mandelbrot", "visualization", "napari", "interactive"]
# classifiers = [
#     "Development Status :: 5 - Production/Stable",
#     "Intended Audience :: Developers",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.11"
# dependencies = [
#     "napari[all]>=0.5.4",
#     "numpy",
#     "typer",
#     "magicgui",
# ]
# ///

import typer
import napari
import numpy as np
from magicgui import magicgui

app = typer.Typer(help="Visualize the Mandelbrot set interactively using napari.")

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Generate the Mandelbrot set for the given range and resolution."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    c = X + 1j * Y 
    z = np.zeros_like(c)
    img = np.zeros(c.shape, dtype=int)

    for i in range(max_iter):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]
        img[mask] += 1

    return img

@app.command()
def visualize(
    xmin: float = typer.Option(-2.0, help="Minimum x-coordinate of the range."),
    xmax: float = typer.Option(1.0, help="Maximum x-coordinate of the range."),
    ymin: float = typer.Option(-1.5, help="Minimum y-coordinate of the range."),
    ymax: float = typer.Option(1.5, help="Maximum y-coordinate of the range."),
    width: int = typer.Option(800, help="Width of the image."),
    height: int = typer.Option(800, help="Height of the image."),
    max_iter: int = typer.Option(100, help="Maximum number of iterations."),
):
    """
    Generate and visualize the Mandelbrot set using napari.
    """
    # Generate the Mandelbrot set
    mandelbrot_image = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

    # Create a Napari viewer and add the Mandelbrot set image
    viewer = napari.Viewer()
    layer = viewer.add_image(mandelbrot_image, name="Mandelbrot Set")

    @magicgui(
        auto_call=True,
        xmin={'label': 'X Min', 'widget_type': 'FloatSpinBox', 'step': 0.1, 'min': -10.0, 'max': 10.0},
        xmax={'label': 'X Max', 'widget_type': 'FloatSpinBox', 'step': 0.1, 'min': -10.0, 'max': 10.0},
        ymin={'label': 'Y Min', 'widget_type': 'FloatSpinBox', 'step': 0.1, 'min': -10.0, 'max': 10.0},
        ymax={'label': 'Y Max', 'widget_type': 'FloatSpinBox', 'step': 0.1, 'min': -10.0, 'max': 10.0},
        mwidth={'label': 'Width', 'widget_type': 'SpinBox', 'step': 10, 'min': 100, 'max': 2000},
        mheight={'label': 'Height', 'widget_type': 'SpinBox', 'step': 10, 'min': 100, 'max': 2000},
        max_iter={'label': 'Max Iterations', 'widget_type': 'SpinBox', 'step': 10, 'min': 10, 'max': 1000}
    )
    def update_mandelbrot(xmin: float, xmax: float, ymin: float, ymax: float, mwidth: int, mheight: int, max_iter: int):
        """Update the Mandelbrot set parameters and refresh the image."""
        mandelbrot_image = mandelbrot(xmin, xmax, ymin, ymax, mwidth, mheight, max_iter)
        layer.data = mandelbrot_image

    # Add the magicgui widget to the Napari viewer
    viewer.window.add_dock_widget(update_mandelbrot, area='right')

    # Set initial values for the widget using the .value attribute correctly
    update_mandelbrot.xmin.value = xmin
    update_mandelbrot.xmax.value = xmax
    update_mandelbrot.ymin.value = ymin
    update_mandelbrot.ymax.value = ymax
    update_mandelbrot.mwidth.value = width
    update_mandelbrot.mheight.value = height
    update_mandelbrot.max_iter.value = max_iter

    # Start Napari event loop
    napari.run()

if __name__ == "__main__":
    app()
