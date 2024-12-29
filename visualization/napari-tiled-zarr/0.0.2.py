# /// script
# title = "napari tiled zarr"
# description = "Open a multiscale zarr with napari and use multiscale tiled loading"
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.0.2"
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
#     "napari @ git+https://github.com/kephale/napari.git@multiscale-mandelbrot",
#     "pyside6",
#     "typer",
#     "dask",
#     "s3fs",
# ]
# ///

import typer
import zarr
import numpy as np
import dask.array as da
from typing import Optional

from napari.experimental._progressive_loading import add_progressive_loading_image
from napari.experimental._generative_zarr import MandelbulbStore

def mandelbulb_dataset(max_levels=14):
    """Generate a multiscale image of the mandelbulb set for a given number
    of levels/scales. Scale 0 will be the highest resolution.

    Parameters
    ----------
    max_levels: int
        Maximum number of levels (scales) to generate

    Returns
    -------
    List of arrays representing different scales of the mandelbulb
    """
    chunk_size = (32, 32, 32)

    # Initialize the store
    store = zarr.storage.KVStore(
        MandelbulbStore(
            levels=max_levels,
            tilesize=32,
            compressor=None,
            maxiter=255
        )
    )

    # This store implements the 'multiscales' zarr specification
    z_grp = zarr.open(store, mode="r")

    multiscale_img = [z_grp[str(k)] for k in range(max_levels)]

    arrays = []
    for _scale, a in enumerate(multiscale_img):
        da.core.normalize_chunks(
            chunk_size,
            a.shape,
            dtype=np.uint8,
            previous_chunks=None,
        )

        arrays += [a]

    return arrays

def load_ome_zarr(path):
    """Load an OME-zarr dataset with multiple resolution levels.
    
    Parameters
    ----------
    path : str
        Path to the zarr store
        
    Returns
    -------
    list
        List of arrays representing different scales
    """
    import s3fs
    
    if path.startswith('s3://'):
        fs = s3fs.S3FileSystem(anon=True)
        store = zarr.open(fs.get_mapper(path), mode='r')
    else:
        store = zarr.open(path, mode='r')
    
    # Get all scale levels (s0, s1, s2, etc.)
    scales = sorted([k for k in store.keys() if k.startswith('s')])
    if not scales:
        raise ValueError(f"No scale levels found in {path}")
        
    arrays = []
    for scale in scales:
        arrays.append(store[scale])
    
    return arrays

def main(
    zarr_path: Optional[str] = typer.Argument(None, help="Path to the Zarr store."),
    demo: bool = typer.Option(False, "--demo", help="Open the mandelbulb dataset demo")
):
    """
    Open a Zarr dataset as a multiscale image in napari using progressive (tiled) loading.
    If --demo is specified, opens the mandelbulb dataset instead.
    """
    import napari

    # Create a napari viewer with 3D display
    viewer = napari.Viewer(ndisplay=3)
    viewer.axes.visible = True

    if demo:
        # Load the mandelbulb demo dataset
        arrays = mandelbulb_dataset(max_levels=16)
    else:
        if zarr_path is None:
            raise typer.BadParameter("zarr_path is required when not using --demo")
            
        arrays = load_ome_zarr(zarr_path)

    # Add the progressive-loading multiscale image
    add_progressive_loading_image(
        arrays,
        viewer=viewer,
        contrast_limits=[0, 255],
        colormap='twilight_shifted',
        ndisplay=3,
    )

    napari.run()

if __name__ == "__main__":
    typer.run(main)