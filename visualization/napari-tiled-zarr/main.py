# /// script
# title = "napari tiled zarr"
# description = "Open a multiscale zarr with napari and use multiscale tiled loading"
# author = "Kyle Harrington <napari@kyleharrington.com>"
# license = "MIT"
# version = "0.0.3"
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
#     "napari @ git+https://github.com/kephale/napari.git@1e43826432aaebc9e0c7c3c5f6467476859f15de",
#     "pyside6",
#     "typer",
#     "dask",
#     "s3fs",
#     "napari-screen-recorder @ git+https://github.com/kephale/napari-screen-recorder.git",
# ]
# ///

import typer
import zarr
import numpy as np
import dask.array as da
from typing import Optional, Tuple

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
    tuple
        (List of arrays representing different scales of the mandelbulb,
         Scale tuple with value of 1 for each dimension)
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

    # For mandelbulb, use scale of 1 for each dimension
    scale = (1,) * len(arrays[0].shape) if arrays else None

    return arrays, scale

def load_ome_zarr(path):
    """Load an OME-zarr dataset with multiple resolution levels."""
    import s3fs
    import logging
    
    LOGGER = logging.getLogger("zarr_loader")
    LOGGER.setLevel(logging.DEBUG)
    
    if path.startswith('s3://'):
        LOGGER.info(f"Opening S3 path: {path}")
        fs = s3fs.S3FileSystem(anon=True)
        store = zarr.open(fs.get_mapper(path), mode='r')
    else:
        store = zarr.open(path, mode='r')
    
    # Get all scale levels (s0, s1, s2, etc.)
    scales = sorted([k for k in store.keys() if k.startswith('s')])
    if not scales:
        raise ValueError(f"No scale levels found in {path}")
    
    LOGGER.info(f"Found scale levels: {scales}")
    
    arrays = []
    for scale in scales:
        arr = store[scale]
        LOGGER.info(f"Scale {scale} shape: {arr.shape}, chunks: {arr.chunks}")
        arrays.append(arr)
    
    # Extract scale from OME-Zarr metadata
    scale = None
    try:
        LOGGER.info("Store attributes: " + str(store.attrs.asdict()))
        multiscales = store.attrs['multiscales'][0]
        LOGGER.info("Multiscales metadata: " + str(multiscales))
        
        if 'datasets' in multiscales and len(multiscales['datasets']) > 0:
            # Get scale from the highest resolution (first dataset)
            first_dataset = multiscales['datasets'][0]
            LOGGER.info("First dataset metadata: " + str(first_dataset))
            
            if 'coordinateTransformations' in first_dataset:
                for transform in first_dataset['coordinateTransformations']:
                    if transform['type'] == 'scale':
                        scale = tuple(transform['scale'])
                        LOGGER.info(f"Found scale: {scale}")
                        break
            
            # Fallback: try to get scale from pixel_size if available
            if scale is None and 'metadata' in multiscales:
                metadata = multiscales['metadata']
                if 'pixel_size' in metadata:
                    pixel_size = metadata['pixel_size']
                    if all(key in pixel_size for key in ['x', 'y', 'z']):
                        scale = (
                            float(pixel_size['z']), 
                            float(pixel_size['y']), 
                            float(pixel_size['x'])
                        )
                        LOGGER.info(f"Using pixel_size as scale: {scale}")
    except Exception as e:
        LOGGER.warning(f"Error extracting scale from metadata: {str(e)}")
        LOGGER.warning("Defaulting to scale=None")
        scale = None

    if scale is None:
        LOGGER.warning("No scale found in metadata, using default scale=(1,1,1)")
        scale = (1,) * len(arrays[0].shape)
        
    LOGGER.info(f"Final scale: {scale}")
    return arrays, scale

def main(
    zarr_path: Optional[str] = typer.Argument(None, help="Path to the Zarr store."),
    demo: bool = typer.Option(False, "--demo", help="Open the mandelbulb dataset demo"),
    ndisplay: int = typer.Option(3, "--ndisplay", "-n", help="Number of dimensions to display (2 or 3)", min=2, max=3),
    colormap: str = typer.Option("twilight_shifted", "--colormap", "-c", help="Colormap to use for visualization"),
    contrast_limits: Optional[Tuple[float, float]] = typer.Option(
        None, 
        "--contrast-limits", 
        "-l",
        help="Contrast limits as min,max (e.g., '0,255'). Default is [0, 255]"
    )
):
    """Open a Zarr dataset as a multiscale image in napari using progressive loading."""
    import napari
    import logging
    
    LOGGER = logging.getLogger("main")
    LOGGER.setLevel(logging.DEBUG)
    
    # Create a napari viewer with specified dimension display
    viewer = napari.Viewer(ndisplay=ndisplay)
    viewer.axes.visible = True
    
    try:
        if demo:
            arrays, scale = mandelbulb_dataset(max_levels=16)
        else:
            if zarr_path is None:
                raise typer.BadParameter("zarr_path is required when not using --demo")
            
            arrays, scale = load_ome_zarr(zarr_path)
            
        LOGGER.info(f"Loaded arrays with shapes: {[arr.shape for arr in arrays]}")
        LOGGER.info(f"Using scale: {scale}")
        
        if scale is not None and len(scale) != len(arrays[0].shape):
            LOGGER.error(f"Scale dimensions ({len(scale)}) don't match data dimensions ({len(arrays[0].shape)})")
            scale = (1,) * len(arrays[0].shape)
            LOGGER.info(f"Falling back to default scale: {scale}")

        # Set default contrast limits if not provided
        if contrast_limits is None:
            contrast_limits = [0, 255]
        
        LOGGER.info("Adding progressive loading image...")
        viewer = add_progressive_loading_image(
            arrays,
            viewer=viewer,
            contrast_limits=contrast_limits,
            colormap=colormap,
            ndisplay=ndisplay,
            # scale=scale,
        )
        LOGGER.info("Progressive loading image added successfully")
        
        LOGGER.info("Starting napari event loop...")
        napari.run()
        
    except Exception as e:
        LOGGER.error(f"Error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    typer.run(main)