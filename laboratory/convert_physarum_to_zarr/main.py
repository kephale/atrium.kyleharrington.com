# /// script
# title = "Physarum Time-Series to Zarr"
# description = "Extract and compress physarum petri dish images from time-series into efficient Zarr format"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["physarum", "time-series", "zarr", "image-processing", "compression"]
# classifiers = [
#     "Development Status :: 4 - Beta", 
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Scientific/Engineering :: Image Processing"
# ]
# requires-python = ">=3.11"
# dependencies = [
#     "zarr>=2.15.0",
#     "numpy>=1.24.0",
#     "pillow>=10.0.0",
#     "tqdm>=4.65.0",
#     "typer>=0.9.0"
# ]
# ///

import os
import re
import zarr
import numpy as np
from PIL import Image
from datetime import datetime
import glob
from tqdm import tqdm
import typer
from pathlib import Path

app = typer.Typer()

def extract_timestamp(filename: str) -> datetime | None:
    """Extract timestamp from filename pattern physarum_YYYYMMDD_HHMMSS.png"""
    pattern = r'physarum_(\d{8}_\d{6})\.png'
    match = re.search(pattern, filename)
    if match:
        timestamp_str = match.group(1)
        # Convert to datetime object
        return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    return None

def detect_dish_layout(image: Image.Image) -> tuple[int, int]:
    """
    Analyze image to determine dish layout (2x3 or 3x2)
    This should be customized based on your specific setup
    """
    # For now, let's assume 2x3 (2 rows, 3 columns) based on the aspect ratio
    # You may need to adjust this based on visual inspection
    height, width = image.size
    if height > width:  # Taller than wide
        return 2, 3  # 2 rows, 3 columns
    else:
        return 3, 2  # 3 rows, 2 columns

def extract_dishes(image: Image.Image, rows: int = 2, cols: int = 3) -> list[np.ndarray]:
    """Extract individual dishes from the composite image"""
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    dish_height = height // rows
    dish_width = width // cols
    
    dishes = []
    for row in range(rows):
        for col in range(cols):
            top = row * dish_height
            left = col * dish_width
            bottom = top + dish_height
            right = left + dish_width
            
            dish = img_array[top:bottom, left:right]
            dishes.append(dish)
    
    return dishes

def create_zarr_dataset(
    input_dir: Path, 
    output_path: Path, 
    chunk_size: tuple[int, int, int, int, int] = (10, 1, 400, 400, 3)
) -> zarr.Group:
    """Create Zarr dataset from physarum images"""
    
    # Get all png files
    files = sorted(glob.glob(str(input_dir / '*.png')))
    typer.echo(f"Found {len(files)} files")
    
    # Extract timestamps
    timestamps = []
    for file in files:
        ts = extract_timestamp(os.path.basename(file))
        if ts:
            timestamps.append(ts)
    
    # Sort files by timestamp
    file_timestamp_pairs = list(zip(files, timestamps))
    file_timestamp_pairs.sort(key=lambda x: x[1])
    files, timestamps = zip(*file_timestamp_pairs)
    
    # Determine dish layout from first image
    first_image = Image.open(files[0])
    rows, cols = detect_dish_layout(first_image)
    typer.echo(f"Detected dish layout: {rows}x{cols}")
    
    # Extract sample to get dimensions
    sample_dishes = extract_dishes(first_image, rows, cols)
    dish_shape = sample_dishes[0].shape
    typer.echo(f"Each dish shape: {dish_shape}")
    
    # Create Zarr dataset
    total_dishes = rows * cols
    zarr_shape = (len(files), total_dishes, dish_shape[0], dish_shape[1], dish_shape[2])
    
    typer.echo(f"Creating Zarr dataset with shape: {zarr_shape}")
    typer.echo(f"Chunk size: {chunk_size}")
    
    # Create the main dataset
    zarr_store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=zarr_store)
    
    # Create the main data array
    data = root.create_dataset(
        'images',
        shape=zarr_shape,
        chunks=chunk_size,
        dtype=np.uint8,
        compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    )
    
    # Add metadata
    data.attrs['description'] = 'Physarum polycephalum time-series imaging data'
    data.attrs['experiment'] = 'experiment_010'
    data.attrs['dimensions'] = ['time', 'dish', 'height', 'width', 'channels']
    data.attrs['dish_layout'] = f'{rows}x{cols}'
    data.attrs['created_at'] = datetime.now().isoformat()
    
    # Create timestamps array
    timestamp_array = root.create_dataset(
        'timestamps',
        data=[ts.isoformat() for ts in timestamps],
        dtype='<U25'
    )
    timestamp_array.attrs['description'] = 'Timestamp for each image in ISO format'
    
    # Process all files
    typer.echo("Processing images...")
    for i, file in enumerate(tqdm(files)):
        try:
            img = Image.open(file)
            dishes = extract_dishes(img, rows, cols)
            
            for j, dish in enumerate(dishes):
                data[i, j] = dish
                
        except Exception as e:
            typer.echo(f"Error processing {file}: {e}", err=True)
    
    # Add dish metadata
    dish_metadata = root.create_group('dish_metadata')
    for i in range(total_dishes):
        row = i // cols
        col = i % cols
        dish_meta = dish_metadata.create_group(f'dish_{i}')
        dish_meta.attrs['row'] = row
        dish_meta.attrs['column'] = col
        dish_meta.attrs['position'] = f'row_{row}_col_{col}'
    
    typer.echo(f"Dataset created successfully at: {output_path}")
    typer.echo(f"Total size on disk: {get_dir_size(output_path) / (1024**3):.2f} GB")
    
    return root

def get_dir_size(path: Path) -> int:
    """Get size of directory in bytes"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

def verify_zarr_dataset(zarr_path: Path) -> bool:
    """Verify the created Zarr dataset"""
    root = zarr.open(str(zarr_path), mode='r')
    
    typer.echo("\nDataset Information:")
    typer.echo(f"Shape: {root.images.shape}")
    typer.echo(f"Chunk shape: {root.images.chunks}")
    typer.echo(f"Compression: {root.images.compressor}")
    typer.echo(f"Size on disk: {get_dir_size(zarr_path) / (1024**2):.1f} MB")
    
    # Print metadata
    typer.echo("\nMetadata:")
    for key, value in root.images.attrs.items():
        typer.echo(f"  {key}: {value}")
    
    # Sample access test
    typer.echo("\nSample access test:")
    sample_timestamp = root.timestamps[0]
    typer.echo(f"First timestamp: {sample_timestamp}")
    
    return True

@app.command()
def convert(
    input_dir: str = typer.Argument(..., help="Directory containing physarum PNG images"),
    output_path: str = typer.Argument(..., help="Output path for Zarr dataset"),
    chunk_size: str = typer.Option("10,1,400,400,3", "--chunk-size", "-c", help="Chunk size as comma-separated values")
):
    """Convert physarum time-series PNG images to optimized Zarr format"""
    input_path = Path(input_dir)
    output_zarr = Path(output_path)
    
    # Parse chunk size
    chunk_tuple = tuple(map(int, chunk_size.split(',')))
    if len(chunk_tuple) != 5:
        typer.echo("Error: Chunk size must have 5 values (time, dish, height, width, channels)", err=True)
        raise typer.Exit(1)
    
    # Create the Zarr dataset
    zarr_dataset = create_zarr_dataset(
        input_dir=input_path,
        output_path=output_zarr,
        chunk_size=chunk_tuple
    )
    
    # Verify the dataset
    verify_zarr_dataset(output_zarr)

@app.command()
def info(zarr_path: str = typer.Argument(..., help="Path to Zarr dataset")):
    """Display information about a Zarr dataset"""
    verify_zarr_dataset(Path(zarr_path))

@app.command()
def example():
    """Show example usage of the created Zarr dataset"""
    typer.echo("\nExample usage of the Zarr dataset:")
    typer.echo("""
    import zarr
    
    # Load the dataset
    root = zarr.open("./physarum_experiment_010.zarr", mode='r')
    
    # Access specific data
    first_dish_all_times = root.images[:, 0]  # All timepoints for dish 0
    specific_timepoint = root.images[100]     # All dishes at timepoint 100
    dish_timeseries = root.images[:, 3]       # Time series for dish 3
    
    # Access metadata
    print(root.images.attrs['dish_layout'])
    print(root.timestamps[0])  # First timestamp
    """)

if __name__ == "__main__":
    app()