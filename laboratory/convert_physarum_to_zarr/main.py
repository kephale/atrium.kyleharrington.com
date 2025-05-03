# /// script
# title = "Physarum Time-Series to Zarr"
# description = "Extract and compress physarum petri dish images from time-series into efficient Zarr format"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.2.0"
# keywords = ["physarum", "time-series", "zarr", "image-processing", "compression", "circular-detection"]
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
#     "zarr>=2.15.0,<3",
#     "numpy>=1.24.0",
#     "pillow>=10.0.0",
#     "tqdm>=4.65.0",
#     "typer>=0.9.0",
#     "opencv-python>=4.8.0",
#     "scikit-image>=0.21.0"
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
import cv2
from skimage import measure

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

def detect_circular_dish(image: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    """
    Detect the circular petri dish in the image and return its center and radius.
    Also returns a binary mask of the petri dish area.
    
    Args:
        image: Input image array
    
    Returns:
        tuple: (center_x, center_y, radius, mask)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding (adjust threshold as needed)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours = measure.find_contours(binary, 0.5)
    
    # Filter contours by area and circularity
    dish_contour = None
    max_area = 0
    
    for contour in contours:
        # Convert to format expected by OpenCV
        contour_int = np.around(contour).astype(np.int32)
        
        # Calculate area and perimeter
        area = cv2.contourArea(contour_int)
        perimeter = cv2.arcLength(contour_int, True)
        
        # Calculate circularity (perfect circle has circularity = 1)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
            
        # Check if this could be our dish (large, circular)
        if area > max_area and circularity > 0.7:  # Adjust threshold as needed
            dish_contour = contour_int
            max_area = area
    
    if dish_contour is None:
        # Fallback to using the entire image if no good dish contour is found
        height, width = gray.shape
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 2 - 10  # Slightly smaller than half the shortest dimension
        
        # Create a circular mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    else:
        # Find the center and radius of the best fitting circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(dish_contour)
        center_x, center_y, radius = int(center_x), int(center_y), int(radius)
        
        # Create a circular mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return center_x, center_y, radius, mask > 0

def extract_dishes(image: Image.Image, rows: int = 2, cols: int = 3, apply_mask: bool = True) -> tuple[list[np.ndarray], list[tuple[int, int, int, np.ndarray]]]:
    """
    Extract individual dishes from the composite image and apply circular masks
    
    Args:
        image: Input composite image
        rows: Number of rows in the dish layout
        cols: Number of columns in the dish layout
        apply_mask: Whether to apply circular masking
        
    Returns:
        tuple: (list of dish images, list of dish metadata (center_x, center_y, radius, mask))
    """
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    dish_height = height // rows
    dish_width = width // cols
    
    dishes = []
    dish_metadata = []
    
    for row in range(rows):
        for col in range(cols):
            top = row * dish_height
            left = col * dish_width
            bottom = top + dish_height
            right = left + dish_width
            
            # Extract the dish region
            dish = img_array[top:bottom, left:right].copy()
            
            if apply_mask:
                # Detect the circular dish and create a mask
                cx, cy, radius, mask = detect_circular_dish(dish)
                
                # Apply the mask (set background to black)
                masked_dish = dish.copy()
                if len(dish.shape) == 3:
                    # For RGB images
                    for c in range(dish.shape[2]):
                        masked_dish[:,:,c] = dish[:,:,c] * mask
                else:
                    # For grayscale images
                    masked_dish = dish * mask
                
                dishes.append(masked_dish)
                dish_metadata.append((cx, cy, radius, mask))
            else:
                dishes.append(dish)
                # Create a default mask (all ones) if not applying masks
                default_mask = np.ones(dish.shape[:2], dtype=bool)
                dish_metadata.append((dish_width//2, dish_height//2, min(dish_width, dish_height)//2, default_mask))
    
    return dishes, dish_metadata

def create_zarr_dataset(
    input_dir: Path, 
    output_path: Path, 
    chunk_size: tuple[int, int, int, int, int] = (10, 1, 400, 400, 3),
    apply_mask: bool = True,
    mask_background: bool = True
) -> zarr.Group:
    """Create Zarr dataset from physarum images with circular masking"""
    
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
    sample_dishes, dish_metadata = extract_dishes(first_image, rows, cols, apply_mask)
    dish_shape = sample_dishes[0].shape
    typer.echo(f"Each dish shape: {dish_shape}")
    
    # Create Zarr dataset
    total_dishes = rows * cols
    zarr_shape = (len(files), total_dishes, dish_shape[0], dish_shape[1], dish_shape[2])
    
    typer.echo(f"Creating Zarr dataset with shape: {zarr_shape}")
    typer.echo(f"Chunk size: {chunk_size}")
    typer.echo(f"Applying circular masking: {apply_mask}")
    
    # Create the main dataset
    zarr_store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=zarr_store)
    
    # Create the main data array
    data = root.create_dataset(
        'images',
        shape=zarr_shape,
        chunks=chunk_size,
        dtype=np.uint8,
        compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2)  # Increased compression level
    )
    
    # Add metadata
    experiment_name = os.path.basename(output_path).split('.')[0]
    data.attrs['description'] = 'Physarum polycephalum time-series imaging data'
    data.attrs['experiment'] = experiment_name
    data.attrs['dimensions'] = ['time', 'dish', 'height', 'width', 'channels']
    data.attrs['dish_layout'] = f'{rows}x{cols}'
    data.attrs['created_at'] = datetime.now().isoformat()
    data.attrs['circular_masking_applied'] = apply_mask
    
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
            dishes, _ = extract_dishes(img, rows, cols, apply_mask)
            
            for j, dish in enumerate(dishes):
                data[i, j] = dish
                
        except Exception as e:
            typer.echo(f"Error processing {file}: {e}", err=True)
    
    # Add dish metadata
    dish_metadata = root.create_group('dish_metadata')
    
    # Get first image dish metadata for storing masks
    _, first_dish_metadata = extract_dishes(first_image, rows, cols, apply_mask)
    
    for i in range(total_dishes):
        row = i // cols
        col = i % cols
        dish_meta = dish_metadata.create_group(f'dish_{i}')
        dish_meta.attrs['row'] = row
        dish_meta.attrs['column'] = col
        dish_meta.attrs['position'] = f'row_{row}_col_{col}'
        
        # Add circular dish metadata
        cx, cy, radius, mask = first_dish_metadata[i]
        dish_meta.attrs['center_x'] = cx
        dish_meta.attrs['center_y'] = cy
        dish_meta.attrs['radius'] = radius
        
        # Store binary mask (compressed)
        dish_meta.create_dataset(
            'mask',
            data=mask.astype(np.uint8),
            compressor=zarr.Blosc(cname='zstd', clevel=9, shuffle=2)
        )
    
    # Calculate compression efficiency
    original_size = sum(os.path.getsize(f) for f in files)
    compressed_size = get_dir_size(output_path)
    compression_ratio = original_size / max(compressed_size, 1)  # Avoid division by zero
    
    typer.echo(f"Dataset created successfully at: {output_path}")
    typer.echo(f"Original size: {original_size / (1024**3):.2f} GB")
    typer.echo(f"Compressed size: {compressed_size / (1024**3):.2f} GB")
    typer.echo(f"Compression ratio: {compression_ratio:.2f}x")
    
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
    
    # Check for masks
    if 'dish_metadata' in root:
        has_masks = False
        for dish_name in root.dish_metadata:
            if 'mask' in root.dish_metadata[dish_name]:
                has_masks = True
                break
        typer.echo(f"Contains dish masks: {has_masks}")
    
    return True

@app.command()
def convert(
    input_dir: str = typer.Argument(..., help="Directory containing physarum PNG images"),
    output_path: str = typer.Argument(..., help="Output path for Zarr dataset"),
    chunk_size: str = typer.Option("10,1,400,400,3", "--chunk-size", "-c", help="Chunk size as comma-separated values"),
    apply_mask: bool = typer.Option(True, "--mask/--no-mask", help="Apply circular dish masking"),
    mask_background: bool = typer.Option(True, "--mask-background/--keep-background", help="Set background outside dish to black")
):
    """Convert physarum time-series PNG images to optimized Zarr format with circular masking"""
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
        chunk_size=chunk_tuple,
        apply_mask=apply_mask,
        mask_background=mask_background
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
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load the dataset
    root = zarr.open("./physarum_experiment.zarr", mode='r')
    
    # Access specific data
    first_dish_all_times = root.images[:, 0]  # All timepoints for dish 0
    specific_timepoint = root.images[100]     # All dishes at timepoint 100
    dish_timeseries = root.images[:, 3]       # Time series for dish 3
    
    # Access metadata
    print(root.images.attrs['dish_layout'])
    print(root.timestamps[0])  # First timestamp
    
    # Load mask for a specific dish
    dish_id = 0
    mask = root.dish_metadata[f'dish_{dish_id}'].mask[:]
    
    # Visualize an image with its mask
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(root.images[0, dish_id])
    plt.title("Original dish image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Dish mask")
    
    plt.subplot(1, 3, 3)
    # Apply mask to show what's being tracked
    img = root.images[0, dish_id].copy()
    masked_img = img.copy()
    for c in range(img.shape[2]):
        masked_img[:,:,c] = img[:,:,c] * mask
    plt.imshow(masked_img)
    plt.title("Masked dish image")
    
    plt.tight_layout()
    plt.show()
    """)

if __name__ == "__main__":
    app()