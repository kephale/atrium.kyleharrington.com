# /// script
# title = "Physarum Time-Series to Zarr"
# description = "Extract and compress physarum petri dish images from time-series into efficient Zarr format"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.3.0"
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
from skimage import measure, filters, morphology, feature

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
    width, height = image.size
    if width < height:  # Taller than wide
        return 2, 3  # 2 rows, 3 columns
    else:
        return 3, 2  # 3 rows, 2 columns

def improved_circular_dish_detection(image: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    """
    Enhanced detection for circular glass petri dishes.
    Uses multiple detection strategies and selects the best one.
    
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
    
    height, width = gray.shape
    
    # Multiple detection strategies
    strategies = []
    
    # 1. Edge-based detection (good for glass dishes with clear edges)
    try:
        # Apply edge detection
        edges = feature.canny(gray, sigma=3.0)
        
        # Dilate edges to connect any gaps
        edges = morphology.dilation(edges, morphology.disk(2))
        
        # Apply Hough Circle Transform to find circles in the edge image
        circles = cv2.HoughCircles(
            edges.astype(np.uint8) * 255, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=min(width, height)/2,  # Only detect one circle
            param1=50, 
            param2=30, 
            minRadius=int(min(width, height) * 0.3),  # Dish should be at least 30% of image size
            maxRadius=int(min(width, height) * 0.5)   # Dish should be at most 50% of image size
        )
        
        if circles is not None:
            # Convert circle parameters to integers
            circles = np.round(circles[0, :]).astype(int)
            
            # Take the first circle (assuming only one was detected)
            x, y, r = circles[0]
            
            # Create a circular mask
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Add to strategies
            strategies.append((x, y, r, mask, "edge_hough"))
    except Exception:
        pass
    
    # 2. Contour-based detection with adaptive thresholding
    try:
        # Apply adaptive thresholding to handle uneven lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 101, 5
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea, default=None)
        
        if largest_contour is not None:
            # Find the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            
            # Ensure the circle fits within the image
            x, y, radius = int(x), int(y), int(radius)
            radius = min(radius, min(x, y, width-x, height-y))
            
            # Create a circular mask
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            
            # Calculate circularity
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
            else:
                circularity = 0
                
            # Add to strategies
            strategies.append((x, y, radius, mask, "adaptive_contour", circularity))
    except Exception:
        pass
    
    # 3. Simple circular approximation (based on image size)
    try:
        # Assume dish is centered and nearly filling the image
        center_x, center_y = width // 2, height // 2
        
        # The dish typically takes up most of the image
        # Estimate radius to be 40% of the smaller dimension to ensure we get the inner area
        radius = min(width, height) * 0.4
        
        # Create a circular mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(center_x), int(center_y)), int(radius), 255, -1)
        
        # Add to strategies
        strategies.append((int(center_x), int(center_y), int(radius), mask, "estimated"))
    except Exception:
        pass
    
    # 4. Try classic Hough Circle detection with the original image
    try:
        # Blur the image to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=min(width, height)/2,
            param1=50, 
            param2=30, 
            minRadius=int(min(width, height) * 0.3), 
            maxRadius=int(min(width, height) * 0.48)
        )
        
        if circles is not None:
            # Convert circle parameters to integers
            circles = np.round(circles[0, :]).astype(int)
            
            # Take the first circle (assuming only one was detected)
            x, y, r = circles[0]
            
            # Create a circular mask
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Add to strategies
            strategies.append((x, y, r, mask, "direct_hough"))
    except Exception:
        pass
    
    # 5. Use OTSU thresholding with morphological operations
    try:
        # Apply OTSU thresholding 
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Find the minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            x, y, radius = int(x), int(y), int(radius)
            
            # Create a circular mask
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)
            
            # Add to strategies
            strategies.append((x, y, radius, mask, "otsu"))
    except Exception:
        pass
    
    # If no strategies worked, use a fallback
    if not strategies:
        # Fallback to using a centered circle covering most of the image
        center_x, center_y = width // 2, height // 2
        radius = int(min(width, height) * 0.45)  # Cover most of the image
        
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        return center_x, center_y, radius, mask > 0
    
    # Choose the best strategy:
    # Preference: direct_hough > edge_hough > adaptive_contour > otsu > estimated
    # This is based on how reliable each method typically is for glass petri dishes
    
    method_preference = {
        "direct_hough": 1,
        "edge_hough": 2,
        "adaptive_contour": 3,
        "otsu": 4,
        "estimated": 5
    }
    
    # Sort strategies by preference
    strategies.sort(key=lambda s: method_preference.get(s[4], 99))
    
    # Select the best strategy
    best = strategies[0]
    
    # Return the results
    return best[0], best[1], best[2], best[3] > 0

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
                cx, cy, radius, mask = improved_circular_dish_detection(dish)
                
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

@app.command()
def test_detection(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    dish_index: int = typer.Option(0, "--dish", "-d", help="Index of the dish to test (0-5)"),
    save_path: str = typer.Option(None, "--save", "-s", help="Path to save visualization (optional)")
):
    """Test the dish detection algorithm on a single dish from an image"""
    # Load image
    image = Image.open(image_path)
    img_array = np.array(image)
    
    # Detect dish layout
    rows, cols = detect_dish_layout(image)
    typer.echo(f"Detected dish layout: {rows}x{cols}")
    
    # Calculate dish dimensions
    height, width = img_array.shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    # Calculate dish coordinates
    row = dish_index // cols
    col = dish_index % cols
    
    top = row * dish_height
    left = col * dish_width
    bottom = top + dish_height
    right = left + dish_width
    
    # Extract the dish region
    dish = img_array[top:bottom, left:right].copy()
    
    # Detect the circular dish using the improved algorithm
    cx, cy, radius, mask = improved_circular_dish_detection(dish)
    
    # Apply the mask
    masked_dish = dish.copy()
    if len(dish.shape) == 3:
        for c in range(dish.shape[2]):
            masked_dish[:,:,c] = dish[:,:,c] * mask
    else:
        masked_dish = dish * mask
    
    # Visualize the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(dish)
    plt.title("Original Dish")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"Detected Mask (center: {cx},{cy}, radius: {radius})")
    
    plt.subplot(1, 3, 3)
    plt.imshow(masked_dish)
    plt.title("Masked Dish")
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150)
        typer.echo(f"Saved visualization to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    app()