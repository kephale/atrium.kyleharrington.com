# /// script
# title = "Physarum Time-Series to Zarr"
# description = "Extract and compress physarum petri dish images from time-series into efficient Zarr format with optimized glass dish detection"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.3.0"
# keywords = ["physarum", "time-series", "zarr", "image-processing", "compression", "circular-detection", "glass-dish"]
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
#     "scikit-image>=0.21.0",
#     "matplotlib>=3.7.0"
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
from skimage import measure, filters, exposure, feature
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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

def detect_dish_layout(image: Image.Image, manual_rows: int = None, manual_cols: int = None) -> tuple[int, int]:
    """
    Analyze image to determine dish layout or use manual layout if provided
    
    Args:
        image: Input image
        manual_rows: Manually specified number of rows
        manual_cols: Manually specified number of columns
        
    Returns:
        tuple: (rows, cols)
    """
    if manual_rows is not None and manual_cols is not None:
        return manual_rows, manual_cols
    
    # Automatic detection based on image aspect ratio
    width, height = image.size
    if width > height:
        return 3, 2  # Default to 3x2 for wider images
    else:
        return 2, 3  # 2x3 for taller images

def extract_dish_region(image: np.ndarray, row: int, col: int, rows: int, cols: int) -> np.ndarray:
    """Extract a single dish region from the image"""
    height, width = image.shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    top = row * dish_height
    left = col * dish_width
    bottom = min(top + dish_height, height)
    right = min(left + dish_width, width)
    
    return image[top:bottom, left:right].copy()

def detect_glass_dish(image: np.ndarray, 
                     margin_factor: float = 0.03, 
                     downsample_ratio: float = 0.5) -> tuple:
    """
    Specialized detection for glass petri dishes with clear boundaries
    
    Args:
        image: Input image (single dish region)
        margin_factor: How much margin to leave from the edges (0-1)
        downsample_ratio: Ratio to downsample image for processing (0-1)
    
    Returns:
        tuple: (center_x, center_y, radius, mask)
    """
    # Get original image dimensions
    orig_height, orig_width = image.shape[:2]
    
    # Downsample if requested
    if downsample_ratio < 1.0:
        # Calculate new dimensions
        new_width = int(orig_width * downsample_ratio)
        new_height = int(orig_height * downsample_ratio)
        
        # Ensure minimum size
        new_width = max(new_width, 100)
        new_height = max(new_height, 100)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
        new_width, new_height = orig_width, orig_height
    
    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = resized.copy()
    
    # Apply histogram equalization to enhance contrast
    gray_eq = exposure.equalize_hist(gray)
    gray_eq = (gray_eq * 255).astype(np.uint8)
    
    # Create edge images using different methods
    edges1 = cv2.Canny(gray_eq, 50, 150)
    edges2 = feature.canny(gray, sigma=2)
    
    # Combine edge images
    edges = np.maximum(edges1, edges2.astype(np.uint8) * 255)
    
    # Dilate edges to connect gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply circular Hough transform with parameters tuned for glass petri dishes
    circles = cv2.HoughCircles(
        gray_eq,
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=min(new_width, new_height) * 0.8,  # Only detect one circle
        param1=30,  # Lower threshold for edge detection
        param2=30,  # Accumulator threshold (lower = more circles)
        minRadius=int(min(new_height, new_width) * (0.3 + margin_factor)),
        maxRadius=int(min(new_height, new_width) * (0.5 - margin_factor))
    )
    
    # If circles are detected
    if circles is not None:
        # Get the circle with highest detection confidence
        x, y, r = np.round(circles[0, 0]).astype(int)
        
        # Create mask
        mask = np.zeros((new_height, new_width), dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)
        
        # If downsampled, scale back the coordinates and mask
        if downsample_ratio < 1.0:
            scale = 1.0 / downsample_ratio
            x_orig = int(x * scale)
            y_orig = int(y * scale)
            r_orig = int(r * scale)
            
            # Create full-size mask
            full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
            cv2.circle(full_mask, (x_orig, y_orig), r_orig, 255, -1)
            
            return x_orig, y_orig, r_orig, full_mask > 0
        else:
            return x, y, r, mask > 0
    
    # If Hough transform fails, try direct edge-based detection
    try:
        # Find contours in edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Filter contours by area and circularity
            candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Skip tiny contours
                    continue
                    
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # Calculate how central the contour is
                moments = cv2.moments(contour)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # Distance from center
                    center_dist = np.sqrt((cx - new_width/2)**2 + (cy - new_height/2)**2)
                    center_dist_ratio = center_dist / (min(new_width, new_height) / 2)
                    
                    candidates.append((contour, circularity, area, center_dist_ratio))
            
            if candidates:
                # Sort by combination of circularity, area, and centrality
                candidates.sort(key=lambda x: (-x[1] * 0.5 - (x[2] / (new_width * new_height)) * 0.3 - (1 - x[3]) * 0.2))
                best_contour = candidates[0][0]
                
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                x, y, radius = int(x), int(y), int(radius)
                
                # Create mask
                mask = np.zeros((new_height, new_width), dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)
                
                # If downsampled, scale back the coordinates and mask
                if downsample_ratio < 1.0:
                    scale = 1.0 / downsample_ratio
                    x_orig = int(x * scale)
                    y_orig = int(y * scale)
                    r_orig = int(radius * scale)
                    
                    # Create full-size mask
                    full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
                    cv2.circle(full_mask, (x_orig, y_orig), r_orig, 255, -1)
                    
                    return x_orig, y_orig, r_orig, full_mask > 0
                else:
                    return x, y, radius, mask > 0
    except Exception:
        pass
    
    # Fallback: Use a centered circle with fixed size ratio
    center_x = new_width // 2
    center_y = new_height // 2
    radius = int(min(new_width, new_height) * 0.45)
    
    mask = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    # If downsampled, scale back the coordinates and mask
    if downsample_ratio < 1.0:
        scale = 1.0 / downsample_ratio
        x_orig = int(center_x * scale)
        y_orig = int(center_y * scale)
        r_orig = int(radius * scale)
        
        # Create full-size mask
        full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        cv2.circle(full_mask, (x_orig, y_orig), r_orig, 255, -1)
        
        return x_orig, y_orig, r_orig, full_mask > 0
    else:
        return center_x, center_y, radius, mask > 0

def detect_dishes_first_frame(image: np.ndarray, rows: int, cols: int, 
                             margin_factor: float, downsample_ratio: float) -> list:
    """
    Detect dishes from the first frame to use as reference
    
    Args:
        image: Input first frame image array
        rows: Number of rows in the dish layout
        cols: Number of columns in the dish layout
        margin_factor: Margin factor for detection
        downsample_ratio: Downsample ratio for processing
        
    Returns:
        list: Reference dish metadata (center_x, center_y, radius, mask)
    """
    height, width = image.shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    reference_dishes = []
    
    for row in range(rows):
        for col in range(cols):
            # Extract dish region
            dish_img = extract_dish_region(image, row, col, rows, cols)
            
            # Detect dish
            x, y, r, mask = detect_glass_dish(dish_img, margin_factor, downsample_ratio)
            
            # Convert coordinates to full image space
            full_x = x + (col * dish_width)
            full_y = y + (row * dish_height)
            
            # Create full image mask
            full_mask = np.zeros((height, width), dtype=bool)
            
            # Add the dish mask to the right position in the full mask
            top = row * dish_height
            left = col * dish_width
            bottom = min(top + dish_height, height)
            right = min(left + dish_width, width)
            
            full_mask[top:bottom, left:right] = mask
            
            reference_dishes.append((full_x, full_y, r, full_mask))
    
    return reference_dishes

def apply_reference_masks(image: np.ndarray, reference_dishes: list, rows: int, cols: int) -> list:
    """
    Apply reference dish masks to an image
    
    Args:
        image: Input image array
        reference_dishes: Reference dish metadata
        rows: Number of rows in the dish layout
        cols: Number of columns in the dish layout
        
    Returns:
        list: Masked dish images
    """
    height, width = image.shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    masked_dishes = []
    
    for i, (_, _, _, mask) in enumerate(reference_dishes):
        row = i // cols
        col = i % cols
        
        # Extract dish region
        top = row * dish_height
        left = col * dish_width
        bottom = min(top + dish_height, height)
        right = min(left + dish_width, width)
        
        dish = image[top:bottom, left:right].copy()
        
        # Get corresponding mask for this dish
        dish_mask = mask[top:bottom, left:right]
        
        # Apply mask
        masked_dish = dish.copy()
        if len(dish.shape) == 3:
            for c in range(dish.shape[2]):
                masked_dish[:,:,c] = dish[:,:,c] * dish_mask
        else:
            masked_dish = dish * dish_mask
        
        masked_dishes.append(masked_dish)
    
    return masked_dishes

def extract_dishes(image: np.ndarray, reference_dishes: list, rows: int, cols: int) -> list:
    """
    Extract and mask dishes from an image using reference masks
    
    Args:
        image: Input image array
        reference_dishes: Reference dish metadata (if None, new masks are detected)
        rows: Number of rows in the dish layout
        cols: Number of columns in the dish layout
        
    Returns:
        list: Masked dish images
    """
    return apply_reference_masks(image, reference_dishes, rows, cols)

def create_zarr_dataset(
    input_dir: Path, 
    output_path: Path, 
    chunk_size: tuple[int, int, int, int, int] = (10, 1, 400, 400, 3),
    rows: int = 3,
    cols: int = 2,
    apply_mask: bool = True,
    margin_factor: float = 0.03,
    downsample_ratio: float = 0.5,
    sample_interval: int = 10
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
    
    # Process first image to get reference dish metadata
    typer.echo("Processing first frame to detect reference dishes...")
    first_image = np.array(Image.open(files[0]))
    reference_dishes = detect_dishes_first_frame(
        first_image, rows, cols, margin_factor, downsample_ratio
    )
    
    # Get sample dishes to determine shape
    sample_dishes = extract_dishes(first_image, reference_dishes, rows, cols)
    dish_shape = sample_dishes[0].shape
    typer.echo(f"Each dish shape: {dish_shape}")
    
    # Create Zarr dataset
    total_dishes = rows * cols
    zarr_shape = (len(files), total_dishes, dish_shape[0], dish_shape[1], dish_shape[2])
    
    typer.echo(f"Creating Zarr dataset with shape: {zarr_shape}")
    typer.echo(f"Chunk size: {chunk_size}")
    typer.echo(f"Using {rows}x{cols} layout with circular masking")
    
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
    data.attrs['mask_margin_factor'] = margin_factor
    data.attrs['detection_downsample_ratio'] = downsample_ratio
    
    # Create timestamps array
    timestamp_array = root.create_dataset(
        'timestamps',
        data=[ts.isoformat() for ts in timestamps],
        dtype='<U25'
    )
    timestamp_array.attrs['description'] = 'Timestamp for each image in ISO format'
    
    # Process all files
    typer.echo("Processing all frames...")
    for i, file in enumerate(tqdm(files)):
        try:
            img = np.array(Image.open(file))
            
            # Extract and mask dishes using reference masks
            dishes = extract_dishes(img, reference_dishes, rows, cols)
            
            # Store in zarr array
            for j, dish in enumerate(dishes):
                data[i, j] = dish
                
            # Periodically verify masks against a new detection (every sample_interval frames)
            # This helps ensure masks remain correct even if camera or dishes shift slightly
            if i > 0 and i % sample_interval == 0 and apply_mask:
                # Detect dishes in current frame
                current_dishes = detect_dishes_first_frame(
                    img, rows, cols, margin_factor, downsample_ratio
                )
                
                # Check if positions have changed significantly
                position_changed = False
                for j, ((ref_x, ref_y, ref_r, _), (cur_x, cur_y, cur_r, _)) in enumerate(zip(reference_dishes, current_dishes)):
                    # Calculate position change
                    dist = np.sqrt((ref_x - cur_x)**2 + (ref_y - cur_y)**2)
                    radius_ratio = abs(ref_r - cur_r) / max(ref_r, cur_r)
                    
                    # If position changed by more than 10% of radius or radius changed by more than 10%
                    if dist > 0.1 * ref_r or radius_ratio > 0.1:
                        position_changed = True
                        typer.echo(f"Frame {i}: Dish {j} position changed significantly. Updating reference masks.")
                        break
                
                # Update reference masks if needed
                if position_changed:
                    reference_dishes = current_dishes
                    
        except Exception as e:
            typer.echo(f"Error processing {file}: {e}", err=True)
    
    # Add dish metadata
    dish_metadata = root.create_group('dish_metadata')
    
    for i, (cx, cy, radius, mask) in enumerate(reference_dishes):
        row = i // cols
        col = i % cols
        dish_meta = dish_metadata.create_group(f'dish_{i}')
        dish_meta.attrs['row'] = row
        dish_meta.attrs['column'] = col
        dish_meta.attrs['position'] = f'row_{row}_col_{col}'
        
        # Add circular dish metadata
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
def visualize_detection(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    output_path: str = typer.Option(None, "--output", "-o", help="Path to save visualization"),
    rows: int = typer.Option(3, "--rows", "-r", help="Number of rows in dish layout"),
    cols: int = typer.Option(2, "--cols", "-c", help="Number of columns in dish layout"),
    margin_factor: float = typer.Option(0.03, "--margin", "-m", help="Margin factor for dish detection"),
    downsample_ratio: float = typer.Option(0.5, "--downsample", "-d", help="Downsample ratio for processing")
):
    """Visualize dish detection on a sample image"""
    try:
        # Load image
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Detect dishes
        reference_dishes = detect_dishes_first_frame(
            img_array, rows, cols, margin_factor, downsample_ratio
        )
        
        # Apply masks
        masked_dishes = apply_reference_masks(img_array, reference_dishes, rows, cols)
        
        # Create visualization
        height, width = img_array.shape[:2]
        dish_height = height // rows
        dish_width = width // cols
        
        # Plot original with circles
        plt.figure(figsize=(12, 10))
        plt.imshow(img_array)
        plt.title(f"Detected Dishes ({rows}x{cols} layout)")
        
        # Draw grid lines
        for i in range(1, rows):
            plt.axhline(y=i * dish_height, color='white', linestyle='--', alpha=0.5)
        
        for i in range(1, cols):
            plt.axvline(x=i * dish_width, color='white', linestyle='--', alpha=0.5)
        
        # Draw circles
        for i, (cx, cy, r, _) in enumerate(reference_dishes):
            circle = Circle((cx, cy), r, fill=False, edgecolor='cyan', linewidth=2)
            plt.gca().add_patch(circle)
            plt.text(cx, cy, f"#{i}", color='white', fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, pad=1))
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=150)
            typer.echo(f"Saved visualization to {output_path}")
        
        plt.show()
        
        # Also show individual dishes
        fig, axs = plt.subplots(2, len(masked_dishes), figsize=(4*len(masked_dishes), 8))
        
        for i, dish in enumerate(masked_dishes):
            row = i // cols
            col = i % cols
            
            # Original dish
            original = extract_dish_region(img_array, row, col, rows, cols)
            axs[0, i].imshow(original)
            axs[0, i].set_title(f"Dish #{i} - Original")
            axs[0, i].axis('off')
            
            # Masked dish
            axs[1, i].imshow(dish)
            axs[1, i].set_title(f"Dish #{i} - Masked")
            axs[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            crops_output = os.path.splitext(output_path)[0] + "_crops" + os.path.splitext(output_path)[1]
            plt.savefig(crops_output, dpi=150)
            typer.echo(f"Saved crops to {crops_output}")
        
        plt.show()
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        import traceback
        traceback.print_exc()

@app.command()
def convert(
    input_dir: str = typer.Argument(..., help="Directory containing physarum PNG images"),
    output_path: str = typer.Argument(..., help="Output path for Zarr dataset"),
    chunk_size: str = typer.Option("10,1,400,400,3", "--chunk-size", "-c", help="Chunk size as comma-separated values"),
    rows: int = typer.Option(3, "--rows", "-r", help="Number of rows in dish layout"),
    cols: int = typer.Option(2, "--cols", "-c", help="Number of columns in dish layout"),
    apply_mask: bool = typer.Option(True, "--mask/--no-mask", help="Apply circular dish masking"),
    margin_factor: float = typer.Option(0.03, "--margin", "-m", help="Margin factor for dish detection"),
    downsample_ratio: float = typer.Option(0.5, "--downsample", "-d", help="Downsample ratio for processing"),
    sample_interval: int = typer.Option(10, "--interval", "-i", help="Interval for checking dish position consistency")
):
    """Convert physarum time-series PNG images to optimized Zarr format with circular masking"""
    input_path = Path(input_dir)
    output_zarr = Path(output_path)
    
    # Parse chunk size
    chunk_tuple = tuple(map(int, chunk_size.split(',')))
    if len(chunk_tuple) != 5:
        typer.echo("Error: Chunk size must have 5 values (time, dish, height, width, channels)", err=True)
        raise typer.Exit(1)
    
    # Validate parameters
    if rows <= 0 or cols <= 0:
        typer.echo("Error: Rows and columns must be positive integers", err=True)
        raise typer.Exit(1)
        
    if margin_factor < 0 or margin_factor > 0.2:
        typer.echo("Warning: Margin factor should typically be between 0 and 0.2", err=True)
        
    if downsample_ratio <= 0 or downsample_ratio > 1.0:
        typer.echo("Error: Downsample ratio must be between 0 and 1.0", err=True)
        raise typer.Exit(1)
    
    # Create the Zarr dataset
    zarr_dataset = create_zarr_dataset(
        input_dir=input_path,
        output_path=output_zarr,
        chunk_size=chunk_tuple,
        rows=rows,
        cols=cols,
        apply_mask=apply_mask,
        margin_factor=margin_factor,
        downsample_ratio=downsample_ratio,
        sample_interval=sample_interval
    )
    
    # Verify the dataset
    verify_zarr_dataset(output_zarr)

@app.command()
def info(zarr_path: str = typer.Argument(..., help="Path to Zarr dataset")):
    """Display information about a Zarr dataset"""
    verify_zarr_dataset(Path(zarr_path))

@app.command()
def test_detection(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    dish_index: int = typer.Option(0, "--dish", "-d", help="Index of the dish to test (0-5)"),
    rows: int = typer.Option(3, "--rows", "-r", help="Number of rows in dish layout"),
    cols: int = typer.Option(2, "--cols", "-c", help="Number of columns in dish layout"),
    margin_factor: float = typer.Option(0.03, "--margin", "-m", help="Margin factor for dish detection"),
    downsample_ratio: float = typer.Option(0.5, "--downsample", "-d", help="Downsample ratio for processing"),
    save_path: str = typer.Option(None, "--save", "-s", help="Path to save visualization (optional)")
):
    """Test the dish detection algorithm on a single dish from an image"""
    # Load image
    image = Image.open(image_path)
    img_array = np.array(image)
    
    # Validate dish index
    max_dish_index = rows * cols - 1
    if dish_index < 0 or dish_index > max_dish_index:
        typer.echo(f"Error: Dish index must be between 0 and {max_dish_index}", err=True)
        raise typer.Exit(1)
    
    # Calculate dish coordinates
    height, width = img_array.shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    row = dish_index // cols
    col = dish_index % cols
    
    # Extract the dish region
    dish = extract_dish_region(img_array, row, col, rows, cols)
    
    # Detect the circular dish
    cx, cy, radius, mask = detect_glass_dish(dish, margin_factor, downsample_ratio)
    
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
    plt.title(f"Original Dish #{dish_index} (r{row}c{col})")
    
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