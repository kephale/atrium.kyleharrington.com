# /// script
# title = "Visualize Physarum Petri Dish Detection"
# description = "Demonstration tool for visualizing the glass petri dish detection used in physarum zarr conversion"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.2.0"
# keywords = ["physarum", "image-processing", "circular-detection", "visualization", "glass-dish"]
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
#     "numpy>=1.24.0",
#     "pillow>=10.0.0",
#     "typer>=0.9.0",
#     "opencv-python>=4.8.0",
#     "scikit-image>=0.21.0",
#     "matplotlib>=3.7.0"
# ]
# ///

import os
import numpy as np
from PIL import Image
import typer
from pathlib import Path
import cv2
from skimage import measure, filters, exposure, morphology, feature
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

app = typer.Typer()

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
        tuple: (center_x, center_y, radius, mask, method)
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
            
            return x_orig, y_orig, r_orig, full_mask > 0, "hough"
        else:
            return x, y, r, mask > 0, "hough"
    
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
                    
                    return x_orig, y_orig, r_orig, full_mask > 0, "contour"
                else:
                    return x, y, radius, mask > 0, "contour"
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
        
        return x_orig, y_orig, r_orig, full_mask > 0, "fallback"
    else:
        return center_x, center_y, radius, mask > 0, "fallback"

def enhance_dish_visibility(image: np.ndarray) -> np.ndarray:
    """Enhance the visibility of glass dish edges"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply unsharp mask to enhance edges
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    return unsharp

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
        list: Reference dish metadata (center_x, center_y, radius, mask, method)
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
            x, y, r, mask, method = detect_glass_dish(dish_img, margin_factor, downsample_ratio)
            
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
            
            reference_dishes.append((full_x, full_y, r, full_mask, method))
    
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
    
    for i, (_, _, _, mask, _) in enumerate(reference_dishes):
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

@app.command()
def visualize(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    output_path: str = typer.Option(None, "--output", "-o", help="Path to save visualization image"),
    show_plot: bool = typer.Option(True, "--show/--no-show", help="Whether to display the plot"),
    margin_factor: float = typer.Option(0.03, "--margin", "-m", help="Margin factor to leave from edges (0-0.2)"),
    manual_rows: int = typer.Option(3, "--rows", "-r", help="Number of rows in dish layout"),
    manual_cols: int = typer.Option(2, "--cols", "-c", help="Number of columns in dish layout"),
    downsample: float = typer.Option(0.5, "--downsample", "-d", help="Downsample ratio for processing (0.1-1.0)")
):
    """Visualize glass petri dish detection"""
    # Load image
    try:
        image = Image.open(image_path)
        img_array = np.array(image)
        print(f"Loaded image with size: {image.size}")
        
        # Use manual layout if specified, otherwise detect
        rows, cols = manual_rows, manual_cols
        print(f"Using dish layout: {rows}x{cols}")
            
        # Validate downsample ratio
        downsample = max(0.1, min(1.0, downsample))
        if downsample < 1.0:
            print(f"Using downsample ratio: {downsample:.2f}")
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot original image
        plt.imshow(img_array)
        plt.title(f"Glass Petri Dish Detection - {rows}x{cols} layout", fontsize=16)
        
        # Draw grid lines
        height, width = img_array.shape[:2]
        dish_height = height // rows
        dish_width = width // cols
        
        for i in range(1, rows):
            plt.axhline(y=i * dish_height, color='white', linestyle='--', alpha=0.7)
        
        for i in range(1, cols):
            plt.axvline(x=i * dish_width, color='white', linestyle='--', alpha=0.7)
        
        # Detect all dishes
        reference_dishes = detect_dishes_first_frame(
            img_array, rows, cols, margin_factor, downsample
        )
        
        # Draw circles for each dish
        for i, (x, y, r, _, method) in enumerate(reference_dishes):
            circle = Circle((x, y), r, fill=False, edgecolor='cyan', linewidth=3)
            plt.gca().add_patch(circle)
            
            # Add label
            plt.text(x, y, f"#{i}\n{method}", 
                    color='white', fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved visualization to {output_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Also create a figure showing individual dishes
        masked_dishes = apply_reference_masks(
            img_array, reference_dishes, rows, cols
        )
        
        fig, axs = plt.subplots(2, len(masked_dishes), figsize=(4*len(masked_dishes), 8))
        
        for i in range(len(masked_dishes)):
            # Calculate dish coordinates
            row = i // cols
            col = i % cols
            
            # Extract original dish
            dish_img = extract_dish_region(img_array, row, col, rows, cols)
            
            # Get circle parameters
            cx, cy, r, _, method = reference_dishes[i]
            
            # Convert to local coordinates
            local_x = cx - (col * dish_width)
            local_y = cy - (row * dish_height)
            
            # Original with circle
            axs[0, i].imshow(dish_img)
            axs[0, i].set_title(f"Dish #{i} - Original")
            circle = Circle((local_x, local_y), r, fill=False, edgecolor='cyan', linewidth=2)
            axs[0, i].add_patch(circle)
            axs[0, i].axis('off')
            
            # Masked
            axs[1, i].imshow(masked_dishes[i])
            axs[1, i].set_title(f"Dish #{i} - Masked ({method})")
            axs[1, i].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            details_output = os.path.splitext(output_path)[0] + "_details" + os.path.splitext(output_path)[1]
            plt.savefig(details_output, dpi=150)
            print(f"Saved detection details to {details_output}")
        
        # Show if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

@app.command()
def compare_methods(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    dish_index: int = typer.Option(0, "--dish", "-d", help="Index of the dish to analyze (0-5)"),
    rows: int = typer.Option(3, "--rows", "-r", help="Number of rows in dish layout"),
    cols: int = typer.Option(2, "--cols", "-c", help="Number of columns in dish layout"),
    output_path: str = typer.Option(None, "--output", "-o", help="Path to save comparison image")
):
    """Compare different processing steps for a single dish"""
    # Load image
    try:
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Validate dish index
        if dish_index < 0 or dish_index >= rows * cols:
            print(f"Error: Dish index must be between 0 and {rows * cols - 1}")
            raise typer.Exit(1)
        
        # Extract the specified dish
        row = dish_index // cols
        col = dish_index % cols
        dish = extract_dish_region(img_array, row, col, rows, cols)
        
        # Initialize figure with enough subplots for all processing steps
        plt.figure(figsize=(20, 15))
        
        # 1. Original image
        plt.subplot(3, 3, 1)
        plt.imshow(dish)
        plt.title("Original Dish", fontsize=14)
        
        # 2. Grayscale conversion
        if len(dish.shape) == 3:
            gray = cv2.cvtColor(dish, cv2.COLOR_RGB2GRAY)
        else:
            gray = dish.copy()
            
        plt.subplot(3, 3, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale", fontsize=14)
        
        # 3. Histogram equalized version
        gray_eq = exposure.equalize_hist(gray)
        gray_eq_vis = (gray_eq * 255).astype(np.uint8)
        
        plt.subplot(3, 3, 3)
        plt.imshow(gray_eq_vis, cmap='gray')
        plt.title("Histogram Equalized", fontsize=14)
        
        # 4. Edge detection 1 (Canny on equalized)
        edges1 = cv2.Canny(gray_eq_vis, 50, 150)
        
        plt.subplot(3, 3, 4)
        plt.imshow(edges1, cmap='gray')
        plt.title("Canny Edge Detection 1", fontsize=14)
        
        # 5. Edge detection 2 (skimage Canny)
        edges2 = feature.canny(gray, sigma=2)
        edges2_vis = edges2.astype(np.uint8) * 255
        
        plt.subplot(3, 3, 5)
        plt.imshow(edges2_vis, cmap='gray')
        plt.title("Canny Edge Detection 2", fontsize=14)
        
        # 6. Combined edges
        edges_combined = np.maximum(edges1, edges2_vis)
        
        plt.subplot(3, 3, 6)
        plt.imshow(edges_combined, cmap='gray')
        plt.title("Combined Edges", fontsize=14)
        
        # 7. Dilated edges
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges_combined, kernel, iterations=1)
        
        plt.subplot(3, 3, 7)
        plt.imshow(edges_dilated, cmap='gray')
        plt.title("Dilated Edges", fontsize=14)
        
        # 8. Final result with full resolution
        x, y, r, mask, method = detect_glass_dish(dish, margin_factor=0.03, downsample_ratio=0.5)
        
        plt.subplot(3, 3, 8)
        plt.imshow(dish)
        circle = Circle((x, y), r, fill=False, edgecolor='cyan', linewidth=2)
        plt.gca().add_patch(circle)
        plt.title(f"Detection Result ({method})", fontsize=14)
        
        # 9. Masked result
        masked_dish = dish.copy()
        if len(dish.shape) == 3:
            for c in range(dish.shape[2]):
                masked_dish[:,:,c] = dish[:,:,c] * mask
        else:
            masked_dish = dish * mask
            
        plt.subplot(3, 3, 9)
        plt.imshow(masked_dish)
        plt.title("Masked Result", fontsize=14)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"Saved comparison to {output_path}")
            
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

@app.command()
def test_multiple_frames(
    input_dir: str = typer.Argument(..., help="Directory containing frames"),
    pattern: str = typer.Option("*.png", "--pattern", "-p", help="File pattern to match"),
    rows: int = typer.Option(3, "--rows", "-r", help="Number of rows in dish layout"),
    cols: int = typer.Option(2, "--cols", "-c", help="Number of columns in dish layout"),
    margin_factor: float = typer.Option(0.03, "--margin", "-m", help="Margin factor for detection"),
    downsample: float = typer.Option(0.5, "--downsample", "-d", help="Downsample ratio for processing"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save visualizations")
):
    """Test dish detection consistency across multiple frames"""
    try:
        # Find all matching files
        import glob
        files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        if not files:
            print(f"Error: No files found matching pattern '{pattern}' in '{input_dir}'")
            raise typer.Exit(1)
            
        print(f"Found {len(files)} files")
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Process first frame to get reference dishes
        print("Processing first frame...")
        first_img = np.array(Image.open(files[0]))
        reference_dishes = detect_dishes_first_frame(
            first_img, rows, cols, margin_factor, downsample
        )
        
        # Create a visualization of first frame
        plt.figure(figsize=(15, 10))
        plt.imshow(first_img)
        plt.title(f"First Frame - Reference Dish Detection", fontsize=16)
        
        # Draw grid lines
        height, width = first_img.shape[:2]
        dish_height = height // rows
        dish_width = width // cols
        
        for i in range(1, rows):
            plt.axhline(y=i * dish_height, color='white', linestyle='--', alpha=0.7)
        
        for i in range(1, cols):
            plt.axvline(x=i * dish_width, color='white', linestyle='--', alpha=0.7)
        
        # Draw reference circles
        for i, (x, y, r, _, method) in enumerate(reference_dishes):
            circle = Circle((x, y), r, fill=False, edgecolor='cyan', linewidth=3)
            plt.gca().add_patch(circle)
            plt.text(x, y, f"#{i}\n{method}", 
                    color='white', fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
                    
        plt.tight_layout()
        
        # Save first frame visualization
        if output_dir:
            plt.savefig(os.path.join(output_dir, "reference_frame.png"), dpi=150)
            
        plt.show()
        
        # Process sample frames
        num_sample_frames = min(5, len(files))
        sample_indices = [0] + [int(i * len(files) / num_sample_frames) for i in range(1, num_sample_frames)]
        
        print(f"Processing {num_sample_frames} sample frames...")
        
        # Create figure for sample frames
        fig, axs = plt.subplots(1, num_sample_frames, figsize=(5*num_sample_frames, 8))
        
        # Check if multiple subplots exist
        if num_sample_frames == 1:
            axs = [axs]
            
        for i, idx in enumerate(sample_indices):
            frame_img = np.array(Image.open(files[idx]))
            
            # Apply reference masks
            masked_frame = np.copy(frame_img)
            for j, (_, _, _, mask, _) in enumerate(reference_dishes):
                masked_frame = masked_frame * mask.astype(np.uint8)[:,:,np.newaxis] + (1 - mask.astype(np.uint8)[:,:,np.newaxis]) * np.array([0, 0, 0])
            
            # Display
            axs[i].imshow(masked_frame)
            axs[i].set_title(f"Frame {idx}")
            axs[i].axis('off')
            
            # Draw reference circles
            for j, (x, y, r, _, _) in enumerate(reference_dishes):
                circle = Circle((x, y), r, fill=False, edgecolor='cyan', linewidth=2)
                axs[i].add_patch(circle)
        
        plt.tight_layout()
        
        # Save sample frames visualization
        if output_dir:
            plt.savefig(os.path.join(output_dir, "sample_frames.png"), dpi=150)
            
        plt.show()
        
        print("Finished testing across multiple frames")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)

if __name__ == "__main__":
    app()