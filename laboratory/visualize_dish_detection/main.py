# /// script
# title = "Visualize Physarum Petri Dish Detection"
# description = "Demonstration tool for visualizing the circular dish detection used in physarum zarr conversion"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.2.0"
# keywords = ["physarum", "image-processing", "circular-detection", "visualization"]
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
from skimage import measure, filters, morphology, feature
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

app = typer.Typer()

def detect_dish_layout(image: Image.Image) -> tuple[int, int]:
    """
    Analyze image to determine dish layout (2x3 or 3x2)
    This should be customized based on your specific setup
    """
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

def extract_dishes(image: Image.Image, rows: int = 2, cols: int = 3) -> tuple[list[np.ndarray], list[tuple[int, int, int, np.ndarray, str]]]:
    """
    Extract individual dishes from the composite image and detect circles in each
    
    Args:
        image: Input composite image
        rows: Number of rows in the dish layout
        cols: Number of columns in the dish layout
        
    Returns:
        tuple: (list of dish images, list of dish metadata (center_x, center_y, radius, mask, method))
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
            
            # Use the improved detection algorithm
            cx, cy, radius, mask = improved_circular_dish_detection(dish)
            
            dishes.append(dish)
            
            # Detect which method was used (simplified version)
            if radius >= min(dish_height, dish_width) * 0.3:
                method = "Enhanced detection"
            else:
                method = "Small region detected"
                
            dish_metadata.append((cx, cy, radius, mask, method))
    
    return dishes, dish_metadata

def visualize_detection_steps(dish_image: np.ndarray, save_path: str = None):
    """Visualize the steps in the dish detection process for a single dish"""
    # Convert to grayscale if needed
    if len(dish_image.shape) == 3:
        gray = cv2.cvtColor(dish_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = dish_image.copy()
    
    height, width = gray.shape
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Original grayscale image
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    
    # Edge detection
    edges = feature.canny(gray, sigma=3.0)
    plt.subplot(2, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    
    # Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 101, 5
    )
    plt.subplot(2, 3, 3)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title("Adaptive Threshold")
    
    # OTSU threshold
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.subplot(2, 3, 4)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("OTSU Threshold")
    
    # Final dish detection
    cx, cy, radius, mask = improved_circular_dish_detection(dish_image)
    
    # Original with circle overlay
    plt.subplot(2, 3, 5)
    plt.imshow(dish_image)
    circle = Circle((cx, cy), radius, fill=False, edgecolor='red', linewidth=2)
    plt.gca().add_patch(circle)
    plt.title(f"Detected Circle (r={radius})")
    
    # Masked result
    masked_dish = dish_image.copy()
    if len(dish_image.shape) == 3:
        for c in range(dish_image.shape[2]):
            masked_dish[:,:,c] = dish_image[:,:,c] * mask
    else:
        masked_dish = dish_image * mask
    
    plt.subplot(2, 3, 6)
    plt.imshow(masked_dish)
    plt.title("Final Masked Result")
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return plt.gcf()

@app.command()
def visualize(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save visualization (optional)"),
    show_plots: bool = typer.Option(True, "--show/--no-show", help="Show visualization plots"),
    dish_index: int = typer.Option(None, "--dish", "-d", help="Visualize detection steps for a specific dish index")
):
    """Visualize the dish detection and masking process"""
    # Load image
    image = Image.open(image_path)
    typer.echo(f"Loaded image with size: {image.size}")
    
    # Detect dish layout
    rows, cols = detect_dish_layout(image)
    typer.echo(f"Detected dish layout: {rows}x{cols}")
    
    # Extract dishes and detect circles
    dishes, dish_metadata = extract_dishes(image, rows, cols)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # If a specific dish is requested for detailed visualization
    if dish_index is not None and 0 <= dish_index < len(dishes):
        typer.echo(f"Visualizing detection steps for dish {dish_index}")
        save_path = os.path.join(output_dir, f"dish_{dish_index}_steps.png") if output_dir else None
        fig = visualize_detection_steps(dishes[dish_index], save_path)
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        return
    
    # Create a figure to visualize the whole process
    plt.figure(figsize=(18, 10))
    
    # Plot composite image with grid
    plt.subplot(2, 3, 1)
    plt.imshow(np.array(image))
    plt.title("Original Image with Dish Grid")
    
    # Draw grid lines
    height, width = np.array(image).shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    for i in range(1, rows):
        plt.axhline(y=i * dish_height, color='r', linestyle='--')
    
    for i in range(1, cols):
        plt.axvline(x=i * dish_width, color='r', linestyle='--')
    
    # Plot each dish with detected circle
    dish_idx = 1
    for row in range(rows):
        for col in range(cols):
            dish_idx += 1
            if dish_idx <= 6:  # Only show first 5 dishes to avoid overflow
                plt.subplot(2, 3, dish_idx)
                
                dish = dishes[row * cols + col]
                cx, cy, radius, mask, method = dish_metadata[row * cols + col]
                
                # Display the dish
                plt.imshow(dish)
                plt.title(f"Dish {row * cols + col + 1} ({method})")
                
                # Add circle overlay
                circle = Circle((cx, cy), radius, fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(circle)
    
    plt.tight_layout()
    
    # Another figure to show the masking effect
    plt.figure(figsize=(18, 10))
    
    for i, (dish, (cx, cy, radius, mask, _)) in enumerate(zip(dishes, dish_metadata)):
        if i >= 6:  # Only show first 6 dishes
            break
            
        plt.subplot(2, 3, i+1)
        
        # Create masked version
        masked_dish = dish.copy()
        if len(dish.shape) == 3:
            # For RGB images
            for c in range(dish.shape[2]):
                masked_dish[:,:,c] = dish[:,:,c] * mask
        else:
            # For grayscale images
            masked_dish = dish * mask
        
        # Side by side comparison with half width for each
        h, w = dish.shape[:2]
        comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
        
        # Handle both RGB and grayscale images for display
        if len(dish.shape) == 3:
            comparison[:, :w] = dish
            comparison[:, w:] = masked_dish
        else:
            # Convert grayscale to RGB for display
            comparison[:, :w, 0] = comparison[:, :w, 1] = comparison[:, :w, 2] = dish
            comparison[:, w:, 0] = comparison[:, w:, 1] = comparison[:, w:, 2] = masked_dish
        
        plt.imshow(comparison)
        plt.title(f"Dish {i+1}: Original vs Masked")
        plt.axis('off')
        
        # Draw a vertical line separating the images
        plt.axvline(x=w, color='white', linewidth=2)
    
    plt.tight_layout()
    
    # Save figures if output directory is specified
    if output_dir:
        plt.figure(1)
        plt.savefig(os.path.join(output_dir, "dish_detection.png"), dpi=150)
        plt.figure(2)
        plt.savefig(os.path.join(output_dir, "dish_masking.png"), dpi=150)
        typer.echo(f"Saved visualization to {output_dir}")
    
    # Show plots if requested
    if show_plots:
        plt.show()
    else:
        plt.close('all')
        
@app.command()
def compare_methods(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    dish_index: int = typer.Option(0, "--dish", "-d", help="Index of the dish to analyze (0-5)"),
    output_path: str = typer.Option(None, "--output", "-o", help="Path to save comparison image")
):
    """Compare different circle detection methods on a single dish"""
    # Load image
    image = Image.open(image_path)
    
    # Detect dish layout
    rows, cols = detect_dish_layout(image)
    
    # Calculate dish dimensions
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    dish_height = height // rows
    dish_width = width // cols
    
    # Extract the specified dish
    row = dish_index // cols
    col = dish_index % cols
    
    top = row * dish_height
    left = col * dish_width
    bottom = top + dish_height
    right = left + dish_width
    
    dish = img_array[top:bottom, left:right].copy()
    
    # Convert to grayscale for processing
    if len(dish.shape) == 3:
        gray = cv2.cvtColor(dish, cv2.COLOR_RGB2GRAY)
    else:
        gray = dish.copy()
    
    # Create figure for comparison
    plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(dish)
    plt.title("Original Dish")
    
    # Method 1: Edge-based Hough Circle detection
    try:
        edges = feature.canny(gray, sigma=3.0)
        edges = morphology.dilation(edges, morphology.disk(2))
        circles = cv2.HoughCircles(
            edges.astype(np.uint8) * 255, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=min(dish_width, dish_height)/2,
            param1=50, 
            param2=30, 
            minRadius=int(min(dish_width, dish_height) * 0.3),
            maxRadius=int(min(dish_width, dish_height) * 0.5)
        )
        
        plt.subplot(3, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        
        plt.subplot(3, 3, 3)
        plt.imshow(dish)
        if circles is not None:
            x, y, r = np.round(circles[0][0]).astype(int)
            circle = Circle((x, y), r, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(circle)
            plt.title(f"Edge Hough (r={r})")
        else:
            plt.title("Edge Hough (Failed)")
    except Exception:
        plt.subplot(3, 3, 2)
        plt.imshow(np.zeros_like(gray), cmap='gray')
        plt.title("Edge Detection (Failed)")
        
        plt.subplot(3, 3, 3)
        plt.imshow(dish)
        plt.title("Edge Hough (Failed)")
    
    # Method 2: Adaptive Threshold
    try:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 101, 5
        )
        
        plt.subplot(3, 3, 4)
        plt.imshow(binary, cmap='gray')
        plt.title("Adaptive Threshold")
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea, default=None)
        
        plt.subplot(3, 3, 5)
        plt.imshow(dish)
        if largest_contour is not None:
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            x, y, radius = int(x), int(y), int(radius)
            circle = Circle((x, y), radius, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(circle)
            plt.title(f"Contour (r={radius})")
        else:
            plt.title("Contour (Failed)")
    except Exception:
        plt.subplot(3, 3, 4)
        plt.imshow(np.zeros_like(gray), cmap='gray')
        plt.title("Adaptive Threshold (Failed)")
        
        plt.subplot(3, 3, 5)
        plt.imshow(dish)
        plt.title("Contour (Failed)")
    
    # Method 3: Direct Hough Circle Transform
    try:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=min(dish_width, dish_height)/2,
            param1=50, 
            param2=30, 
            minRadius=int(min(dish_width, dish_height) * 0.3),
            maxRadius=int(min(dish_width, dish_height) * 0.48)
        )
        
        plt.subplot(3, 3, 6)
        plt.imshow(dish)
        if circles is not None:
            x, y, r = np.round(circles[0][0]).astype(int)
            circle = Circle((x, y), r, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(circle)
            plt.title(f"Direct Hough (r={r})")
        else:
            plt.title("Direct Hough (Failed)")
    except Exception:
        plt.subplot(3, 3, 6)
        plt.imshow(dish)
        plt.title("Direct Hough (Failed)")
    
    # Method 4: OTSU Thresholding
    try:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        plt.subplot(3, 3, 7)
        plt.imshow(binary, cmap='gray')
        plt.title("OTSU Threshold")
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            x, y, radius = int(x), int(y), int(radius)
            
            plt.subplot(3, 3, 8)
            plt.imshow(dish)
            circle = Circle((x, y), radius, fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(circle)
            plt.title(f"OTSU (r={radius})")
        else:
            plt.subplot(3, 3, 8)
            plt.imshow(dish)
            plt.title("OTSU (Failed)")
    except Exception:
        plt.subplot(3, 3, 7)
        plt.imshow(np.zeros_like(gray), cmap='gray')
        plt.title("OTSU Threshold (Failed)")
        
        plt.subplot(3, 3, 8)
        plt.imshow(dish)
        plt.title("OTSU (Failed)")
    
    # Final improved method
    try:
        cx, cy, radius, mask = improved_circular_dish_detection(dish)
        
        plt.subplot(3, 3, 9)
        plt.imshow(dish)
        circle = Circle((cx, cy), radius, fill=False, edgecolor='green', linewidth=3)
        plt.gca().add_patch(circle)
        plt.title(f"IMPROVED (r={radius})")
    except Exception:
        plt.subplot(3, 3, 9)
        plt.imshow(dish)
        plt.title("IMPROVED (Failed)")
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150)
        typer.echo(f"Saved comparison to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    app()