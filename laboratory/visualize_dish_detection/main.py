# /// script
# title = "Visualize Physarum Petri Dish Detection"
# description = "Demonstration tool for visualizing the circular dish detection used in physarum zarr conversion"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
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
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

app = typer.Typer()

def detect_dish_layout(image: Image.Image) -> tuple[int, int]:
    """
    Analyze image to determine dish layout (2x3 or 3x2)
    This should be customized based on your specific setup
    """
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

def extract_dishes(image: Image.Image, rows: int = 2, cols: int = 3) -> tuple[list[np.ndarray], list[tuple[int, int, int, np.ndarray]]]:
    """Extract individual dishes from the composite image"""
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
            
            # Detect the circular dish and create a mask
            cx, cy, radius, mask = detect_circular_dish(dish)
            
            dishes.append(dish)
            dish_metadata.append((cx, cy, radius, mask))
    
    return dishes, dish_metadata

@app.command()
def visualize(
    image_path: str = typer.Argument(..., help="Path to input image with petri dishes"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Directory to save visualization (optional)"),
    show_plots: bool = typer.Option(True, "--show/--no-show", help="Show visualization plots")
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
                cx, cy, radius, mask = dish_metadata[row * cols + col]
                
                # Display the dish
                plt.imshow(dish)
                plt.title(f"Dish {row * cols + col + 1}")
                
                # Add circle overlay
                circle = Circle((cx, cy), radius, fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(circle)
    
    plt.tight_layout()
    
    # Another figure to show the masking effect
    plt.figure(figsize=(18, 10))
    
    for i, (dish, (cx, cy, radius, mask)) in enumerate(zip(dishes, dish_metadata)):
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
        
        # Side by side comparison
        comparison = np.hstack([dish, masked_dish])
        plt.imshow(comparison)
        plt.title(f"Dish {i+1}: Original vs Masked")
        plt.axis('off')
    
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

if __name__ == "__main__":
    app()