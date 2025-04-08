# /// script
# title = "Copick Spliced Volume Renderer"
# description = "Renders orthogonal views of spliced 3D volumes, combining synthetic and experimental CryoET data."
# author = "Kyle Harrington <czi@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
# keywords = ["tomogram", "visualization", "copick", "cryoet", "synthetic", "splice"]
# classifiers = [
#     "Development Status :: 3 - Alpha",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.9",
#     "Topic :: Scientific/Engineering :: Bio-Informatics",
#     "Topic :: Scientific/Engineering :: Visualization"
# ]
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "copick @ git+https://github.com/copick/copick",
#     "zarr<3",
#     "numcodecs<0.16.0",
#     "tqdm",
#     "scikit-image"
# ]
# ///

"""
Copick Spliced Volume Renderer

Renders orthogonal views of spliced 3D volumes by combining synthetic and experimental CryoET data.
Uses masks from synthetic dataset segmentations to extract intensity data from synthetic tomograms
and insert it into crops from experimental data tomograms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import copick
import zarr
import argparse
from tqdm import tqdm
from skimage import measure
from skimage.transform import resize
from scipy.ndimage import binary_dilation
import random
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_copick_datasets(exp_dataset_id, synth_dataset_id, overlay_root="/tmp/test/"):
    """
    Load the experimental and synthetic CoPick datasets.
    
    Args:
        exp_dataset_id: Dataset ID for the experimental dataset
        synth_dataset_id: Dataset ID for the synthetic dataset
        overlay_root: Root directory for the overlay storage
        
    Returns:
        Tuple of (experimental_root, synthetic_root)
    """
    logger.info(f"Loading experimental dataset {exp_dataset_id} and synthetic dataset {synth_dataset_id}")
    exp_root = copick.from_czcdp_datasets([exp_dataset_id], overlay_root=overlay_root)
    synth_root = copick.from_czcdp_datasets([synth_dataset_id], overlay_root=overlay_root)
    
    logger.info(f"Experimental dataset: {len(exp_root.runs)} runs")
    logger.info(f"Synthetic dataset: {len(synth_root.runs)} runs")
    
    return exp_root, synth_root

def get_available_tomograms(root, voxel_spacing, tomo_type="wbp"):
    """
    Get available tomograms from a CoPick dataset for a specific voxel spacing.
    
    Args:
        root: CoPick root object
        voxel_spacing: Target voxel spacing
        tomo_type: Tomogram type (default: "wbp")
        
    Returns:
        List of tomograms matching the criteria
    """
    available_tomograms = []
    
    for run in root.runs:
        # Get the closest voxel spacing to the target
        closest_vs = None
        min_diff = float('inf')
        
        for vs in run.voxel_spacings:
            diff = abs(vs.meta.voxel_size - voxel_spacing)
            if diff < min_diff:
                min_diff = diff
                closest_vs = vs
        
        if closest_vs:
            tomograms = closest_vs.get_tomograms(tomo_type)
            if tomograms:
                available_tomograms.extend(tomograms)
                logger.info(f"Found {len(tomograms)} tomograms in run {run.meta.name} with voxel spacing {closest_vs.meta.voxel_size}")
    
    return available_tomograms

def get_segmentation_masks(root, voxel_spacing, pickable_objects=None):
    """
    Get segmentation masks from a CoPick dataset for a specific voxel spacing.
    
    Args:
        root: CoPick root object
        voxel_spacing: Target voxel spacing
        pickable_objects: List of pickable object names to filter by (default: None, all objects)
        
    Returns:
        Dictionary mapping segmentation names to segmentation objects
    """
    segmentation_masks = {}
    
    for run in root.runs:
        # Get the closest voxel spacing to the target
        closest_vs = None
        min_diff = float('inf')
        
        for vs in run.voxel_spacings:
            diff = abs(vs.meta.voxel_size - voxel_spacing)
            if diff < min_diff:
                min_diff = diff
                closest_vs = vs
        
        if closest_vs:
            segmentations = run.get_segmentations(voxel_size=closest_vs.meta.voxel_size)
            
            for seg in segmentations:
                # Only include segmentations matching requested pickable objects
                if pickable_objects is None or seg.meta.name in pickable_objects:
                    segmentation_masks[seg.meta.name] = seg
                    logger.info(f"Found segmentation mask for '{seg.meta.name}' in run {run.meta.name}")
    
    return segmentation_masks

def extract_bounding_boxes(mask_data, min_size=100, box_size=48):
    """
    Extract bounding boxes for all connected components in a segmentation mask.
    Uses a constant box size and ensures the particle is centered.
    
    Args:
        mask_data: 3D segmentation mask array
        min_size: Minimum size in voxels to consider a component
        box_size: Size of the cubic bounding box (default: 48)
        
    Returns:
        List of dictionaries with bounding box information
    """
    # Label connected components
    labels = measure.label(mask_data > 0)
    regions = measure.regionprops(labels)
    
    # Log some information about the mask
    logger.info(f"Mask data shape: {mask_data.shape}, min: {np.min(mask_data)}, max: {np.max(mask_data)}")
    logger.info(f"Found {len(regions)} regions in the mask")
    
    # Extract bounding boxes
    bounding_boxes = []
    for region in regions:
        if region.area >= min_size:
            # Get the centroid of the region
            z_center, y_center, x_center = region.centroid
            logger.info(f"Processing region with center ({z_center}, {y_center}, {x_center}) and area {region.area}")
            
            # Calculate box boundaries centered on the particle
            half_size = box_size // 2
            
            z_min = max(0, int(z_center - half_size))
            y_min = max(0, int(y_center - half_size))
            x_min = max(0, int(x_center - half_size))
            
            # Adjust if box would go beyond bounds
            if z_min + box_size > mask_data.shape[0]:
                z_min = max(0, mask_data.shape[0] - box_size)
            if y_min + box_size > mask_data.shape[1]:
                y_min = max(0, mask_data.shape[1] - box_size)
            if x_min + box_size > mask_data.shape[2]:
                x_min = max(0, mask_data.shape[2] - box_size)
            
            # Calculate max coordinates
            z_max = min(mask_data.shape[0], z_min + box_size)
            y_max = min(mask_data.shape[1], y_min + box_size)
            x_max = min(mask_data.shape[2], x_min + box_size)
            
            # Log the chosen box coordinates
            logger.info(f"Box coordinates: ({z_min}, {y_min}, {x_min}) to ({z_max}, {y_max}, {x_max})")
            
            # Check if we can extract a full box
            if (z_max - z_min) != box_size or (y_max - y_min) != box_size or (x_max - x_min) != box_size:
                logger.warning(f"Cannot extract a full {box_size}^3 box at the edge of the volume. "
                              f"Got size {(z_max - z_min, y_max - y_min, x_max - x_min)}")
                continue
            
            # Create a mask for this specific region
            region_mask = np.zeros(mask_data.shape, dtype=bool)
            region_mask[labels == region.label] = True
            
            # Dilate the mask slightly for smoother boundaries
            dilated_mask = binary_dilation(region_mask, iterations=2)
            
            # Extract the fixed-size box from the mask
            box_mask = dilated_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            
            # Verify box mask has expected dimensions
            if box_mask.shape != (box_size, box_size, box_size):
                logger.warning(f"Box mask has unexpected shape: {box_mask.shape}")
                continue
            
            logger.info(f"Final box mask shape: {box_mask.shape}, non-zero elements: {np.count_nonzero(box_mask)}")
            
            bounding_boxes.append({
                'bbox': (z_min, y_min, x_min, z_max, y_max, x_max),
                'region_mask': box_mask,
                'center': region.centroid,
                'size': region.area
            })
    
    # Sort by size (largest first)
    bounding_boxes.sort(key=lambda x: x['size'], reverse=True)
    
    logger.info(f"Generated {len(bounding_boxes)} valid bounding boxes")
    return bounding_boxes
    
    # Sort by size (largest first)
    bounding_boxes.sort(key=lambda x: x['size'], reverse=True)
    
    return bounding_boxes

def extract_random_crop(tomogram_data, crop_size):
    """
    Extract a random crop from a tomogram.
    
    Args:
        tomogram_data: 3D tomogram data array
        crop_size: Tuple of (depth, height, width) for the crop
        
    Returns:
        Cropped tomogram data
    """
    depth, height, width = tomogram_data.shape
    
    # Ensure crop sizes don't exceed tomogram dimensions
    crop_depth = min(crop_size[0], depth)
    crop_height = min(crop_size[1], height)
    crop_width = min(crop_size[2], width)
    
    # Calculate valid ranges for the random crop
    max_z = depth - crop_depth
    max_y = height - crop_height
    max_x = width - crop_width
    
    if max_z <= 0 or max_y <= 0 or max_x <= 0:
        # Tomogram is smaller than crop size in at least one dimension
        return resize(tomogram_data, crop_size, mode='reflect', anti_aliasing=True)
    
    # Get random start coordinates
    z_start = random.randint(0, max_z)
    y_start = random.randint(0, max_y)
    x_start = random.randint(0, max_x)
    
    # Extract the crop
    crop = tomogram_data[
        z_start:z_start+crop_depth,
        y_start:y_start+crop_height,
        x_start:x_start+crop_width
    ]
    
    return crop

def splice_volumes(synthetic_tomogram, synthetic_mask, exp_tomogram, bbox_info, blend_sigma=2.0):
    """
    Splice a synthetic structure into an experimental tomogram.
    
    Args:
        synthetic_tomogram: Synthetic tomogram data
        synthetic_mask: Binary mask for the structure to extract
        exp_tomogram: Experimental tomogram to splice into
        bbox_info: Dictionary with bounding box information
        blend_sigma: Sigma for Gaussian blending at boundaries
        
    Returns:
        Spliced tomogram with synthetic structure inserted into experimental data
    """
    # Extract bounding box coordinates
    z_min, y_min, x_min, z_max, y_max, x_max = bbox_info['bbox']
    region_mask = bbox_info['region_mask']
    
    logger.info(f"Original bbox coords: z({z_min}:{z_max}), y({y_min}:{y_max}), x({x_min}:{x_max})")
    
    # Try different coordinate orders to fix possible axis swaps
    # The original order is (z, y, x) but we'll try different permutations
    coordinate_orders = [
        # Original order (z, y, x)
        (z_min, y_min, x_min, z_max, y_max, x_max),
        # x and z swapped (x, y, z)
        (x_min, y_min, z_min, x_max, y_max, z_max),
        # y and z swapped (y, z, x)
        (y_min, z_min, x_min, y_max, z_max, x_max),
    ]
    
    best_synth_region = None
    best_match_score = -float('inf')
    best_coords = None
    best_order_name = None
    
    # Test each coordinate order and find the best match
    for idx, (min1, min2, min3, max1, max2, max3) in enumerate(coordinate_orders):
        order_name = ['(z,y,x)', '(x,y,z)', '(y,z,x)'][idx]
        logger.info(f"Trying coordinate order {order_name}: ({min1}:{max1}, {min2}:{max2}, {min3}:{max3})")
        
        try:
            # Check if the coordinates are valid
            if min1 < 0 or min2 < 0 or min3 < 0 or \
               max1 > synthetic_tomogram.shape[0] or \
               max2 > synthetic_tomogram.shape[1] or \
               max3 > synthetic_tomogram.shape[2]:
                logger.warning(f"Coordinates out of bounds for order {order_name}")
                continue
                
            # Extract region with this coordinate order
            synth_region = synthetic_tomogram[min1:max1, min2:max2, min3:max3].copy()
            
            # Skip if shapes don't match
            if synth_region.shape != region_mask.shape:
                logger.warning(f"Shape mismatch for order {order_name}: {synth_region.shape} vs {region_mask.shape}")
                continue
            
            # Compute a match score (correlation between mask and synthetic data)
            masked_synth = synth_region * region_mask.astype(float)
            match_score = np.mean(masked_synth) / (np.std(masked_synth) + 1e-8)
            
            logger.info(f"Order {order_name} - Shape: {synth_region.shape}, Score: {match_score:.4f}")
            
            # Update best match if this is better
            if match_score > best_match_score:
                best_match_score = match_score
                best_synth_region = synth_region
                best_coords = (min1, min2, min3, max1, max2, max3)
                best_order_name = order_name
                
        except Exception as e:
            logger.error(f"Error with coordinate order {order_name}: {e}")
    
    # If we couldn't find a good match, use a fallback strategy
    if best_synth_region is None:
        logger.warning("No valid coordinate ordering found, using fallback strategy")
        # Extract center region from synthetic tomogram
        sz, sy, sx = synthetic_tomogram.shape
        center_z, center_y, center_x = sz//2, sy//2, sx//2
        box_size = region_mask.shape[0]
        half_size = box_size // 2
        
        # Extract a region from the center of the tomogram
        safe_z_min = max(0, center_z - half_size)
        safe_z_max = min(sz, center_z + half_size)
        safe_y_min = max(0, center_y - half_size)
        safe_y_max = min(sy, center_y + half_size)
        safe_x_min = max(0, center_x - half_size)
        safe_x_max = min(sx, center_x + half_size)
        
        best_synth_region = synthetic_tomogram[safe_z_min:safe_z_max, safe_y_min:safe_y_max, safe_x_min:safe_x_max].copy()
        best_coords = (safe_z_min, safe_y_min, safe_x_min, safe_z_max, safe_y_max, safe_x_max)
        best_order_name = "fallback"
        
        # Resize to match mask if needed
        if best_synth_region.shape != region_mask.shape:
            best_synth_region = resize(best_synth_region, region_mask.shape, mode='reflect', anti_aliasing=True)
    
    logger.info(f"Selected coordinate order: {best_order_name} with coords {best_coords}")
    logger.info(f"Final synthetic region shape: {best_synth_region.shape}, mask shape: {region_mask.shape}")
    
    # Extract corresponding region from experimental tomogram with matching shape
    exp_crop = extract_random_crop(exp_tomogram, region_mask.shape)
    
    # Create a spliced volume by starting with the experimental crop
    spliced_volume = exp_crop.copy()
    
    # Create a debug synthetic region that only shows the masked parts
    masked_synth = best_synth_region.copy()
    masked_synth[~region_mask] = 0  # Zero out non-mask areas for debug visualization
    
    # Save the masked synthetic region for debug purposes
    debug_mask_only = np.zeros_like(masked_synth)
    debug_mask_only[region_mask] = best_synth_region[region_mask]
    
    # Replace the masked region with synthetic data
    spliced_volume[region_mask] = best_synth_region[region_mask]
    
    # Apply Gaussian weight blending at the boundary for smoother transition
    if blend_sigma > 0:
        try:
            # Create a weight map that transitions smoothly across the boundary
            from scipy.ndimage import gaussian_filter
            weight_map = gaussian_filter(region_mask.astype(float), sigma=blend_sigma)
            weight_map = np.clip(weight_map, 0, 1)
            
            # Blend the synthetic and experimental data
            blended = best_synth_region * weight_map + exp_crop * (1 - weight_map)
            spliced_volume = blended
        except Exception as e:
            logger.error(f"Error during Gaussian blending: {e}")
            # Fallback to the non-blended result
    
    return spliced_volume, {
        'exp_crop': exp_crop,
        'synth_region': best_synth_region,
        'mask': region_mask,
        'mask_only': debug_mask_only,  # Add debug data
        'masked_synth': masked_synth,    # Add debug data
        'coord_order': best_order_name  # Track which coordinate order worked best
    }

def render_orthogonal_views(volume_data, title=None, savepath=None, colormap='viridis'):
    """
    Render and save orthogonal views of a 3D volume.
    
    Args:
        volume_data: 3D volume data array
        title: Optional title for the plot
        savepath: Path to save the rendered image
        colormap: Matplotlib colormap to use
    """
    # Calculate central slices
    z_mid = volume_data.shape[0] // 2
    y_mid = volume_data.shape[1] // 2
    x_mid = volume_data.shape[2] // 2
    
    # Calculate maximum intensity projections along each axis
    max_proj_z = np.max(volume_data, axis=0)
    max_proj_y = np.max(volume_data, axis=1)
    max_proj_x = np.max(volume_data, axis=2)
    
    # Create figure with 2×3 layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Determine global min/max for consistent color scaling
    vmin = np.min(volume_data)
    vmax = np.max(volume_data)
    
    # Plot central slices in first row
    axes[0, 0].imshow(volume_data[z_mid, :, :], cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Central Z Slice (z={z_mid})')
    
    axes[0, 1].imshow(volume_data[:, y_mid, :], cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Central Y Slice (y={y_mid})')
    
    axes[0, 2].imshow(volume_data[:, :, x_mid], cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'Central X Slice (x={x_mid})')
    
    # Plot maximum projections in second row
    axes[1, 0].imshow(max_proj_z, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1, 0].set_title('Maximum Z Projection')
    
    axes[1, 1].imshow(max_proj_y, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Maximum Y Projection')
    
    axes[1, 2].imshow(max_proj_x, cmap=colormap, vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('Maximum X Projection')
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if a path is provided
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def render_comparison_views(original, spliced, metadata, title=None, savepath=None, colormap='viridis'):
    """
    Render comparison views between original experimental data and spliced volume.
    
    Args:
        original: Original experimental volume
        spliced: Spliced volume with synthetic structure
        metadata: Dictionary with additional data (mask, etc.)
        title: Optional title for the plot
        savepath: Path to save the rendered image
        colormap: Matplotlib colormap to use
    """
    # Create a 4×3 grid for more detailed comparisons
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    # Get central slices
    z_mid = original.shape[0] // 2
    
    # Determine global min/max for consistent color scaling
    vmin = min(np.min(original), np.min(spliced))
    vmax = max(np.max(original), np.max(spliced))
    
    # Add coordinate order information to title if available
    if 'coord_order' in metadata:
        coord_order = metadata['coord_order']
        if title:
            title = f"{title} (Coords: {coord_order})"
        else:
            title = f"Spliced Volume (Coords: {coord_order})"
    
    # Row 1: Experimental data
    axes[0, 0].imshow(original[z_mid, :, :], cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Experimental (z={z_mid})')
    
    axes[0, 1].imshow(np.max(original, axis=0), cmap=colormap, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('Experimental (Max Z Proj)')
    
    # Show the mask in the third column
    if 'mask' in metadata:
        mask_slice = metadata['mask'][z_mid, :, :]
        axes[0, 2].imshow(mask_slice, cmap='gray')
        axes[0, 2].set_title('Structure Mask (Slice)')
    else:
        # If no mask, show exp data from another angle
        axes[0, 2].imshow(np.max(original, axis=1), cmap=colormap, vmin=vmin, vmax=vmax)
        axes[0, 2].set_title('Experimental (Max Y Proj)')
    
    # Row 2: Synthetic region
    if 'synth_region' in metadata:
        synth_region = metadata['synth_region']
        axes[1, 0].imshow(synth_region[z_mid, :, :], cmap=colormap, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'Synthetic (z={z_mid})')
        
        axes[1, 1].imshow(np.max(synth_region, axis=0), cmap=colormap, vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Synthetic (Max Z Proj)')
        
        # Show masked synthetic data in third column
        if 'mask' in metadata:
            mask = metadata['mask']
            axes[1, 2].imshow(np.max(mask, axis=0), cmap='gray')
            axes[1, 2].set_title('Mask (Max Z Proj)')
        else:
            axes[1, 2].imshow(np.max(synth_region, axis=1), cmap=colormap, vmin=vmin, vmax=vmax)
            axes[1, 2].set_title('Synthetic (Max Y Proj)')
    
    # Row 3: Synthetic data with mask applied
    if 'masked_synth' in metadata:
        masked_synth = metadata['masked_synth']
        axes[2, 0].imshow(masked_synth[z_mid, :, :], cmap=colormap)
        axes[2, 0].set_title(f'Masked Synthetic (z={z_mid})')
        
        # Sum projection of the masked synthetic region
        axes[2, 1].imshow(np.sum(masked_synth, axis=0), cmap=colormap)
        axes[2, 1].set_title('Masked Synthetic (Sum Z Proj)')
        
        # Max projection of the masked synthetic region
        axes[2, 2].imshow(np.max(masked_synth, axis=0), cmap=colormap)
        axes[2, 2].set_title('Masked Synthetic (Max Z Proj)')
    elif 'mask_only' in metadata:
        # If masked_synth isn't available but mask_only is
        mask_only = metadata['mask_only']
        axes[2, 0].imshow(mask_only[z_mid, :, :], cmap=colormap)
        axes[2, 0].set_title(f'Mask Only (z={z_mid})')
        
        axes[2, 1].imshow(np.sum(mask_only, axis=0), cmap=colormap)
        axes[2, 1].set_title('Mask Only (Sum Z Proj)')
        
        axes[2, 2].imshow(np.max(mask_only, axis=0), cmap=colormap)
        axes[2, 2].set_title('Mask Only (Max Z Proj)')
    else:
        # If neither is available, show empty plots
        for i in range(3):
            axes[2, i].axis('off')
    
    # Row 4: Spliced result
    axes[3, 0].imshow(spliced[z_mid, :, :], cmap=colormap, vmin=vmin, vmax=vmax)
    axes[3, 0].set_title(f'Spliced Result (z={z_mid})')
    
    axes[3, 1].imshow(np.max(spliced, axis=0), cmap=colormap, vmin=vmin, vmax=vmax)
    axes[3, 1].set_title('Spliced Result (Max Z Proj)')
    
    axes[3, 2].imshow(np.max(spliced, axis=1), cmap=colormap, vmin=vmin, vmax=vmax)
    axes[3, 2].set_title('Spliced Result (Max Y Proj)')
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    
    # Save figure if a path is provided
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the experimental and synthetic datasets
    exp_root, synth_root = load_copick_datasets(
        args.exp_dataset_id, 
        args.synth_dataset_id,
        overlay_root=args.overlay_root
    )
    
    # Get tomograms from both datasets
    exp_tomograms = get_available_tomograms(exp_root, args.voxel_spacing, args.tomo_type)
    synth_tomograms = get_available_tomograms(synth_root, args.voxel_spacing, args.tomo_type)
    
    if not exp_tomograms:
        raise ValueError(f"No experimental tomograms found with voxel spacing {args.voxel_spacing}")
    
    if not synth_tomograms:
        raise ValueError(f"No synthetic tomograms found with voxel spacing {args.voxel_spacing}")
    
    # Get segmentation masks from synthetic dataset
    logger.info("Getting segmentation masks from synthetic dataset")
    segmentation_masks = get_segmentation_masks(synth_root, args.voxel_spacing)
    
    if not segmentation_masks:
        raise ValueError(f"No segmentation masks found in synthetic dataset with voxel spacing {args.voxel_spacing}")
    
    # Process each segmentation mask to extract structures and splice them into experimental data
    results = []
    
    # Limit to the specified number of examples
    num_examples = min(args.num_examples, len(segmentation_masks))
    
    # Select random segmentation masks if there are more than we need
    selected_masks = list(segmentation_masks.items())
    if len(selected_masks) > num_examples:
        selected_masks = random.sample(selected_masks, num_examples)
    
    for mask_name, mask_obj in tqdm(selected_masks, desc="Processing masks"):
        logger.info(f"Processing mask: {mask_name}")
        
        # Skip membrane segmentation masks
        if mask_name.lower() == "membrane":
            logger.info(f"Skipping membrane segmentation mask: {mask_name}")
            continue
        
        # Access the mask data
        mask_zarr = zarr.open(mask_obj.zarr(), "r")
        mask_data = mask_zarr["data" if "data" in mask_zarr else "0"][:]
        
        # Extract bounding boxes for structures in the mask
        bboxes = extract_bounding_boxes(mask_data, min_size=args.min_structure_size, box_size=args.box_size)
        
        if not bboxes:
            logger.warning(f"No structures found in mask {mask_name} larger than {args.min_structure_size} voxels")
            continue
        
        # Select a random synthetic tomogram
        synth_tomogram_obj = random.choice(synth_tomograms)
        synth_zarr = zarr.open(synth_tomogram_obj.zarr(), "r")
        synth_data = synth_zarr["0"][:]
        logger.info(f"Synthetic tomogram shape: {synth_data.shape}")
        
        # Select a random experimental tomogram
        exp_tomogram_obj = random.choice(exp_tomograms)
        exp_zarr = zarr.open(exp_tomogram_obj.zarr(), "r")
        exp_data = exp_zarr["0"][:]
        logger.info(f"Experimental tomogram shape: {exp_data.shape}")
        
        # Normalize tomogram data
        synth_data = (synth_data - np.mean(synth_data)) / np.std(synth_data)
        exp_data = (exp_data - np.mean(exp_data)) / np.std(exp_data)
        logger.info(f"Normalized synthetic tomogram range: [{np.min(synth_data):.2f}, {np.max(synth_data):.2f}]")
        logger.info(f"Normalized experimental tomogram range: [{np.min(exp_data):.2f}, {np.max(exp_data):.2f}]")
        
        # Process each bounding box
        for i, bbox in enumerate(bboxes):
            # Only process a limited number of structures per mask
            if i >= args.structures_per_mask:
                break
            
            # Splice the structure into experimental data
            spliced_volume, metadata = splice_volumes(
                synth_data, mask_data, exp_data, bbox, blend_sigma=args.blend_sigma
            )
            
            # Save the result information
            result_info = {
                'mask_name': mask_name,
                'bbox_idx': i,
                'bbox_center': bbox['center'],
                'spliced_volume': spliced_volume,
                'metadata': metadata
            }
            
            results.append(result_info)
            
            # Render and save comparison views
            title = f"Spliced Structure: {mask_name} (Structure {i+1})"
            savepath = output_dir / f"{mask_name}_structure_{i+1}_comparison.png"
            
            render_comparison_views(
                metadata['exp_crop'],
                spliced_volume,
                metadata,
                title=title,
                savepath=savepath,
                colormap=args.colormap
            )
            
            # Optionally, save the volumes for further analysis
            if args.save_volumes:
                # Save spliced volume
                np.save(output_dir / f"{mask_name}_structure_{i+1}_spliced.npy", spliced_volume)
                
                # Save original experimental crop
                np.save(output_dir / f"{mask_name}_structure_{i+1}_experimental.npy", metadata['exp_crop'])
                
                # Save synthetic region
                np.save(output_dir / f"{mask_name}_structure_{i+1}_synthetic.npy", metadata['synth_region'])
                
                # Save mask
                np.save(output_dir / f"{mask_name}_structure_{i+1}_mask.npy", metadata['mask'])
    
    # Create summary HTML file with links to all images
    if results:
        logger.info(f"Created {len(results)} spliced volumes")
        logger.info(f"Results saved to {output_dir}")
        
        # Create summary HTML file with links to all images
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spliced Volume Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .result {{ 
                    margin: 20px 0; 
                    padding: 15px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }}
                img {{ max-width: 100%; height: auto; }}
                .metadata {{ 
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-family: monospace;
                }}
            </style>
        </head>
        <body>
            <h1>Spliced Volume Results</h1>
            <p>Generated {len(results)} spliced volumes from synthetic dataset {args.synth_dataset_id} to experimental dataset {args.exp_dataset_id}</p>
        """
        
        for i, result in enumerate(results):
            img_path = f"{result['mask_name']}_structure_{result['bbox_idx']+1}_comparison.png"
            center_coords = ", ".join([f"{c:.1f}" for c in result['bbox_center']])
            
            html_content += f"""
            <div class="result">
                <h2>Result {i+1}: {result['mask_name']} (Structure {result['bbox_idx']+1})</h2>
                <div class="metadata">
                    <p>Center coordinates: ({center_coords})</p>
                </div>
                <img src="{img_path}" alt="Comparison view">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_dir / "results.html", "w") as f:
            f.write(html_content)
        
        logger.info(f"Summary HTML saved to {output_dir / 'results.html'}")
    else:
        logger.warning("No results were generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render orthogonal views of spliced 3D volumes from CryoET data")
    
    # Dataset parameters
    parser.add_argument("--exp-dataset-id", type=int, default=10440,
                        help="Dataset ID for experimental data")
    parser.add_argument("--synth-dataset-id", type=int, default=10441,
                        help="Dataset ID for synthetic data with segmentation masks")
    parser.add_argument("--overlay-root", type=str, default="/tmp/test/",
                        help="Root directory for overlay storage")
    
    # Volume parameters
    parser.add_argument("--voxel-spacing", type=float, default=10.0,
                        help="Target voxel spacing for tomograms")
    parser.add_argument("--tomo-type", type=str, default="wbp",
                        help="Tomogram type to use")
    
    # Processing parameters
    parser.add_argument("--num-examples", type=int, default=3,
                        help="Number of example pairs to create")
    parser.add_argument("--structures-per-mask", type=int, default=1,
                        help="Number of structures to extract per mask")
    parser.add_argument("--min-structure-size", type=int, default=500,
                        help="Minimum structure size in voxels")
    parser.add_argument("--box-size", type=int, default=48,
                        help="Size of the cubic box for extracting structures (default: 48)")
    parser.add_argument("--blend-sigma", type=float, default=2.0,
                        help="Sigma for Gaussian blending at boundaries")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="./spliced_volumes",
                        help="Directory to save output files")
    parser.add_argument("--colormap", type=str, default="viridis",
                        help="Matplotlib colormap for rendering")
    parser.add_argument("--save-volumes", action="store_true",
                        help="Save volume data as numpy arrays")
    
    args = parser.parse_args()
    
    main(args)
