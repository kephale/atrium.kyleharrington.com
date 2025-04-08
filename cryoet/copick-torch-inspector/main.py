# /// script
# title = "Copick Tomogram Visualization Server"
# description = "A FastAPI server that extends copick-server to provide visualization of tomogram samples."
# author = "Kyle Harrington <czi@kyleharrington.com>"
# license = "MIT"
# version = "0.0.7"
# keywords = ["tomogram", "visualization", "fastapi", "copick", "server"]
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
#     "fastapi",
#     "uvicorn",
#     "zarr<3",
#     "numcodecs<0.16.0",  
#     "copick>=0.8.0",
#     "copick-torch @ git+https://github.com/copick/copick-torch.git",
#     "copick-server @ git+https://github.com/copick/copick-server.git"
# ]
# ///

"""
Copick Tomogram Visualization Server

A FastAPI server that extends copick-server to provide visualization of tomogram samples.
Displays central slices and average projections along all axes for tomogram samples.
"""

import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn
import threading
import tempfile
import json
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Import from copick-server
from copick_server.server import CopickRoute
from fastapi.middleware.cors import CORSMiddleware
import copick

# Import from copick-torch
import copick_torch
from copick_torch.copick import CopickDataset
from copick_torch import SimpleCopickDataset, ClassBalancedSampler, MixupAugmentation
from torch.utils.data import DataLoader

# Port configuration - define once and reuse
PORT = 8018
HOST = "0.0.0.0"

# Create a new FastAPI app and add our custom routes first
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/tomogram-viz", response_class=HTMLResponse)
async def visualize_tomograms(
    dataset_id: int = Query(..., description="Dataset ID"),
    overlay_root: str = Query("/tmp/test/", description="Overlay root directory"),
    run_name: Optional[str] = Query(None, description="Run name to visualize (if None, uses first run)"),
    voxel_spacing: Optional[float] = Query(None, description="Voxel spacing to use (if None, uses first available)"),
    tomo_type: str = Query("wbp", description="Tomogram type (e.g., 'wbp')"),
    batch_size: int = Query(25, description="Number of samples to visualize"),
    box_size: int = Query(64, description="Box size for subvolume extraction"),
    slice_colormap: str = Query("gray", description="Colormap for slices"),
    projection_colormap: str = Query("viridis", description="Colormap for projections"),
    augment: bool = Query(True, description="Apply augmentations"),
    show_augmentations: bool = Query(True, description="Show augmentation stages"),
    use_balanced_sampling: bool = Query(True, description="Use class balanced sampling"),
    background_ratio: float = Query(0.2, description="Ratio of background samples"),
    mixup_alpha: float = Query(0.2, description="Alpha parameter for mixup augmentation")
):
    """
    Visualize tomogram samples from a CoPick dataset, showing central slices and average projections
    along all axes.
    
    Args:
        dataset_id: Dataset ID from CZ cryoET Data Portal
        overlay_root: Root directory for the overlay storage
        run_name: Name of the run to visualize
        voxel_spacing: Voxel spacing to use
        tomo_type: Tomogram type (e.g., 'wbp')
        batch_size: Number of samples to visualize
        box_size: Box size for subvolume extraction
        slice_colormap: Matplotlib colormap for slices
        projection_colormap: Matplotlib colormap for projections
    
    Returns:
        HTML page with visualizations
    """
    try:
        # Create a temporary config file for the dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            config_path = temp_file.name
            
            # Load the dataset using the copick API
            root = copick.from_czcdp_datasets([dataset_id], overlay_root=overlay_root)
            
            # Save the config to a file for CopickDataset to use
            # config_data = root._config.model_dump(exclude_unset=True)
            # json.dump(config_data, temp_file, indent=4)
        
        # Get available runs
        runs = root.runs
        if not runs:
            raise HTTPException(status_code=404, detail="No runs found in the dataset")
        
        # Select the run to use
        selected_run = None
        if run_name:
            for run in runs:
                if str(run.meta.name) == run_name:
                    selected_run = run
                    break
            if selected_run is None:
                raise HTTPException(status_code=404, detail=f"Run {run_name} not found in the dataset")
        else:
            selected_run = runs[0]
            run_name = str(selected_run.meta.name)
        
        # Get available voxel spacings
        voxel_spacings = [vs.meta.voxel_size for vs in selected_run.voxel_spacings]
        if not voxel_spacings:
            raise HTTPException(status_code=404, detail=f"No voxel spacings found for run {run_name}")
        
        # Select the voxel spacing to use
        if voxel_spacing is None:
            voxel_spacing = voxel_spacings[0]
        elif voxel_spacing not in voxel_spacings:
            # Find the closest voxel spacing
            voxel_spacing = min(voxel_spacings, key=lambda vs: abs(vs - voxel_spacing))
        
        # Get the tomograms for the selected run and voxel spacing
        voxel_spacing_obj = selected_run.get_voxel_spacing(voxel_spacing)
        tomograms = voxel_spacing_obj.get_tomograms(tomo_type)
        if not tomograms:
            raise HTTPException(status_code=404, detail=f"No tomograms found for run {run_name} with voxel spacing {voxel_spacing} and type {tomo_type}")
        
        # Register all available object classes from the picks
        available_objects = set()
        for run in root.runs:
            for picks in run.picks:
                available_objects.add(picks.pickable_object_name)
        
        print(f"Available object classes: {', '.join(sorted(available_objects))}")
        
        # Create the dataset - capture and print any warnings during creation
        import warnings
        original_warn = warnings.warn
        warning_messages = []
        
        def capture_warning(message, *args, **kwargs):
            warning_messages.append(str(message))
            original_warn(message, *args, **kwargs)
        
        warnings.warn = capture_warning
        
        try:
            dataset = SimpleCopickDataset(
                copick_root=root,
                boxsize=(box_size, box_size, box_size),
                voxel_spacing=voxel_spacing,
                augment=augment,
                include_background=True,
                background_ratio=background_ratio,
                cache_dir="copick_torch_demo_cache"
            )
            
            # Verify dataset integrity
            # Check a sample to see if indexing works as expected
            if len(dataset) > 0:
                print(f"Dataset has {len(dataset)} samples")
                # Check several samples to see if class indices are consistent
                for sample_idx in range(min(10, len(dataset))):
                    try:
                        sample, label = dataset[sample_idx]
                        print(f"Sample {sample_idx} - Shape: {sample.shape}")
                        
                        # Check if label makes sense
                        if isinstance(label, dict):
                            class_idx = label['class_idx']
                        else:
                            class_idx = label.item() if hasattr(label, 'item') else label
                        
                        # Get class name for this index
                        if class_idx == -1:
                            class_name = "background"
                        elif class_idx >= 0 and class_idx < len(dataset.keys()):
                            class_name = dataset.keys()[class_idx]
                        else:
                            class_name = f"unknown-class-{class_idx}"
                        
                        print(f"Sample {sample_idx} - Class index: {class_idx}, Class name: {class_name}")
                    except Exception as e:
                        print(f"Error inspecting dataset sample {sample_idx}: {str(e)}")
            else:
                print("Warning: Dataset is empty!")
                
        finally:
            warnings.warn = original_warn
            
        if warning_messages:
            print("Warnings during dataset creation:")
            for msg in warning_messages:
                print(f" - {msg}")
        
        # Force all object classes to be registered in the dataset
        # This ensures all classes appear in the visualization, even if subvolume extraction fails
        for obj_name in available_objects:
            if obj_name not in dataset._keys:
                dataset._keys.append(obj_name)
                print(f"Added missing class: {obj_name}")
        
        # Remove duplicate keys (which can happen if the dataset wasn't cleaned properly)
        dataset._keys = list(dict.fromkeys(dataset._keys))
        
        # Log the full list of registered keys for debugging
        print(f"Full list of registered class keys ({len(dataset._keys)} unique classes): {dataset._keys}")
        
        # Create a dataloader with optional class balanced sampling
        if use_balanced_sampling:
            # Create class-balanced sampler
            labels = [dataset[i][1] for i in range(len(dataset))]
            # Print label distribution for debugging
            label_counts = {}
            class_name_counts = {}
            for i, label in enumerate(labels):
                if isinstance(label, dict):
                    label_val = label['class_idx']
                else:
                    label_val = label.item() if hasattr(label, 'item') else label
                
                # Count by numeric index
                if label_val not in label_counts:
                    label_counts[label_val] = 0
                label_counts[label_val] += 1
                
                # Also count by class name for better debugging
                if label_val == -1:
                    class_name = "background"
                elif label_val >= 0 and label_val < len(dataset.keys()):
                    class_name = dataset.keys()[label_val]
                else:
                    class_name = f"unknown-class-{label_val}"
                
                if class_name not in class_name_counts:
                    class_name_counts[class_name] = 0
                class_name_counts[class_name] += 1
                
                # Print every 100th sample for debugging
                if i % 100 == 0:
                    print(f"Sample {i} - Class index: {label_val}, Class name: {class_name}")
            
            print(f"Label distribution by index before sampling: {label_counts}")
            print(f"Label distribution by class name before sampling: {class_name_counts}")
            
            sampler = ClassBalancedSampler(
                labels=labels,
                num_samples=len(dataset),
                replacement=True
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create mixup augmentation for visualization if requested
        mixup = MixupAugmentation(alpha=mixup_alpha) if augment and show_augmentations else None
        
        # Generate visualization for each sample
        images_html = []
        
        try:
            # Try to get one batch
            batch = next(iter(dataloader))
            
            # If showing augmentation stages, apply mixup to half the batch
            aug_samples = None
            if show_augmentations and mixup is not None:
                # Create a custom colormap for distinguishing original vs augmented
                orig_cmap = plt.cm.get_cmap(slice_colormap)
                aug_cmap = LinearSegmentedColormap.from_list(
                    'aug_' + slice_colormap, 
                    [(0, (0, 0.5, 0.5)), (0.5, (0, 0.7, 0.7)), (1, (0, 1, 1))]
                )
                
                # Apply mixup to create augmented samples
                half_batch = batch_size // 2
                if half_batch > 0:
                    half_inputs = batch[0][:half_batch].clone()
                    half_labels = batch[1][:half_batch].clone()
                    aug_inputs, aug_labels_a, aug_labels_b, aug_lambda = mixup(half_inputs, half_labels)
                    # Ensure aug_lambda is a tensor with same batch dimension as the inputs
                    if not hasattr(aug_lambda, 'shape') or len(aug_lambda.shape) == 0:
                        # If aug_lambda is a scalar, convert to a tensor with batch dimension
                        import torch
                        aug_lambda = torch.tensor([aug_lambda] * len(aug_inputs))
                    aug_samples = (aug_inputs, aug_labels_a, aug_labels_b, aug_lambda)
            
            for i in range(min(batch_size, len(batch[0]))):
                # Extract sample
                sample = batch[0][i].cpu().numpy()[0]  # Remove channel dimension
                
                # Get sample dimensions
                depth, height, width = sample.shape
                
                # Generate central slices
                central_slice_z = sample[depth//2, :, :]
                central_slice_y = sample[:, height//2, :]
                central_slice_x = sample[:, :, width//2]
                
                # Generate average projections
                avg_proj_z = np.mean(sample, axis=0)
                avg_proj_y = np.mean(sample, axis=1)
                avg_proj_x = np.mean(sample, axis=2)
                
                # Create a figure with 6 subplots (3 slices, 3 projections)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Initialize min/max values for colorbar consistency
                vmin_slices = np.min([np.min(central_slice_z), np.min(central_slice_y), np.min(central_slice_x)])
                vmax_slices = np.max([np.max(central_slice_z), np.max(central_slice_y), np.max(central_slice_x)])
                
                vmin_projs = np.min([np.min(avg_proj_z), np.min(avg_proj_y), np.min(avg_proj_x)])
                vmax_projs = np.max([np.max(avg_proj_z), np.max(avg_proj_y), np.max(avg_proj_x)])
                
                # Plot central slices with consistent colorbar range
                axes[0, 0].imshow(central_slice_z, cmap=slice_colormap, vmin=vmin_slices, vmax=vmax_slices)
                axes[0, 0].set_title(f"Central Z Slice (z={depth//2})")
                
                axes[0, 1].imshow(central_slice_y, cmap=slice_colormap, vmin=vmin_slices, vmax=vmax_slices)
                axes[0, 1].set_title(f"Central Y Slice (y={height//2})")
                
                axes[0, 2].imshow(central_slice_x, cmap=slice_colormap, vmin=vmin_slices, vmax=vmax_slices)
                axes[0, 2].set_title(f"Central X Slice (x={width//2})")
                
                # Plot average projections with consistent colorbar range
                axes[1, 0].imshow(avg_proj_z, cmap=projection_colormap, vmin=vmin_projs, vmax=vmax_projs)
                axes[1, 0].set_title("Average Z Projection")
                
                axes[1, 1].imshow(avg_proj_y, cmap=projection_colormap, vmin=vmin_projs, vmax=vmax_projs)
                axes[1, 1].set_title("Average Y Projection")
                
                axes[1, 2].imshow(avg_proj_x, cmap=projection_colormap, vmin=vmin_projs, vmax=vmax_projs)
                axes[1, 2].set_title("Average X Projection")
                
                # Check if this is one of the augmented samples and capture augmentation details
                is_augmented = False
                class_a = None
                class_b = None
                lam = None
                
                # Get the keys for class lookup
                keys = dataset.keys() if hasattr(dataset, 'keys') else []
                
                if aug_samples is not None and i < len(aug_samples[0]):
                    is_augmented = True
                    class_idx_a = aug_samples[1][i].item()
                    class_idx_b = aug_samples[2][i].item()
                    
                    # Fix for class label resolution
                    if class_idx_a == -1:
                        class_a = "background"
                    elif class_idx_a >= 0 and class_idx_a < len(keys):
                        class_a = keys[class_idx_a]
                    else:
                        class_a = f"unknown (idx={class_idx_a})"
                        
                    if class_idx_b == -1:
                        class_b = "background"
                    elif class_idx_b >= 0 and class_idx_b < len(keys):
                        class_b = keys[class_idx_b]
                    else:
                        class_b = f"unknown (idx={class_idx_b})"
                        
                    print(f"Augmented sample {i}: Class A idx={class_idx_a} ({class_a}), Class B idx={class_idx_b} ({class_b}), λ={lam:.2f}")
                    # Access lambda value - ensure it's an indexed tensor or a float
                    if hasattr(aug_samples[3], 'shape') and len(aug_samples[3].shape) > 0:
                        lam = aug_samples[3][i].item()
                    else:
                        lam = float(aug_samples[3])
                
                # Add a main title
                if is_augmented:
                    fig.suptitle(f"Augmented Sample {i+1} - Class Mix: {class_a} ({lam:.2f}) + {class_b} ({1-lam:.2f})", fontsize=16)
                else:
                    # Handle the possibility of integer or dict label format
                    if isinstance(batch[1][i], dict):
                        class_idx = batch[1][i]['class_idx']
                    else:
                        class_idx = batch[1][i].item()
                    
                    # Fix for class label resolution - ensure we're using the actual keys array
                    keys = dataset.keys()
                    if class_idx == -1:
                        class_label = "background"
                    elif class_idx >= 0 and class_idx < len(keys):
                        class_label = keys[class_idx]
                    else:
                        class_label = f"unknown (idx={class_idx})"
                        
                    print(f"Regular sample {i}: Class idx={class_idx} ({class_label})")
                    fig.suptitle(f"Sample {i+1} - Class: {class_label}", fontsize=16)
                
                # Add colorbar to show data range
                for j in range(2):
                    for k in range(3):
                        im = axes[j, k].get_images()[0]
                        plt.colorbar(im, ax=axes[j, k])
                
                plt.tight_layout()
                
                # Convert plot to base64 image
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                img_data = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
                
                # Add to HTML
                # Handle the possibility that class_label might not be defined yet
                try:
                    sample_title = f"Sample {i+1} - Class: {class_label}"
                except NameError:
                    sample_title = f"Sample {i+1}"
                
                # Add augmentation details if available
                aug_details = ""
                if is_augmented:
                    aug_details = f'''
                    <div class="augmentation-details">
                        <h3>Augmentation Details:</h3>
                        <ul>
                            <li><strong>Type:</strong> Mixup</li>
                            <li><strong>Class A:</strong> {class_a} (ratio: {lam:.2f})</li>
                            <li><strong>Class B:</strong> {class_b} (ratio: {1-lam:.2f})</li>
                        </ul>
                    </div>
                    '''
                
                images_html.append(f'<div class="sample"><h2>{sample_title}</h2>{aug_details}<img src="data:image/png;base64,{img_data}" /></div>')
        
        except StopIteration:
            # If batch creation failed, try to generate patches from the tomogram directly
            print("No samples from dataloader, creating grid patches instead")
            print(f"StopIteration error - dataset may be empty or have inconsistent labels")
            
            # Get the first tomogram
            tomogram = tomograms[0]
            tomogram_array = tomogram.numpy()
            
            # Define an extract_grid_patches method since SimpleCopickDataset doesn't have one directly
            def extract_grid_patches(tomogram_array, patch_size, overlap=0.25, normalize=True):
                """Extract a grid of patches from a tomogram."""
                # Validate parameters
                if isinstance(patch_size, int):
                    patch_size = (patch_size, patch_size, patch_size)
                elif len(patch_size) != 3:
                    raise ValueError("patch_size must be an integer or tuple of 3 integers")
                    
                if overlap < 0 or overlap >= 1:
                    raise ValueError("overlap must be between 0 and 1")
                
                # Calculate stride (step size between patches)
                stride_z = int(patch_size[0] * (1 - overlap))
                stride_y = int(patch_size[1] * (1 - overlap))
                stride_x = int(patch_size[2] * (1 - overlap))
                
                # Ensure stride is at least 1
                stride_z = max(1, stride_z)
                stride_y = max(1, stride_y)
                stride_x = max(1, stride_x)
                
                # Calculate number of patches in each dimension
                n_patches_z = 1 + (tomogram_array.shape[0] - patch_size[0]) // stride_z
                n_patches_y = 1 + (tomogram_array.shape[1] - patch_size[1]) // stride_y
                n_patches_x = 1 + (tomogram_array.shape[2] - patch_size[2]) // stride_x
                
                # Initialize results
                patches = []
                coordinates = []
                
                # Extract patches
                for iz in range(n_patches_z):
                    z_start = iz * stride_z
                    z_end = z_start + patch_size[0]
                    if z_end > tomogram_array.shape[0]:
                        continue
                        
                    for iy in range(n_patches_y):
                        y_start = iy * stride_y
                        y_end = y_start + patch_size[1]
                        if y_end > tomogram_array.shape[1]:
                            continue
                            
                        for ix in range(n_patches_x):
                            x_start = ix * stride_x
                            x_end = x_start + patch_size[2]
                            if x_end > tomogram_array.shape[2]:
                                continue
                                
                            # Extract the patch
                            patch = tomogram_array[z_start:z_end, y_start:y_end, x_start:x_end].copy()
                            
                            # Normalize if requested
                            if normalize:
                                # Center and scale to unit variance
                                patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-6)
                                
                            # Record patch and its center coordinates
                            patches.append(patch)
                            coordinates.append((z_start + patch_size[0]//2,
                                             y_start + patch_size[1]//2,
                                             x_start + patch_size[2]//2))
                
                print(f"Extracted {len(patches)} patches of size {patch_size} with {overlap:.2f} overlap")
                return patches, coordinates
                
            # Extract patches using our helper function
            patches, coordinates = extract_grid_patches(
                tomogram_array, 
                patch_size=box_size,
                overlap=0.5,
                normalize=True
            )
            
            # Use up to batch_size patches
            for i, (patch, coord) in enumerate(zip(patches[:batch_size], coordinates[:batch_size])):
                # Get patch dimensions
                depth, height, width = patch.shape
                
                # Generate central slices
                central_slice_z = patch[depth//2, :, :]
                central_slice_y = patch[:, height//2, :]
                central_slice_x = patch[:, :, width//2]
                
                # Generate average projections
                avg_proj_z = np.mean(patch, axis=0)
                avg_proj_y = np.mean(patch, axis=1)
                avg_proj_x = np.mean(patch, axis=2)
                
                # Create a figure with 6 subplots (3 slices, 3 projections)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Initialize min/max values for colorbar consistency
                vmin_slices = np.min([np.min(central_slice_z), np.min(central_slice_y), np.min(central_slice_x)])
                vmax_slices = np.max([np.max(central_slice_z), np.max(central_slice_y), np.max(central_slice_x)])
                
                vmin_projs = np.min([np.min(avg_proj_z), np.min(avg_proj_y), np.min(avg_proj_x)])
                vmax_projs = np.max([np.max(avg_proj_z), np.max(avg_proj_y), np.max(avg_proj_x)])
                
                # Plot central slices with consistent colorbar range
                axes[0, 0].imshow(central_slice_z, cmap=slice_colormap, vmin=vmin_slices, vmax=vmax_slices)
                axes[0, 0].set_title(f"Central Z Slice (z={depth//2})")
                
                axes[0, 1].imshow(central_slice_y, cmap=slice_colormap, vmin=vmin_slices, vmax=vmax_slices)
                axes[0, 1].set_title(f"Central Y Slice (y={height//2})")
                
                axes[0, 2].imshow(central_slice_x, cmap=slice_colormap, vmin=vmin_slices, vmax=vmax_slices)
                axes[0, 2].set_title(f"Central X Slice (x={width//2})")
                
                # Plot average projections with consistent colorbar range
                axes[1, 0].imshow(avg_proj_z, cmap=projection_colormap, vmin=vmin_projs, vmax=vmax_projs)
                axes[1, 0].set_title("Average Z Projection")
                
                axes[1, 1].imshow(avg_proj_y, cmap=projection_colormap, vmin=vmin_projs, vmax=vmax_projs)
                axes[1, 1].set_title("Average Y Projection")
                
                axes[1, 2].imshow(avg_proj_x, cmap=projection_colormap, vmin=vmin_projs, vmax=vmax_projs)
                axes[1, 2].set_title("Average X Projection")
                
                # Add a main title
                coord_str = f"({coord[0]}, {coord[1]}, {coord[2]})"
                fig.suptitle(f"Patch {i+1} - Coordinates: {coord_str}", fontsize=16)
                plt.tight_layout()
                
                # Convert plot to base64 image
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                img_data = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close(fig)
                
                # Add to HTML
                images_html.append(f'<div class="sample"><h2>Patch {i+1} - Coordinates: {coord_str}</h2><img src="data:image/png;base64,{img_data}" /></div>')
        
        # Get the dataset's class distribution and ensure all classes are represented
        try:
            class_distribution = dataset.get_class_distribution()
            print(f"Class distribution from dataset: {class_distribution}")
        except Exception as e:
            print(f"Error getting class distribution: {str(e)}")
            # Create a manual class distribution based on available objects
            class_distribution = {obj: 0 for obj in available_objects}
            if "background" not in class_distribution:
                class_distribution["background"] = 0
            
        # Add a function to build class distribution HTML that's robust to errors
        def get_class_distribution_html(distribution, available_objects=None):
            if not distribution and available_objects:
                # If we have no distribution but know the available objects, use those
                html = '<p><em>Unable to determine class counts, but these classes were detected:</em></p><ul>'
                html += ''.join([f'<li><strong>{cls}</strong></li>' for cls in sorted(available_objects)])
                html += '</ul>'
                return html
            elif distribution:
                # Normal case with distribution
                html = '<ul>'
                # Ensure consistent capitalization for 'background'
                distribution = {cls.lower() if cls.lower() == 'background' else cls: count 
                               for cls, count in distribution.items()}
                html += ''.join([f'<li><strong>{cls}:</strong> {count} samples</li>' 
                               for cls, count in sorted(distribution.items())])
                html += '</ul>'
                return html
            else:
                # Fallback case
                return '<p><em>No class information available</em></p>'
        
        # If using balanced sampling, we should show the actual distribution of the batch instead
        if use_balanced_sampling and 'batch' in locals() and len(batch) > 1:
            batch_class_distribution = {}
            
            # Count the frequency of each class in the batch
            keys = dataset.keys() if hasattr(dataset, 'keys') else []
            for i in range(len(batch[1])):
                if isinstance(batch[1][i], dict):
                    class_idx = batch[1][i]['class_idx']
                else:
                    class_idx = batch[1][i].item()
                
                # Fix for class label resolution
                if class_idx == -1:
                    class_name = "background"
                elif class_idx >= 0 and class_idx < len(keys):
                    class_name = keys[class_idx]
                else:
                    class_name = f"unknown-class-{class_idx}"
                    
                print(f"Class distribution: Added {class_name} (class_idx={class_idx})")
                
                if class_name in batch_class_distribution:
                    batch_class_distribution[class_name] += 1
                else:
                    batch_class_distribution[class_name] = 1
            
            # If we have augmentation with mixup, also count the mixed classes
            if aug_samples is not None:
                for i in range(len(aug_samples[1])):
                    # Class A
                    class_idx_a = aug_samples[1][i].item()
                    if class_idx_a == -1:
                        class_name_a = "background"
                    elif class_idx_a >= 0 and class_idx_a < len(keys):
                        class_name_a = keys[class_idx_a]
                    else:
                        class_name_a = f"unknown-class-{class_idx_a}"
                    
                    # Class B
                    class_idx_b = aug_samples[2][i].item()
                    if class_idx_b == -1:
                        class_name_b = "background"
                    elif class_idx_b >= 0 and class_idx_b < len(keys):
                        class_name_b = keys[class_idx_b]
                    else:
                        class_name_b = f"unknown-class-{class_idx_b}"
                        
                    # Get lambda value for these samples
                    if hasattr(aug_samples[3], 'shape') and len(aug_samples[3].shape) > 0:
                        lam = aug_samples[3][i].item()
                    else:
                        lam = float(aug_samples[3])
                    
                    print(f"Class distribution: Added {class_name_a} (λ={lam:.2f}) and {class_name_b} (λ={1-lam:.2f})")
                    
                    # Count both classes (we visualize both in the mixup)
                    for class_name in [class_name_a, class_name_b]:
                        if class_name in batch_class_distribution:
                            batch_class_distribution[class_name] += 0.5  # Count as half since it's mixed
                        else:
                            batch_class_distribution[class_name] = 0.5
                
                # Round the counts for display
                for cls in batch_class_distribution:
                    batch_class_distribution[cls] = round(batch_class_distribution[cls])
            
            # Add note that this is the batch distribution, not the full dataset
            if batch_class_distribution:
                class_distribution = batch_class_distribution
                class_dist_html = '<p><em>Showing combined class distribution of current batch, including augmentations:</em></p>' + \
                    get_class_distribution_html(class_distribution)
            else:
                # Make sure all detected classes are in the distribution
                for obj_name in available_objects:
                    if obj_name not in class_distribution:
                        class_distribution[obj_name] = 0
                class_dist_html = '<p><em>Showing distribution of full dataset:</em></p>' + \
                    get_class_distribution_html(class_distribution)
        else:
            # Make sure all detected classes are in the distribution
            for obj_name in available_objects:
                if obj_name not in class_distribution:
                    class_distribution[obj_name] = 0
            class_dist_html = get_class_distribution_html(class_distribution, available_objects)
        
        # Create HTML page
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Copick Tomogram Visualization</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    text-align: center;
                }}
                .sample {{
                    margin: 20px 0;
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .sample h2 {{
                    margin-top: 0;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                .info {{
                    background-color: #e0f7fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .class-distribution {{
                    background-color: #f1f8e9;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .augmentation-details {{
                    background-color: #fff9c4;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <h1>Copick Tomogram Visualization</h1>
            <div class="info">
                <p><strong>Dataset ID:</strong> {dataset_id}</p>
                <p><strong>Run:</strong> {run_name}</p>
                <p><strong>Voxel Spacing:</strong> {voxel_spacing}</p>
                <p><strong>Tomogram Type:</strong> {tomo_type}</p>
                <p><strong>Box Size:</strong> {box_size}</p>
                <p><strong>Samples:</strong> {len(images_html)}</p>
                <p><strong>Augmentations:</strong> {augment}</p>
                <p><strong>Show Augmentations:</strong> {show_augmentations}</p>
                <p><strong>Class Balanced Sampling:</strong> {use_balanced_sampling}</p>
                <p><strong>Background Ratio:</strong> {background_ratio}</p>
                <p><strong>Mixup Alpha:</strong> {mixup_alpha}</p>
                <p><strong>Slice Colormap:</strong> {slice_colormap}</p>
                <p><strong>Projection Colormap:</strong> {projection_colormap}</p>
                <p><strong>Dataset Size:</strong> {len(dataset) if 'dataset' in locals() else 'Unknown'}</p>
                <p><strong>Registered Classes:</strong> {', '.join(sorted(set(dataset.keys()))) if 'dataset' in locals() and hasattr(dataset, 'keys') else 'Unknown'}</p>
                <p><strong>Total Unique Classes:</strong> {len(set(dataset.keys())) if 'dataset' in locals() and hasattr(dataset, 'keys') else 'Unknown'}</p>
            </div>
            <div class="class-distribution">
                <h3>Class Distribution:</h3>
                {class_dist_html}
            </div>
            {''.join(images_html)}
        </body>
        </html>
        """
        
        # Clean up the temporary file
        try:
            os.unlink(config_path)
        except:
            pass
        
        return html_content
    
    except Exception as e:
        # Create an error HTML page
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Copick Tomogram Visualization</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #c00;
                    text-align: center;
                }}
                .error {{
                    margin: 20px 0;
                    background-color: #ffebee;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    border-left: 5px solid #c00;
                }}
                pre {{
                    background-color: #fff;
                    padding: 10px;
                    border-radius: 3px;
                    overflow-x: auto;
                }}
                .info {{
                    background-color: #e0f7fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .actions {{
                    background-color: #fff9c4;
                    padding: 10px;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Error in Copick Tomogram Visualization</h1>
            <div class="info">
                <p><strong>Dataset ID:</strong> {dataset_id}</p>
                <p><strong>Run:</strong> {run_name or "Not specified"}</p>
                <p><strong>Voxel Spacing:</strong> {voxel_spacing or "Not specified"}</p>
                <p><strong>Tomogram Type:</strong> {tomo_type}</p>
                <p><strong>Augmentations:</strong> {augment}</p>
                <p><strong>Class Balanced Sampling:</strong> {use_balanced_sampling}</p>
            </div>
            <div class="error">
                <h3>Error Details:</h3>
                <p>{str(e)}</p>
            </div>
            <div class="actions">
                <h3>Try the following:</h3>
                <ul>
                    <li>Check if the dataset ID is correct</li>
                    <li>Verify that the overlay root directory is accessible</li>
                    <li>Try a different run name or voxel spacing</li>
                    <li>Ensure the tomogram type is valid</li>
                </ul>
                <p>You can find available runs and their details using the Copick tools:</p>
                <pre>
# List runs
list_runs(dataset_id={dataset_id}, overlay_root="{overlay_root}")

# Get run details
get_run_details(run_name="Run-1", dataset_id={dataset_id}, overlay_root="{overlay_root}")

# List voxel spacings
list_voxel_spacings(run_name="Run-1", dataset_id={dataset_id}, overlay_root="{overlay_root}")

# List objects (particles)
list_objects(dataset_id={dataset_id}, overlay_root="{overlay_root}")
                </pre>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=200)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that provides links to available endpoints"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Copick Tomogram Visualization Server</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 2px solid #3498db;
            }
            .endpoint {
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            h2 {
                color: #3498db;
                margin-top: 0;
            }
            code {
                background-color: #eee;
                padding: 3px 5px;
                border-radius: 3px;
                font-family: monospace;
            }
            .param {
                margin-left: 20px;
                margin-bottom: 5px;
            }
            .param code {
                font-weight: bold;
            }
            .example {
                margin-top: 15px;
                background-color: #e8f4fd;
                padding: 10px;
                border-radius: 5px;
            }
            .main-example {
                margin-top: 20px;
                background-color: #e0f7fa;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .main-example a {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
            }
            .main-example a:hover {
                background-color: #2980b9;
            }
        </style>
    </head>
    <body>
        <h1>Copick Tomogram Visualization Server</h1>
        
        <div class="main-example">
            <p>Try the CZ cryoET Data Portal sample dataset:</p>
            <a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp">View Dataset 10440 - Run 16463</a>
            <br><br>
            <a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp&use_balanced_sampling=true&show_augmentations=true&mixup_alpha=0.2">View with Balanced Sampling & Augmentations</a>
        </div>
        
        <div class="endpoint">
            <h2>Tomogram Visualization</h2>
            <p>Visualizes tomogram samples from a dataset, showing central slices and average projections along all axes.</p>
            <p><strong>Endpoint:</strong> <code>/tomogram-viz</code></p>
            <p><strong>Parameters:</strong></p>
            <div class="param">
                <code>dataset_id</code>: Dataset ID from CZ cryoET Data Portal (required)
            </div>
            <div class="param">
                <code>overlay_root</code>: Root directory for the overlay storage (default: "/tmp/test/")
            </div>
            <div class="param">
                <code>run_name</code>: Name of the run to visualize (if not provided, uses the first run)
            </div>
            <div class="param">
                <code>voxel_spacing</code>: Voxel spacing to use (if not provided, uses the first available)
            </div>
            <div class="param">
                <code>tomo_type</code>: Tomogram type, e.g. "wbp" (default: "wbp")
            </div>
            <div class="param">
                <code>batch_size</code>: Number of samples to visualize (default: 25)
            </div>
            <div class="param">
                <code>box_size</code>: Box size for subvolume extraction (default: 64)
            </div>
            <div class="param">
                <code>slice_colormap</code>: Matplotlib colormap for slices (default: "gray")
            </div>
            <div class="param">
                <code>projection_colormap</code>: Matplotlib colormap for projections (default: "viridis")
            </div>
            <div class="param">
                <code>augment</code>: Apply augmentations (default: true)
            </div>
            <div class="param">
                <code>show_augmentations</code>: Show augmentation stages (default: false)
            </div>
            <div class="param">
                <code>use_balanced_sampling</code>: Use class balanced sampling (default: true)
            </div>
            <div class="param">
                <code>background_ratio</code>: Ratio of background samples (default: 0.2)
            </div>
            <div class="example">
                <p><strong>Example:</strong></p>
                <p><a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp">/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp</a></p>
                <p><a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp&use_balanced_sampling=true&show_augmentations=true&mixup_alpha=0.2">/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp&use_balanced_sampling=true&show_augmentations=true&mixup_alpha=0.2</a></p>
            </div>
        </div>
        
        <div class="endpoint">
            <h2>Available Datasets</h2>
            <p>The CZ cryoET Data Portal has several datasets that can be used with this visualization server:</p>
            <ul>
                <li>Dataset ID: 10440 - Example dataset with various particle types</li>
            </ul>
            <p>You can explore datasets and runs using the Copick tools:</p>
            <pre>
# List runs in a dataset
list_runs(dataset_id=10440, overlay_root="/tmp/test/")

# Get run details
get_run_details(run_name="16463", dataset_id=10440, overlay_root="/tmp/test/")

# List voxel spacings for a run
list_voxel_spacings(run_name="16463", dataset_id=10440, overlay_root="/tmp/test/")

# List available objects in a dataset
list_objects(dataset_id=10440, overlay_root="/tmp/test/")
            </pre>
        </div>
    </body>
    </html>
    """

# Initialize the Copick server with CZII dataset
try:
    # Try using direct CZII dataset loading
    root = copick.from_czcdp_datasets([10440], overlay_root="/tmp/test/")
    print("Successfully loaded CoPick project from CZII dataset")
except Exception as e:
    # If that fails, create a dummy root for testing
    print(f"Error loading CZII dataset: {str(e)}")
    print("Creating dummy copick project for testing")
    # Create a dummy temporary config for testing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        config_path = temp_file.name
        # Make a basic config
        config = {
            "name": "Test Project",
            "description": "A test CoPick project",
            "config_type": "cryoet_data_portal", 
            "dataset_ids": [10440],
            "overlay_root": "/tmp/test/"
        }
        json.dump(config, temp_file)
    
    try:
        # Try to load from the temp file
        root = copick.from_file(config_path)
    except Exception as ex:
        print(f"Error creating dummy project: {str(ex)}")
        # Just create an empty object as placeholder
        class DummyRoot:
            def __init__(self):
                self.name = "Dummy Project"
                self.description = "This is a placeholder project"
        root = DummyRoot()

# Add the Copick route handler
route_handler = CopickRoute(root)

# Add the Copick catch-all route at the end
app.add_api_route(
    "/{path:path}",
    route_handler.handle_request,
    methods=["GET", "HEAD", "PUT"]
)

if __name__ == "__main__":
    # Print instructions
    print(f"Server is running on http://{HOST}:{PORT}")
    print("Available endpoints:")
    print(f"  - http://{HOST}:{PORT}/ (Home page)")
    print(f"  - http://{HOST}:{PORT}/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp (Visualization)")
    print(f"  - http://{HOST}:{PORT}/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp&use_balanced_sampling=true&show_augmentations=true&mixup_alpha=0.2 (With Balanced Sampling & Augmentations)")
    print("Press Ctrl+C to exit")
    
    # Start the server
    uvicorn.run(app, host=HOST, port=PORT)
