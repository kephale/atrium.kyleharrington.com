# /// script
# title = "Copick Tomogram Visualization Server"
# description = "A FastAPI server that extends copick-server to provide visualization of tomogram samples."
# author = "Kyle Harrington <czi@kyleharrington.com>"
# license = "MIT"
# version = "0.0.2"
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
#     "cryoet-data-portal",
#     "zarr<3",
#     "numcodecs<0.16.0",  
#     "copick>=0.8.0",
#     "copick-torch @ git+https://github.com/kephale/copick-torch.git",
#     "copick-server @ git+https://github.com/kephale/copick-server.git"
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
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn
import threading

# Import from copick-server
from copick_server.server import CopickRoute
from fastapi.middleware.cors import CORSMiddleware
import copick

# Find the example_copick.json file
potential_paths = [
    os.path.expanduser("~/git/copick/copick-server/example_copick.json")
]

config_path = None
for path in potential_paths:
    if os.path.exists(path):
        config_path = path
        break

if config_path is None:
    raise FileNotFoundError("Could not find example_copick.json in any of the expected locations.")

print(f"Using config file: {config_path}")

# Port configuration - define once and reuse
PORT = 8018
HOST = "0.0.0.0"

# Load the Copick project
root = copick.from_file(config_path)

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

# Import from copick-torch
import copick_torch
from copick_torch.copick import CopickDataset
from cryoet_data_portal import client as cryoet_client

@app.get("/tomogram-viz", response_class=HTMLResponse)
async def visualize_tomograms(
    dataset_path: str = Query(..., description="Path to the dataset directory"),
    batch_size: int = Query(25, description="Number of samples to visualize"),
    slice_colormap: str = Query("gray", description="Colormap for slices"),
    projection_colormap: str = Query("viridis", description="Colormap for projections")
):
    """
    Visualize tomogram samples from a dataset, showing central slices and average projections
    along all axes.
    
    Args:
        dataset_path: Path to the dataset directory
        batch_size: Number of samples to visualize
        slice_colormap: Matplotlib colormap for slices
        projection_colormap: Matplotlib colormap for projections
    
    Returns:
        HTML page with visualizations
    """
    try:
        # Load the dataset
        dataset = CopickDataset(dataset_path, augment=False)
        
        # Create a dataloader to get a batch
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Generate visualization for each sample
        images_html = []
        
        for i in range(min(batch_size, len(batch))):
            # Extract sample
            if isinstance(batch, dict):
                sample = batch['image'][i].cpu().numpy() if 'image' in batch else batch['volume'][i].cpu().numpy()
            else:
                sample = batch[i][0].cpu().numpy()  # Get volume from (volume, label) tuple
            
            # Ensure we have a 3D volume
            if len(sample.shape) == 4:
                sample = sample[0]  # Remove channel dimension if present
            
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
            
            # Plot central slices
            axes[0, 0].imshow(central_slice_z, cmap=slice_colormap)
            axes[0, 0].set_title(f"Central Z Slice (z={depth//2})")
            
            axes[0, 1].imshow(central_slice_y, cmap=slice_colormap)
            axes[0, 1].set_title(f"Central Y Slice (y={height//2})")
            
            axes[0, 2].imshow(central_slice_x, cmap=slice_colormap)
            axes[0, 2].set_title(f"Central X Slice (x={width//2})")
            
            # Plot average projections
            axes[1, 0].imshow(avg_proj_z, cmap=projection_colormap)
            axes[1, 0].set_title("Average Z Projection")
            
            axes[1, 1].imshow(avg_proj_y, cmap=projection_colormap)
            axes[1, 1].set_title("Average Y Projection")
            
            axes[1, 2].imshow(avg_proj_x, cmap=projection_colormap)
            axes[1, 2].set_title("Average X Projection")
            
            # Add a main title
            fig.suptitle(f"Sample {i+1}", fontsize=16)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            # Add to HTML
            images_html.append(f'<div class="sample"><h2>Sample {i+1}</h2><img src="data:image/png;base64,{img_data}" /></div>')
        
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
            </style>
        </head>
        <body>
            <h1>Copick Tomogram Visualization</h1>
            <div class="info">
                <p><strong>Dataset:</strong> {dataset_path}</p>
                <p><strong>Samples:</strong> {min(batch_size, len(batch))}</p>
                <p><strong>Slice Colormap:</strong> {slice_colormap}</p>
                <p><strong>Projection Colormap:</strong> {projection_colormap}</p>
            </div>
            {''.join(images_html)}
        </body>
        </html>
        """
        
        return html_content
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")

@app.get("/demo", response_class=HTMLResponse)
async def demo(
    batch_size: int = Query(25, description="Number of samples to visualize"),
    slice_colormap: str = Query("gray", description="Colormap for slices"),
    projection_colormap: str = Query("viridis", description="Colormap for projections")
):
    """Demo endpoint that shows 25 examples from the Copick project with visualizations"""
    try:
        # Get project information
        project_name = root.name
        project_description = root.description
        
        # Get dataset ID if available
        dataset_ids = getattr(root, 'dataset_ids', [])
        dataset_id_text = f"Dataset ID: {dataset_ids[0]}" if dataset_ids else ""
        
        # Get run information - first available run
        runs = root.runs
        if not runs:
            raise HTTPException(status_code=500, detail="No runs found in the Copick project")
        
        run = runs[0]  # Use the first run
        
        # Get voxel spacings for this run
        if not run.voxel_spacings:
            raise HTTPException(status_code=500, detail="No voxel spacings found in the run")
            
        # Use the first voxel spacing
        voxel_spacing = run.voxel_spacings[0]
        
        # Get tomograms for this voxel spacing
        if not voxel_spacing.tomograms:
            raise HTTPException(status_code=500, detail="No tomograms found for the voxel spacing")
            
        # Use the first tomogram
        tomogram = voxel_spacing.tomograms[0]
        
        # Get the raw tomogram data
        tomo_data = tomogram.zarr()
        if not tomo_data:
            raise HTTPException(status_code=500, detail="Could not access tomogram data")
        
        # Extract 3D volume data from the zarr array
        volume_data = tomo_data[:]
        
        # Generate visualization for samples from the volume
        images_html = []
        
        # Get dimensions
        depth, height, width = volume_data.shape
        
        # Calculate step size for sampling
        depth_step = max(1, depth // batch_size)
        
        # Generate samples
        for i in range(min(batch_size, depth // depth_step)):
            # Extract a subvolume
            z_pos = i * depth_step + depth_step // 2  # Center of the step
            z_pos = min(z_pos, depth-1)  # Ensure we don't go out of bounds
            
            # Extract a section of the volume around this position
            half_section = min(10, depth_step // 2)  # Half the section size
            z_start = max(0, z_pos - half_section)
            z_end = min(depth, z_pos + half_section)
            
            # Get the subvolume
            sample = volume_data[z_start:z_end, :, :]
            
            # Use the center slice for this section
            section_center = (z_end - z_start) // 2
            
            # Generate central slices
            central_slice_z = sample[section_center, :, :]
            central_slice_y = sample[:, height//2, :]
            central_slice_x = sample[:, :, width//2]
            
            # Generate average projections
            avg_proj_z = np.mean(sample, axis=0)
            avg_proj_y = np.mean(sample, axis=1)
            avg_proj_x = np.mean(sample, axis=2)
            
            # Create a figure with 6 subplots (3 slices, 3 projections)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Plot central slices
            axes[0, 0].imshow(central_slice_z, cmap=slice_colormap)
            axes[0, 0].set_title(f"Z Slice at z={z_pos}")
            
            axes[0, 1].imshow(central_slice_y, cmap=slice_colormap)
            axes[0, 1].set_title(f"Y Slice at y={height//2}")
            
            axes[0, 2].imshow(central_slice_x, cmap=slice_colormap)
            axes[0, 2].set_title(f"X Slice at x={width//2}")
            
            # Plot average projections
            axes[1, 0].imshow(avg_proj_z, cmap=projection_colormap)
            axes[1, 0].set_title("Average Z Projection")
            
            axes[1, 1].imshow(avg_proj_y, cmap=projection_colormap)
            axes[1, 1].set_title("Average Y Projection")
            
            axes[1, 2].imshow(avg_proj_x, cmap=projection_colormap)
            axes[1, 2].set_title("Average X Projection")
            
            # Add a main title
            fig.suptitle(f"Sample {i+1} (z={z_pos})", fontsize=16)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)
            
            # Add to HTML
            images_html.append(f'<div class="sample"><h2>Sample {i+1} (z={z_pos})</h2><img src="data:image/png;base64,{img_data}" /></div>')
        
        # Create HTML page
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Copick Project Demo: {project_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .project-info {{
                    background-color: #e0f7fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    border-left: 4px solid #3498db;
                }}
                .sample {{
                    margin: 30px 0;
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .sample h2 {{
                    margin-top: 0;
                    color: #3498db;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 8px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }}
                .info {{
                    background-color: #f8f9fa;
                    padding: 10px;
                    border-radius: 5px;
                    margin: 15px 0;
                    border-left: 4px solid #2ecc71;
                }}
            </style>
        </head>
        <body>
            <h1>Copick Project Demo: {project_name}</h1>
            
            <div class="project-info">
                <p><strong>Description:</strong> {project_description}</p>
                <p><strong>{dataset_id_text}</strong></p>
                <p><strong>Run:</strong> {run.name}</p>
                <p><strong>Voxel Spacing:</strong> {voxel_spacing.voxel_spacing}</p>
                <p><strong>Tomogram:</strong> {tomogram.name}</p>
                <p><strong>Volume Shape:</strong> {volume_data.shape}</p>
            </div>
            
            <div class="info">
                <p><strong>Visualization:</strong> Showing {len(images_html)} samples from the tomogram</p>
                <p><strong>Each visualization includes:</strong></p>
                <ul>
                    <li>Central orthogonal slices (top row)</li>
                    <li>Average projections along each axis (bottom row)</li>
                </ul>
                <p><strong>Slice Colormap:</strong> {slice_colormap}</p>
                <p><strong>Projection Colormap:</strong> {projection_colormap}</p>
            </div>
            
            {''.join(images_html)}
        </body>
        </html>
        """
        
        return html_content
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating demo: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that provides links to available endpoints"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Copick Server with Tomogram Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #2c3e50;
            }
            .endpoint {
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 10px 15px;
                margin-bottom: 15px;
            }
            .endpoint h2 {
                margin-top: 0;
                color: #3498db;
            }
            code {
                background-color: #eee;
                padding: 2px 5px;
                border-radius: 3px;
            }
            .example {
                margin-top: 10px;
                background-color: #e8f4fc;
                padding: 10px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Copick Server with Tomogram Visualization</h1>
        
        <div class="endpoint">
            <h2>Demo</h2>
            <p>View 25 examples directly from the Copick project tomogram data with central slices and projections.</p>
            <p><strong>Endpoint:</strong> <code>/demo</code></p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>batch_size</code>: Number of samples to visualize (default: 25)</li>
                <li><code>slice_colormap</code>: Matplotlib colormap for slices (default: gray)</li>
                <li><code>projection_colormap</code>: Matplotlib colormap for projections (default: viridis)</li>
            </ul>
            <div class="example">
                <p><strong>Example:</strong> <a href="/demo">/demo</a></p>
            </div>
        </div>

        <div class="endpoint">
            <h2>Tomogram Visualization</h2>
            <p>Visualizes tomogram samples from a dataset, showing central slices and average projections.</p>
            <p><strong>Endpoint:</strong> <code>/tomogram-viz</code></p>
            <p><strong>Parameters:</strong></p>
            <ul>
                <li><code>dataset_path</code>: Path to the dataset directory (required)</li>
                <li><code>batch_size</code>: Number of samples to visualize (default: 25)</li>
                <li><code>slice_colormap</code>: Matplotlib colormap for slices (default: gray)</li>
                <li><code>projection_colormap</code>: Matplotlib colormap for projections (default: viridis)</li>
            </ul>
            <div class="example">
                <p><strong>Example:</strong> <a href="/tomogram-viz?dataset_path=/path/to/dataset">/tomogram-viz?dataset_path=/path/to/dataset</a></p>
            </div>
        </div>
        
        <div class="endpoint">
            <h2>CoPick API Endpoints</h2>
            <p>The original CoPick server endpoints are also available.</p>
            <p>Check the <a href="https://github.com/kephale/copick-server">CoPick server documentation</a> for details.</p>
        </div>
    </body>
    </html>
    """

# Add the Copick route handler after our custom routes
route_handler = CopickRoute(root)

# Add the Copick catch-all route at the end
app.add_api_route(
    "/{path:path}",
    route_handler.handle_request,
    methods=["GET", "HEAD", "PUT"]
)

if __name__ == "__main__":
    # Now start the server with our custom routes properly included
    print(f"Server is running on http://{HOST}:{PORT}")
    print("Press Ctrl+C to exit")
    uvicorn.run(app, host=HOST, port=PORT)
