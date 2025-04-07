                <p><strong>Run:</strong> {run_name}</p>
                <p><strong>Voxel Spacing:</strong> {voxel_spacing}</p>
                <p><strong>Tomogram Type:</strong> {tomo_type}</p>
                <p><strong>Box Size:</strong> {box_size}</p>
                <p><strong>Samples:</strong> {len(images_html)}</p>
                <p><strong>Slice Colormap:</strong> {slice_colormap}</p>
                <p><strong>Projection Colormap:</strong> {projection_colormap}</p>
            </div>
            <div class="options">
                <h3>Visualization Options:</h3>
                <p><strong>Class Balancing:</strong> 
                    <a href="?dataset_id={dataset_id}&overlay_root={overlay_root}&run_name={run_name}&voxel_spacing={voxel_spacing}&tomo_type={tomo_type}&box_size={box_size}&use_class_balancing=true&apply_mixup={apply_mixup}" 
                       class="toggle-button {'active' if use_class_balancing else 'inactive'}">
                       {'On' if use_class_balancing else 'Off'}
                    </a>
                    <a href="?dataset_id={dataset_id}&overlay_root={overlay_root}&run_name={run_name}&voxel_spacing={voxel_spacing}&tomo_type={tomo_type}&box_size={box_size}&use_class_balancing={not use_class_balancing}&apply_mixup={apply_mixup}" 
                       class="toggle-button {'inactive' if use_class_balancing else 'active'}">
                       {'Off' if use_class_balancing else 'On'}
                    </a>
                </p>
                <p><strong>Mixup Augmentation:</strong> 
                    <a href="?dataset_id={dataset_id}&overlay_root={overlay_root}&run_name={run_name}&voxel_spacing={voxel_spacing}&tomo_type={tomo_type}&box_size={box_size}&use_class_balancing={use_class_balancing}&apply_mixup=true" 
                       class="toggle-button {'active' if apply_mixup else 'inactive'}">
                       {'On' if apply_mixup else 'Off'}
                    </a>
                    <a href="?dataset_id={dataset_id}&overlay_root={overlay_root}&run_name={run_name}&voxel_spacing={voxel_spacing}&tomo_type={tomo_type}&box_size={box_size}&use_class_balancing={use_class_balancing}&apply_mixup={not apply_mixup}" 
                       class="toggle-button {'inactive' if apply_mixup else 'active'}">
                       {'Off' if apply_mixup else 'On'}
                    </a>
                </p>
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
            .feature {
                background-color: #f1f8e9;
                padding: 10px;
                margin-top: 10px;
                border-radius: 5px;
            }
            .feature h3 {
                margin-top: 0;
                color: #43a047;
            }
        </style>
    </head>
    <body>
        <h1>Copick Tomogram Visualization Server</h1>
        
        <div class="main-example">
            <p>Try the CZ cryoET Data Portal sample dataset:</p>
            <a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp">View Dataset 10440 - Run 16463</a>
        </div>
        
        <div class="feature">
            <h3>New Features!</h3>
            <p><strong>Class Balancing:</strong> Balance class distribution in samples using weighted sampling.</p>
            <p><strong>Mixup Augmentation:</strong> Visualize mixup augmentation which blends samples for better generalization.</p>
            <p>Try these new options with the toggles in the visualization page!</p>
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
                <code>use_class_balancing</code>: Whether to use class-balanced sampling (default: true)
            </div>
            <div class="param">
                <code>apply_mixup</code>: Whether to apply mixup augmentation for visualization (default: false)
            </div>
            <div class="example">
                <p><strong>Example:</strong></p>
                <p><a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp">/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp</a></p>
                <p><strong>With new features:</strong></p>
                <p><a href="/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp&use_class_balancing=true&apply_mixup=true">/tomogram-viz?dataset_id=10440&overlay_root=/tmp/test/&run_name=16463&voxel_spacing=10.012&tomo_type=wbp&use_class_balancing=true&apply_mixup=true</a></p>
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
    print("Press Ctrl+C to exit")
    
    # Start the server
    uvicorn.run(app, host=HOST, port=PORT)
