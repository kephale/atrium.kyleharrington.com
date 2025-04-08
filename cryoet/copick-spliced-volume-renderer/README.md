# Copick Spliced Volume Renderer

Renders orthogonal views of spliced 3D volumes by combining synthetic and experimental CryoET data.

## Description

This script creates spliced 3D volumes by extracting structures from synthetic CryoET data (using segmentation masks) and inserting them into experimental CryoET data. The script generates orthogonal views and comparison visualizations for each spliced volume.

## Usage

```bash
uv run https://atrium.kyleharrington.com/cryoet/copick-spliced-volume-renderer/main.py \
  --exp-dataset-id 10440 \
  --synth-dataset-id 10441 \
  --output-dir ./spliced_results
```

## Parameters

- `--exp-dataset-id`: Dataset ID for experimental data (default: 10440)
- `--synth-dataset-id`: Dataset ID for synthetic data with segmentation masks (default: 10441)
- `--overlay-root`: Root directory for overlay storage (default: "/tmp/test/")
- `--voxel-spacing`: Target voxel spacing for tomograms (default: 10.0)
- `--tomo-type`: Tomogram type to use (default: "wbp")
- `--num-examples`: Number of example pairs to create (default: 5)
- `--structures-per-mask`: Number of structures to extract per mask (default: 1)
- `--min-structure-size`: Minimum structure size in voxels (default: 500)
- `--blend-sigma`: Sigma for Gaussian blending at boundaries (default: 2.0)
- `--output-dir`: Directory to save output files (default: "./spliced_volumes")
- `--colormap`: Matplotlib colormap for rendering (default: "viridis")
- `--save-volumes`: Save volume data as numpy arrays
