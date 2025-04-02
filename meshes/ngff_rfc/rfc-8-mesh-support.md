# RFC-8: Mesh Support in OME-NGFF

Add support for mesh representations of segmented objects in OME-NGFF by adapting the Neuroglancer mesh format.

## Status

This proposal is in active development. Status: Draft

```{list-table} Record
:widths: 8, 20, 20, 20, 15, 10
:header-rows: 1
:stub-columns: 1

*   - Role
    - Name
    - GitHub Handle
    - Institution
    - Date
    - Status
*   - Author
    - Kyle Harrington
    - @kephale
    - CZI
    - 2024-01-13
    - 
*   - Updated
    - Kyle Harrington
    - @kephale
    - CZI
    - 2024-04-02
    - 
```

## Overview

This RFC proposes adding support for storing multi-resolution triangle mesh representations of segmented objects in OME-NGFF by adapting the [Neuroglancer mesh format specification](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/meshes.md).

## Background

Surface mesh representations are essential for visualization and analysis of 3D structures in bio-imaging. The current NGFF specification lacks native support for mesh representations. The Neuroglancer project provides a well-tested multi-resolution mesh format that would fill this gap.

## Proposal

Add mesh support to NGFF by integrating with the collections proposal for specifying image-segmentation relationships. The mesh data will be stored as an external artifact within the Zarr hierarchy.

### Integration with OME-NGFF Collections

Meshes are integrated into the OME-NGFF specification as members of collections. A collection can reference both images and their associated mesh representations:

```json
{
    "ome": {
        "version": "0.5",
        "collection": {
            "name": "em_reconstruction",
            "members": [
                {
                    "type": "image",
                    "path": "./raw",
                    "attributes": {
                        // image-specific attributes
                    }
                },
                {
                    "type": "labels",
                    "path": "./labels/segmentation",
                    "attributes": {
                        // labels-specific attributes
                    }
                },
                {
                    "type": "mesh",
                    "path": "./meshes",
                    "attributes": {
                        "type": "neuroglancer_multilod_draco",
                        "vertexQuantizationBits": 10,
                        "lodScaleMultiplier": 2.0,
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0, 1.0]
                            }
                        ]
                    }
                }
            ]
        }
    }
}
```

### Storage Layout

When storing mesh data within a collection:

```
[data].zarr/
  ├── zarr.json          # Collection metadata (shown above)
  ├── raw/               # Image data
  │   ├── zarr.json      # Image metadata
  │   └── ...
  ├── labels/            # Label data
  │   ├── zarr.json
  │   └── segmentation/  # Segmentation data
  │       ├── zarr.json
  │       └── ...
  └── meshes/            # Mesh data
      ├── zarr.json      # External node type metadata
      ├── info           # Mesh format metadata
      ├── 1              # Binary mesh fragment data (unsharded)
      ├── 1.index        # Binary manifest file for mesh 1
      ├── 2              # Binary mesh fragment data (unsharded)
      ├── 2.index        # Binary manifest file for mesh 2
      └── ...            # Additional meshes
```

### Mesh Directory Metadata

The mesh directory contains a `zarr.json` that identifies it as an external node:

```json
{
  "zarr_format": 3,
  "node_type": "external",
  "attributes": {
    "ome": {
      "version": "0.5"
    }
  }
}
```

The `info` file contains the mesh format metadata:

```json
{
  "@type": "neuroglancer_multilod_draco",
  "vertex_quantization_bits": 10,
  "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
  "lod_scale_multiplier": 2.0,
  "data_type": "uint64",
  "num_channels": 1,
  "type": "segmentation"
}
```

The mesh-specific metadata lives in the collection's member attributes rather than in the mesh directory's zarr.json. This allows the same mesh data to be referenced by multiple collections with different transformations or rendering settings.

### Coordinate Systems and Alignment

To ensure proper alignment between mesh and voxel data, meshes should be aligned to the voxel centers rather than the voxel edges. This means:

1. In Neuroglancer mesh fragments, vertex coordinates should be adjusted by adding a half-voxel offset (0.5, 0.5, 0.5) relative to the grid's origin.
2. When visualizing, the same coordinate transform should be applied to mesh and image/label data.
3. In the `info` file, the `transform` field can be used to specify any additional transformations needed for alignment.

The mesh vertex coordinates should be transformed as follows:

1. Denormalize from quantized space: `vertices = vertices / (2^quantization_bits - 1) * scale`
2. Add grid origin and fragment position: `vertices = vertices + grid_origin + box_position * scale`
3. Apply half-voxel correction: `vertices = vertices + [0.5, 0.5, 0.5] * scale`
4. Apply global transformation: `vertices = vertices * transform.scale + transform.translation`

### Relationship to Label Data

Each mesh ID should correspond to a label value in the associated label data. Viewers should allow for toggling between label and mesh representations, and rendering meshes with the same colors assigned to the labels.

### Multi-resolution Levels of Detail (LOD)

Meshes should be stored with multiple Levels of Detail (LOD) to support efficient visualization at different scales. The LOD levels should correspond to the resolution levels in the image and label data.

Each LOD level should follow these guidelines:

1. LOD 0 should be the highest resolution mesh.
2. Each successive LOD level should reduce vertex/face count by approximately 50%.
3. The `lod_scale_multiplier` parameter (default: 2.0) controls the geometric scaling between LOD levels.

## Implementation Notes

Implementation requires:

1. Metadata handling following NGFF conventions
2. Binary manifest and fragment handling according to the Neuroglancer Precomputed format
3. Integration with existing NGFF coordinate transform system

### Binary Manifest Format

Each mesh has a separate manifest file (`[mesh_id].index`) with the following format:

1. Chunk shape (3 x float32): Size of the smallest LOD chunks
2. Grid origin (3 x float32): Origin of the grid in physical coordinates
3. Number of LODs (uint32): Number of LOD levels
4. LOD scales (num_lods x float32): Scale factor for each LOD level
5. Vertex offsets ([num_lods, 3] x float32): Additional offset for each LOD level
6. Number of fragments per LOD (num_lods x uint32): Number of mesh fragments at each LOD
7. For each LOD, fragment positions ([num_fragments, 3] x uint32): Grid positions of fragments
8. For each LOD, fragment sizes (num_fragments x uint32): Size in bytes of each fragment

### Fragment Data Format

The fragment data is stored in a single file per mesh (`[mesh_id]`), with fragments from all LODs concatenated. Each fragment is Draco-encoded for efficient compression.

### Required Software Support

- The NGFF specification should be agnostic to the specific implementation
- Reference implementation libraries should provide:
  - Mesh import/export functionality
  - Coordinate transformation tools
  - LOD generation
  - Draco compression/decompression

## Requirements

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [IETF RFC 2119].

## Stakeholders

The main stakeholders are bio-image visualization tool developers and scientists working with 3D segmentation data.

### Socialization
* [Github issue](https://github.com/ome/ngff/issues/33)
* [Initial meeting with stakeholders](https://github.com/ome/ngff/issues/33#issuecomment-2555637903)

## Drawbacks, risks, alternatives, and unknowns

This proposal depends on Collections RFC-7.

This proposal adds complexity but provides necessary functionality for 3D visualization. The use of Draco compression creates an external dependency but provides significant storage benefits.

### Alternatives Considered

1. **STL/OBJ files**: Traditional mesh formats could be stored as binary blobs, but would lack multi-resolution support and efficient chunking.
2. **Custom NGFF mesh format**: Would require designing a complete specification rather than leveraging the well-tested Neuroglancer format.
3. **Zarr arrays for meshes**: Would be more integrated with NGFF but less efficient for mesh storage.

## Compatibility

This proposal adds new capabilities without affecting existing functionality. Reading mesh data is optional - implementations that don't support meshes can ignore the mesh metadata and data.

## Reference Implementation

A reference implementation exists at [https://github.com/kephale/atrium.kyleharrington.com/tree/main/meshes](https://github.com/kephale/atrium.kyleharrington.com/tree/main/meshes), which provides:

1. Mesh generation from labeled volumes
2. Export to Neuroglancer Precomputed format
3. Integration with OME-NGFF Zarr
4. Visualization in napari with proper alignment

## Changelog

| Date | Description | Link |
|------------|-------------------------------|------------------------------------------|
| 2024-01-13 | Initial RFC draft | TBD |
| 2024-04-02 | Updated with coordinate alignment information | TBD |
