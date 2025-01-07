# /// script
# title = "Convert Directory of PDBs to PNS Files"
# description = "Processes a directory of PDB files to create corresponding PNS files using pure Python"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
# keywords = ["pdbs", "density", "structural biology", "pns"]
# repository = "https://github.com/kephale/pdb-to-pns"
# documentation = "https://github.com/kephale/pdb-to-pns#readme"
# classifiers = [
#   "Development Status :: 4 - Beta",
#   "Intended Audience :: Science/Research", 
#   "License :: OSI Approved :: MIT License",
#   "Programming Language :: Python :: 3.9",
#   "Topic :: Scientific/Engineering :: Bio-Informatics",
# ]
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "gemmi",
#   "scipy",
#   "typer",
#   "scikit-image",
# ]
# ///

import os
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union, Tuple
import typer
import numpy as np
import gemmi
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
from skimage import measure

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pdb_to_coordinates(filename: os.PathLike) -> np.ndarray:
    """Read a PDB/mmCIF file and return the atomic coordinates."""
    try:
        # First try reading as PDB
        st = gemmi.read_structure(str(filename))
        
        # Extract coordinates from all atoms
        coords = []
        for model in st:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        pos = atom.pos
                        coords.append([pos.x, pos.y, pos.z])
        
        coords = np.array(coords)
        
    except Exception as e:
        logger.debug(f"Failed to read as PDB, trying CIF format: {str(e)}")
        try:
            # Try reading as mmCIF
            doc = gemmi.cif.read_file(str(filename))
            block = doc.sole_block()
            data = block.find("_atom_site.", ["Cartn_x", "Cartn_y", "Cartn_z"])
            
            coords = np.stack(
                [[float(r) for r in data.column(idx)] for idx in range(3)],
                axis=-1,
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to read file in both PDB and CIF formats: {str(e2)}")

    # Center the molecule in XYZ
    centroids = np.mean(coords, axis=0)
    coords = coords - centroids

    return coords

def write_pns_file(pns_path: Path, vertices: np.ndarray, faces: np.ndarray):
    """Write a PNS format file."""
    with open(pns_path, 'w') as f:
        # Write header
        f.write(f"{len(vertices)} {len(faces)}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces (adjust indices to 1-based)
        for face in faces:
            f.write(f"3 {face[0]+1} {face[1]+1} {face[2]+1}\n")

class PDBtoPNSConverter:
    """Convert PDB files to PNS format using density-based surface extraction."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        voxel_size: float = 1.0,
        box_size: int = 128,
        sigma: float = 1.0,
        iso_value: float = 0.5
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.voxel_size = voxel_size
        self.box_size = box_size
        self.sigma = sigma
        self.iso_value = iso_value
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_density_map(self, coords: np.ndarray) -> np.ndarray:
        """Create a density map from atomic coordinates."""
        # Scale coordinates to grid
        pad = self.box_size // 2
        scaled_coords = coords / self.voxel_size + pad
        
        # Create density map using histogramdd
        density, _ = np.histogramdd(
            scaled_coords,
            bins=self.box_size,
            range=tuple([(0, self.box_size - 1)] * 3),
        )
        
        # Apply Gaussian filter to smooth
        density = gaussian_filter(density, sigma=self.sigma)
        
        # Normalize
        density = (density - density.min()) / (density.max() - density.min())
        
        return density
    
    def extract_surface(self, density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract surface vertices and faces using marching cubes."""
        # Run marching cubes from skimage
        vertices, faces, normals, values = measure.marching_cubes(density, self.iso_value)
        
        # Scale vertices back to original coordinate system
        vertices = vertices * self.voxel_size - (self.box_size * self.voxel_size) / 2
        
        return vertices, faces
    
    def process_file(self, pdb_path: Path):
        """Process a single PDB file."""
        try:
            # Read coordinates
            coords = pdb_to_coordinates(pdb_path)
            
            # Create density map
            density = self.create_density_map(coords)
            
            # Extract surface
            vertices, faces = self.extract_surface(density)
            
            # Write PNS file
            pns_path = self.output_dir / f"{pdb_path.stem}.pns"
            write_pns_file(pns_path, vertices, faces)
            
            logger.info(f"Successfully processed {pdb_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {pdb_path.name}: {str(e)}")
    
    def process_all(self):
        """Process all PDB files in input directory."""
        pdb_files = list(self.input_dir.glob("*.pdb"))
        logger.info(f"Found {len(pdb_files)} PDB files to process")
        
        for pdb_path in pdb_files:
            self.process_file(pdb_path)

# Create Typer app
app = typer.Typer(help="Convert PDB files to PNS format")

@app.command()
def convert(
    input_dir: str = typer.Option(
        ...,  # Required argument
        help="Directory containing PDB files"
    ),
    output_dir: str = typer.Option(
        "pns_output",
        help="Directory for output PNS files"
    ),
    voxel_size: float = typer.Option(
        1.0,
        help="Voxel size in Angstroms"
    ),
    box_size: int = typer.Option(
        128,
        help="Size of density grid in pixels"
    ),
    sigma: float = typer.Option(
        1.0,
        help="Gaussian smoothing sigma"
    ),
    iso_value: float = typer.Option(
        0.5,
        help="Isosurface threshold value"
    )
):
    """
    Convert a directory of PDB files to PNS format.
    
    This tool:
    1. Reads atomic coordinates from PDB files
    2. Creates a density map
    3. Extracts isosurfaces using marching cubes
    4. Outputs the surfaces in PNS format
    """
    try:
        # Create and run converter
        converter = PDBtoPNSConverter(
            input_dir=input_dir,
            output_dir=output_dir,
            voxel_size=voxel_size,
            box_size=box_size,
            sigma=sigma,
            iso_value=iso_value
        )
        
        converter.process_all()
        typer.echo("Processing complete!")
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()