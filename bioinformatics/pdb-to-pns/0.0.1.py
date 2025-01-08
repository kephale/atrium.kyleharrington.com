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
import typer
import numpy as np
import gemmi
from scipy.ndimage import gaussian_filter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pdb_to_density(filename: os.PathLike, box_size: int = 128, voxel_size: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Convert PDB to density map."""
    try:
        # Read structure
        st = gemmi.read_structure(str(filename))
        
        # Extract coordinates
        coords = []
        for model in st:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        pos = atom.pos
                        coords.append([pos.x, pos.y, pos.z])
        
        coords = np.array(coords)
        
        # Center coordinates
        centroids = np.mean(coords, axis=0)
        coords = coords - centroids
        
        # Create density map
        pad = box_size // 2
        scaled_coords = coords / voxel_size + pad
        
        density, _ = np.histogramdd(
            scaled_coords,
            bins=box_size,
            range=tuple([(0, box_size - 1)] * 3),
        )
        
        # Smooth and normalize
        density = gaussian_filter(density, sigma=sigma)
        density = (density - density.min()) / (density.max() - density.min())
        
        return density
        
    except Exception as e:
        raise RuntimeError(f"Failed to process PDB file: {str(e)}")

def write_mrc(density: np.ndarray, output_path: Path):
    """Write density map to MRC format."""
    # Create grid object
    grid = gemmi.FloatGrid(density.shape[0], density.shape[1], density.shape[2])
    grid.set_unit_cell(gemmi.UnitCell(density.shape[0], density.shape[1], density.shape[2], 90, 90, 90))
    
    # Copy data to grid
    for i in range(density.shape[0]):
        for j in range(density.shape[1]):
            for k in range(density.shape[2]):
                grid.set_value(i, j, k, density[i,j,k])
    
    # Write to file
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    ccp4.write_ccp4_map(str(output_path))

def write_pns(pdb_id: str, mrc_path: Path, output_path: Path, 
              iso_value: float = 0.1, pmer_l: float = 1.2, 
              pmer_l_max: float = 3000.0, pmer_occ: float = 0.5,
              pmer_over_tol: float = 0.01):
    """Write PNS format file with parameters."""
    with open(output_path, 'w') as f:
        f.write(f"MMER_ID = pdb_{pdb_id}\n")
        f.write(f"MMER_SVOL = {str(mrc_path.absolute())}\n")
        f.write(f"MMER_ISO = {iso_value}\n")
        f.write(f"PMER_L = {pmer_l}\n")
        f.write(f"PMER_L_MAX = {pmer_l_max}\n")
        f.write(f"PMER_OCC = {pmer_occ}\n")
        f.write(f"PMER_OVER_TOL = {pmer_over_tol}\n")

class PDBConverter:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        box_size: int = 128,
        voxel_size: float = 1.0,
        sigma: float = 1.0,
        iso_value: float = 0.1
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.box_size = box_size
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.iso_value = iso_value
        
        # Create output directories
        self.mrc_dir = self.output_dir / "phantom_mrcs"
        self.pns_dir = self.output_dir / "phantom_pns"
        self.mrc_dir.mkdir(parents=True, exist_ok=True)
        self.pns_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, pdb_path: Path):
        """Process a single PDB file."""
        try:
            # Get PDB ID from filename
            pdb_id = pdb_path.stem
            
            # Create density map
            density = pdb_to_density(
                pdb_path,
                box_size=self.box_size,
                voxel_size=self.voxel_size,
                sigma=self.sigma
            )
            
            # Write MRC file
            mrc_path = self.mrc_dir / f"{pdb_id}.mrc"
            write_mrc(density, mrc_path)
            
            # Write PNS file
            pns_path = self.pns_dir / f"{pdb_id}.pns"
            write_pns(pdb_id, mrc_path, pns_path, iso_value=self.iso_value)
            
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
app = typer.Typer(help="Convert PDB files to MRC and PNS formats")

@app.command()
def convert(
    input_dir: str = typer.Option(
        ...,
        help="Directory containing PDB files"
    ),
    output_dir: str = typer.Option(
        "output",
        help="Base directory for output files"
    ),
    box_size: int = typer.Option(
        128,
        help="Size of density grid in pixels"
    ),
    voxel_size: float = typer.Option(
        1.0,
        help="Voxel size in Angstroms"
    ),
    sigma: float = typer.Option(
        1.0,
        help="Gaussian smoothing sigma"
    ),
    iso_value: float = typer.Option(
        0.1,
        help="Isosurface threshold value"
    )
):
    """
    Convert a directory of PDB files to MRC density maps and corresponding PNS files.
    The MRC files will be placed in a 'phantom_mrcs' subdirectory and PNS files in 
    a 'phantom_pns' subdirectory.
    """
    try:
        converter = PDBConverter(
            input_dir=input_dir,
            output_dir=output_dir,
            box_size=box_size,
            voxel_size=voxel_size,
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