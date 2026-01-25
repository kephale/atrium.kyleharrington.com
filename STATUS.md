# Atrium.kyleharrington.com Status

**Last Updated**: 2026-01-24 (Ralph Loop Iteration 20 - Rotifer Autonomous Agent)
**Current Branch**: main
**State**: Active Collection - Personal Executable Scripts Gallery

## Overview

Personal website hosting executable code demos and scripts. All scripts are single-file UV-style programs with dependencies and metadata in leading comments. Managed by the [atrium](https://github.com/kephale/atrium) framework.

**Website**: https://atrium.kyleharrington.com/

## Project Status: Active Collection 📚

Well-organized collection of executable scripts across multiple domains (alife, bioinformatics, meshes, visualization, etc.). Scripts are directly executable via UV without cloning.

## Collection Structure

### Current Groups

**alife** (artificial life):
- lambda-chemistry (0.0.1) - Chemical reaction systems
- evolving-swarm - Emergent swarm behavior
- langtons-ant (0.0.2) - Cellular automaton

**bioinformatics**:
- Domain-specific biological analysis scripts

**communication**:
- napari-conference (0.0.1) - Presentation demos

**cryoet** (cryogenic electron tomography):
- Microscopy and tomography utilities

**fractals**:
- Fractal generation and visualization

**grn** (gene regulatory networks):
- Network simulation and analysis

**laboratory**:
- Experimental and prototype scripts

**meshes**:
- complete_blobs_demo (0.0.1) - 3D mesh generation
- zmesh-ng-precomputed (0.0.1) - Mesh preprocessing
- generation, visualization tools

**reinforcement-learning**:
- RL algorithms and experiments

**visualization**:
- Plotting and data visualization tools

**zarr**:
- Zarr format utilities

## Usage

**Direct execution** (no cloning needed):
```bash
uv run https://atrium.kyleharrington.com/[group]/[solution]/[version].py
```

**Example**:
```bash
uv run https://atrium.kyleharrington.com/alife/lambda-chemistry/0.0.1.py
```

## Script Requirements

All scripts must be single-file with proper metadata:

```python
# /// script
# title = "Script Title"
# description = "Brief description"
# author = "Kyle Harrington <kyle.harrington@pm.me>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["keyword1", "keyword2"]
# dependencies = [
#     "numpy>=1.20.0",
#     "matplotlib>=3.0.0"
# ]
# ///
```

## Current Work

**Status**: Active collection, maintained as personal script gallery

**Maintenance tasks**:
1. Keep scripts up-to-date with latest dependencies
2. Ensure all scripts have complete metadata
3. Test scripts periodically for compatibility
4. Add new scripts from public repositories
5. Document cross-references between scripts and rotifer projects

## Next Steps

### Script Additions (Ongoing)

**From rotifer projects** (extract publication-quality demos):
- multiscale-life: Hierarchical ABM demonstrations
- symgrad: Symbolic regression examples
- physical-disentanglement: Rotation disentanglement toy problem
- flowbench-kyle: VideoMAE/VideoGMAE demos

**New categories** (as needed):
- deep-learning: Neural network examples
- optimization: Optimization algorithm demos
- physics-informed: PINNs and physics-based learning

### Documentation Updates (1 session)

- Add index of all scripts with descriptions
- Cross-reference with rotifer projects
- Document which scripts demonstrate rotifer project concepts
- Create usage examples for each major category

### Quality Assurance (Periodic)

- Test all scripts with current UV version
- Update dependencies to latest compatible versions
- Fix any broken scripts
- Remove outdated/deprecated scripts

## Blockers

None. Collection is active and maintained.

## Open PRs

None currently.

## Key Decisions

### Architecture
- **Format**: Single-file UV-style scripts (self-contained)
- **Metadata**: PEP 723 script metadata format
- **Hosting**: GitHub Pages with atrium framework
- **Execution**: Direct via UV (no cloning needed)

### Organization
- **Groups**: Topical directories (alife, bioinformatics, etc.)
- **Solutions**: Subdirectories for related scripts
- **Versions**: Semantic versioning (0.0.1, 0.1.0, etc.)

### Quality Standards
- ✅ Complete metadata (title, description, author, license)
- ✅ All dependencies listed
- ✅ MIT license (default)
- ✅ Executable without installation
- ✅ Clear documentation in comments

## Related Rotifer Projects

**Scripts demonstrating rotifer project concepts**:

**multiscale-life**:
- Could add: Hierarchical ABM demos
- Could add: Differentiable simulation examples
- Could add: AgentType/AgentPopulation tutorials

**symgrad**:
- Could add: Symbolic regression examples
- Could add: Feature engineering demos
- Could add: GPEM benchmark visualizations

**flowbench-kyle**:
- Could add: VideoMAE reconstruction demos
- Could add: Pattern formation visualizations
- Could add: GRN oscillator animations

**physical-disentanglement**:
- Could add: 2D rotation toy problem
- Could add: Embedding visualization examples

**adversarial-annotations**:
- Could add: FGSM attack demonstrations
- Could add: MicroSAM robustness examples

**consistency-learning**:
- Could add: Multi-view consistency demos
- Could add: Temporal consistency visualizations

**symbolic-forests**:
- Could add: Ensemble symbolic regression examples

## Repository Health

- **Code quality**: Clean single-file scripts, proper metadata
- **Documentation**: Clear README, atrium framework docs
- **Testing**: Scripts should be manually tested periodically
- **Dependencies**: UV handles all dependencies automatically
- **Activity**: Recent commit (3 months ago)
- **Branches**: Single main branch (clean)

## Performance Notes

**Execution**:
- UV downloads and caches dependencies automatically
- First run: Slower (dependency installation)
- Subsequent runs: Fast (cached dependencies)
- No git clone needed (direct HTTPS execution)

**Hosting**:
- GitHub Pages (free, reliable)
- Atrium framework (automatic indexing)
- Direct script access via URL

## Use Cases

1. **Quick Demos**
   - Share executable code without repositories
   - Conference presentations (napari-conference)
   - Teaching examples

2. **Personal Reference**
   - Code snippets for common tasks
   - Algorithm implementations
   - Visualization templates

3. **Public Portfolio**
   - Showcase coding style and interests
   - Demonstrate scientific programming skills
   - Provide reusable utilities

4. **Research Artifacts**
   - Publication-quality figure generation scripts
   - Reproducible analysis code
   - Supplementary materials for papers

## Integration Opportunities

### With Rotifer Projects

1. **Multiscale-life demos**
   - Extract hierarchical demo as standalone script
   - Organelle visualization examples
   - GRN network diagrams

2. **SymGrad examples**
   - Benchmark result visualizations
   - Expression simplification demos
   - Symbolic regression tutorials

3. **FlowBench artifacts**
   - Pattern formation animations
   - VideoMAE reconstruction examples
   - GRN oscillator visualizations (from Session 009-011)

4. **Consistency-learning demos**
   - Multi-view temporal consistency examples
   - Embedding visualization tools

### Quality Standards for Extracted Scripts

When adding scripts from rotifer projects:
- ✅ Self-contained (no external file dependencies)
- ✅ Generate outputs to stdout or save to current directory
- ✅ Include sample data or data generation
- ✅ Complete metadata with project reference
- ✅ Clear description of what the script demonstrates
- ✅ Minimal dependencies (prefer stdlib when possible)

## Contact

Primary maintainer: Kyle Harrington (@kephale)
Website: https://atrium.kyleharrington.com/
Repository: github.com/kephale/atrium.kyleharrington.com
Framework: github.com/kephale/atrium
Status: Active collection, maintained as personal script gallery
