# Custom Residue Processor for Boltz

This module provides functionality to process custom residues from SMILES input to Boltz-compatible format with 3D conformer generation.

## Overview

The `CustomResidueProcessor` class provides a complete pipeline to:
1. **Create molecules** from SMILES strings
2. **Identify backbone types** (protein, nucleic acid, or custom)
3. **Rename backbone atoms** to match Boltz conventions
4. **Define leaving atoms** for polymer connectivity
5. **Truncate residues** by removing leaving atoms
6. **Generate 3D conformers** using RDKit's ETKDG algorithm
7. **Convert to ParsedResidue** format with geometry constraints
8. **Save to pickle** files for Boltz use

## Installation

Installation of Meeko or Boltz is optional. But some dependencies are required: 

1. If Meeko is not installed as a package, make sure your `chemtempgen.py` file is in the same directory or in your Python path.

2. If Boltz is not installed as a package, ensure the Boltz source code is available in the `src/` directory.

## Usage

### Command Line Interface

```bash
# Basic usage
python custom_residue_processor.py --smiles "NC(CCOP(=O)(O)O)C(=O)O" --name SEP --output sep_residue.pkl --backbone-type protein --save-sdf
```

### Python API

```python
from custom_residue_processor import CustomResidueProcessor

# Initialize processor
processor = CustomResidueProcessor()

# Process a custom residue
parsed_residue = processor.process_custom_residue(
    smiles="C(C(C(=O)O)N)C",  # Alanine
    resname="ALA",
    output_path="ala_residue.pkl",
    backbone_type="protein",  # Optional: auto-detected if not specified
    generate_3d=True
)
```

## Backbone Types

### 1. Protein Backbone
- **SMARTS Pattern**: `[NX3]([H])([H])[CX4][CX3](=O)[O]`
- **Standard Atom Names**: `["N", "CA", "C", "O", "CB"]`
- **Default Leaving Atoms**: N-terminal hydrogens and C-terminal oxygen

### 2. Nucleic Acid Backbone
- **SMARTS Pattern**: `[O][PX4](=O)([O])[OX2][CX4][CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]`
- **Standard Atom Names**: `["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'"]`
- **Default Leaving Atoms**: 5' phosphate and 3' hydroxyl

### 3. Custom Backbone
- **Pattern**: Uses the entire molecule as the allowed pattern
- **Atom Names**: Sequential naming (C1, C2, O1, etc.)
- **Leaving Atoms**: Must be specified manually

### 4. Custom Leaving Atoms By SMARTS Patterns 
```python
custom_leaving_pattern = {
    "[NX3]([H])([H])[CX4][CX3](=O)[O]": {1, 6},  # Remove N-terminal H's and C-terminal O
    "[CX3](=O)[O]": {1}  # Remove additional carboxyl O
}
```

*Note: SMARTS pattern functionality is available but not demonstrated in the current examples due to complexity. This feature will be documented in future updates.*

## Examples

The following examples demonstrate different use cases for the Custom Residue Processor. Each example generates both a `.pkl` file for Boltz use and an `.sdf` file for debugging and visualization.

### Example 1: Custom Amino Acid with Protein Backbone

**Purpose**: Process a phosphorylated amino acid with explicit protein backbone specification.

**Molecule**: Phosphoserine-like residue
- **SMILES**: `C(C(C(=O)O)N)OP(=O)(O)O`
- **Residue Name**: `SEP`
- **Backbone Type**: `protein` (explicitly specified)

**Code**:
```python
processor = CustomResidueProcessor()
parsed_residue = processor.process_custom_residue(
    smiles="C(C(C(=O)O)N)OP(=O)(O)O",
    resname="SEP",
    output_path="sep_residue.pkl",
    backbone_type="protein",  # Explicitly specify protein backbone
    generate_3d=True,
    save_sdf=True  # Save SDF for debugging
)
```

**Output Files**:
- `sep_residue.pkl`: Boltz-compatible residue file
- `sep_residue.sdf`: 3D structure for visualization

**Key Features**:
- Uses standard protein backbone atom naming (N, CA, C, O, CB)
- Automatically removes N-terminal hydrogens and C-terminal oxygen
- Generates 3D conformer with proper geometry constraints

### Example 2: Custom Nucleotide with Nucleic Acid Backbone

**Purpose**: Process a modified nucleotide with nucleic acid backbone.

**Molecule**: AMP (Adenosine Monophosphate)
- **SMILES**: `O=P(O)(O)OCC3OC(n2cnc1c(ncnc12)N)C(O)C3O`
- **Residue Name**: `AMP`
- **Backbone Type**: `nucleic_acid` (explicitly specified)

**Code**:
```python
processor = CustomResidueProcessor()
parsed_residue = processor.process_custom_residue(
    smiles="O=P(O)(O)OCC3OC(n2cnc1c(ncnc12)N)C(O)C3O",
    resname="AMP",
    output_path="amp_residue.pkl",
    backbone_type="nucleic_acid",  # Explicitly specify nucleic acid backbone
    generate_3d=True,
    save_sdf=True  # Save SDF for debugging
)
```

**Output Files**:
- `amp_residue.pkl`: Boltz-compatible residue file
- `amp_residue.sdf`: 3D structure for visualization

**Key Features**:
- Uses standard nucleic acid backbone atom naming (P, O5', C5', etc.)
- Removes 5' phosphate and 3' hydroxyl groups
- Handles complex ring systems and heteroatoms

### Example 3: Custom Molecule with Custom Backbone

**Purpose**: Process a small organic molecule with custom backbone and specific leaving atoms.

**Molecule**: Toluene (methylbenzene)
- **SMILES**: `CC1=CC=CC=C1`
- **Residue Name**: `BEN`
- **Backbone Type**: `custom`
- **Custom Leaving Atoms**: `{"C1", "H1"}` (methyl group and its hydrogen)

**Code**:
```python
processor = CustomResidueProcessor()
custom_leaving_atoms = {"C1", "H1"}  # Remove methyl group and its H

parsed_residue = processor.process_custom_residue(
    smiles="CC1=CC=CC=C1",
    resname="BEN",
    output_path="ben_residue.pkl",
    backbone_type="custom",  # Custom backbone
    custom_leaving_atoms=custom_leaving_atoms,
    generate_3d=True,
    save_sdf=True  # Save SDF for debugging
)
```

**Output Files**:
- `ben_residue.pkl`: Boltz-compatible residue file
- `ben_residue.sdf`: 3D structure for visualization

**Key Features**:
- Uses sequential atom naming (C1, C2, C3, etc.)
- Manually specifies which atoms to remove
- Demonstrates custom backbone processing

### Example 4: Auto-Detection of Backbone Type

**Purpose**: Let the processor automatically detect the backbone type based on molecular structure.

**Molecule**: Alanine (standard amino acid)
- **SMILES**: `C(C(C(=O)O)N)C`
- **Residue Name**: `ALA`
- **Backbone Type**: Auto-detected as `protein`

**Code**:
```python
processor = CustomResidueProcessor()
parsed_residue = processor.process_custom_residue(
    smiles="C(C(C(=O)O)N)C",
    resname="ALA",
    output_path="ala_residue.pkl",
    # No backbone_type specified - will auto-detect
    generate_3d=True,
    save_sdf=True  # Save SDF for debugging
)
```

**Output Files**:
- `ala_residue.pkl`: Boltz-compatible residue file
- `ala_residue.sdf`: 3D structure for visualization

**Key Features**:
- Automatic detection of protein backbone pattern
- No manual specification required
- Useful for batch processing of known molecule types

## Running the Examples

To run all examples at once:

```bash
python example_usage.py
```

This will generate the following files:
- `sep_residue.pkl` and `sep_residue.sdf` (phosphoserine)
- `amp_residue.pkl` and `amp_residue.sdf` (AMP)
- `ben_residue.pkl` and `ben_residue.sdf` (toluene)
- `ala_residue.pkl` and `ala_residue.sdf` (alanine)

## Debugging with SDF Files

The generated SDF files can be opened in molecular viewers to verify:
- **3D conformer quality**: Are the structures reasonable?
- **Atom connectivity**: Are bonds correct?
- **Leaving atom removal**: Were the right atoms removed?
- **Backbone identification**: Are backbone atoms properly named?

**Recommended viewers**:
- PyMOL, VMD, Chimera (desktop applications)
- MolView, PubChem 3D Viewer (online tools)