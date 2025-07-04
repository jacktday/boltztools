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

1. Ensure you have the required dependencies:
   ```bash
   pip install rdkit numpy
   ```

2. Make sure your `chemtempgen.py` file is in the same directory or in your Python path.

3. Ensure the Boltz source code is available in the `src/` directory.

## Usage

### Command Line Interface

```bash
# Basic usage
python custom_residue_processor.py --smiles "CC1=CC=CC=C1" --name "BEN" --output "ben_residue.pkl"

# With custom backbone type
python custom_residue_processor.py --smiles "C(C(C(=O)O)N)C" --name "ALA" --output "ala_residue.pkl" --backbone-type protein

# With custom leaving atoms
python custom_residue_processor.py --smiles "CC1=CC=CC=C1" --name "BEN" --output "ben_residue.pkl" --leaving-atoms C1 H1

# Skip 3D conformer generation
python custom_residue_processor.py --smiles "CC1=CC=CC=C1" --name "BEN" --output "ben_residue.pkl" --no-3d
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

## Leaving Atoms

### Method 1: By Atom Names
```python
custom_leaving_atoms = {"N", "H1", "H2", "O", "H"}
```

### Method 2: By SMARTS Patterns (More Flexible)
```python
custom_leaving_pattern = {
    "[NX3]([H])([H])[CX4][CX3](=O)[O]": {1, 6},  # Remove N-terminal H's and C-terminal O
    "[CX3](=O)[O]": {1}  # Remove additional carboxyl O
}
```

## Examples

### Example 1: Custom Phosphoserine
```python
smiles = "C(C(C(=O)O)N)OP(=O)(O)O"  # Phosphoserine-like
resname = "SEP"

parsed_residue = processor.process_custom_residue(
    smiles=smiles,
    resname=resname,
    output_path="sep_residue.pkl",
    backbone_type="protein"
)
```

### Example 2: Custom Nucleotide
```python
smiles = "C1=NC(=NC(=N1)N)C2C(C(C(O2)COP(=O)(O)O)O)O"  # Modified adenosine
resname = "MAD"

parsed_residue = processor.process_custom_residue(
    smiles=smiles,
    resname=resname,
    output_path="mad_residue.pkl",
    backbone_type="nucleic_acid"
)
```

### Example 3: Custom Molecule with Custom Leaving Atoms
```python
smiles = "CC1=CC=CC=C1"  # Toluene
resname = "BEN"

parsed_residue = processor.process_custom_residue(
    smiles=smiles,
    resname=resname,
    output_path="ben_residue.pkl",
    backbone_type="custom",
    custom_leaving_atoms={"C1", "H1"}  # Remove methyl group
)
```

## Integration with Boltz

The generated pickle files can be used with Boltz in several ways:

### 1. Direct Loading in Boltz Code
```python
import pickle

# Load the processed residue
with open("ala_residue.pkl", "rb") as f:
    parsed_residue = pickle.load(f)

# Use in Boltz pipeline
# ... Boltz processing code ...
```

### 2. YAML Input with Custom Residues
```yaml
sequences:
  - protein:
      id: A
      sequence: "MADQLTEEQIAEFKEAFSLF"
      # The processed residue will be automatically recognized
      # if it has standard backbone atom names
```

## Key Features

### 1. **Modular Design**
Each step is implemented as a separate function, making it easy to customize the pipeline.

### 2. **Automatic Backbone Detection**
The processor can automatically detect protein and nucleic acid backbones using SMARTS patterns.

### 3. **Flexible Leaving Atom Definition**
Support for both name-based and pattern-based leaving atom specification.

### 4. **3D Conformer Generation**
Uses RDKit's ETKDG algorithm for reliable 3D conformer generation.

### 5. **Boltz Compatibility**
Outputs `ParsedResidue` objects that are directly compatible with Boltz's data structures.

### 6. **Error Handling**
Comprehensive error handling and logging throughout the pipeline.

## Troubleshooting

### Common Issues

1. **Invalid SMILES**: Ensure your SMILES string is valid and can be parsed by RDKit.

2. **Backbone Detection Fails**: If auto-detection fails, explicitly specify the backbone type.

3. **3D Conformer Generation Fails**: Some molecules may not generate 3D conformers. Use `--no-3d` to skip this step.

4. **Leaving Atoms Not Found**: Ensure your leaving atom names or patterns match atoms in the molecule.

### Debug Mode

Enable debug logging to see detailed information:
```bash
python custom_residue_processor.py --smiles "..." --name "..." --output "..." -v
```

### Validation

Check generated files:
```bash
# Load and validate pickle file
python -c "
import pickle
with open('residue.pkl', 'rb') as f:
    mol = pickle.load(f)
print(f'Atoms: {mol.GetNumAtoms()}')
print(f'Bonds: {mol.GetNumBonds()}')
print(f'Conformers: {mol.GetNumConformers()}')
"
```

## Dependencies

- **RDKit**: For molecular processing and 3D conformer generation
- **NumPy**: For numerical operations
- **Boltz**: For constraint generation (must be available in src/ directory)
- **chemtempgen**: For chemical template generation (from meeko or local file)

## File Structure

This module works alongside the existing project structure:

```
project/
├── add_cif_to_ccd.py              # Existing CIF processor
├── custom_residue_processor.py    # New SMILES processor (this module)
├── chemtempgen.py                 # Chemical template generator
├── requirements.txt               # Python dependencies
├── README.md                      # Main project documentation
└── README_custom_residue_processor.md  # This documentation
```

## Contributing

This module is designed to work alongside the existing project. When contributing:

1. Maintain compatibility with existing tools
2. Follow the established code style
3. Test with both SMILES and CIF inputs
4. Ensure Boltz integration works correctly 