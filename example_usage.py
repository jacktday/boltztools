#!/usr/bin/env python3
"""
Example usage of the Custom Residue Processor

This script demonstrates how to use the CustomResidueProcessor class
to process different types of custom residues.
"""

import sys
import os

# Try to import custom_residue_processor with fallbacks
try:
    from custom_residue_processor import CustomResidueProcessor
except ImportError:
    # Add Boltz src to path for imports when not installed as package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    try:
        from custom_residue_processor import CustomResidueProcessor
    except ImportError:
        raise ImportError("Could not import custom_residue_processor. Please ensure the script is in the current directory.")

from pathlib import Path


def example_protein_backbone():
    """Example: Process a custom amino acid with protein backbone."""
    print("=== Example 1: Custom Amino Acid with Protein Backbone ===")
    
    processor = CustomResidueProcessor()
    
    # Example: Custom phosphoserine-like residue
    smiles = "C(C(C(=O)O)N)OP(=O)(O)O"  # Phosphoserine-like
    resname = "SEP"
    
    try:
        parsed_residue = processor.process_custom_residue(
            smiles=smiles,
            resname=resname,
            output_path="sep_residue.pkl",
            backbone_type="protein",  # Explicitly specify protein backbone
            generate_3d=True,
            save_sdf=True  # Save SDF for debugging
        )
        
        print(f"✓ Successfully processed {resname}")
        print(f"  - SMILES: {smiles}")
        print(f"  - Backbone type: protein")
        print(f"  - Atoms: {len(parsed_residue.atoms)}")
        print(f"  - Bonds: {len(parsed_residue.bonds)}")
        
    except Exception as e:
        print(f"✗ Failed to process {resname}: {e}")


def example_nucleic_acid_backbone():
    """Example: Process a custom nucleotide with nucleic acid backbone."""
    print("\n=== Example 2: Custom Nucleotide with Nucleic Acid Backbone ===")
    
    processor = CustomResidueProcessor()
    
    # Example: Custom modified nucleotide
    smiles = "O=P(O)(O)OCC3OC(n2cnc1c(ncnc12)N)C(O)C3O"  # AMP
    resname = "AMP"
    
    try:
        parsed_residue = processor.process_custom_residue(
            smiles=smiles,
            resname=resname,
            output_path="amp_residue.pkl",
            backbone_type="nucleic_acid",  # Explicitly specify nucleic acid backbone
            generate_3d=True,
            save_sdf=True  # Save SDF for debugging
        )
        
        print(f"✓ Successfully processed {resname}")
        print(f"  - SMILES: {smiles}")
        print(f"  - Backbone type: nucleic_acid")
        print(f"  - Atoms: {len(parsed_residue.atoms)}")
        print(f"  - Bonds: {len(parsed_residue.bonds)}")
        
    except Exception as e:
        print(f"✗ Failed to process {resname}: {e}")


def example_custom_backbone():
    """Example: Process a custom molecule with custom backbone and leaving atoms."""
    print("\n=== Example 3: Custom Molecule with Custom Backbone ===")
    
    processor = CustomResidueProcessor()
    
    # Example: Benzene derivative with custom connectivity
    smiles = "CC1=CC=CC=C1"  # Toluene
    resname = "BEN"
    
    # Define custom leaving atoms for this molecule
    custom_leaving_atoms = {"C1", "H1"}  # Remove methyl group and its H
    
    try:
        parsed_residue = processor.process_custom_residue(
            smiles=smiles,
            resname=resname,
            output_path="ben_residue.pkl",
            backbone_type="custom",  # Custom backbone
            custom_leaving_atoms=custom_leaving_atoms,
            generate_3d=True,
            save_sdf=True  # Save SDF for debugging
        )
        
        print(f"✓ Successfully processed {resname}")
        print(f"  - SMILES: {smiles}")
        print(f"  - Backbone type: custom")
        print(f"  - Custom leaving atoms: {custom_leaving_atoms}")
        print(f"  - Atoms: {len(parsed_residue.atoms)}")
        print(f"  - Bonds: {len(parsed_residue.bonds)}")
        
    except Exception as e:
        print(f"✗ Failed to process {resname}: {e}")


def example_auto_detection():
    """Example: Let the processor auto-detect backbone type."""
    print("\n=== Example 4: Auto-Detection of Backbone Type ===")
    
    processor = CustomResidueProcessor()
    
    # Example: Standard amino acid (should auto-detect as protein)
    smiles = "C(C(C(=O)O)N)C"  # Alanine
    resname = "ALA"
    
    try:
        parsed_residue = processor.process_custom_residue(
            smiles=smiles,
            resname=resname,
            output_path="ala_residue.pkl",
            # No backbone_type specified - will auto-detect
            generate_3d=True,
            save_sdf=True  # Save SDF for debugging
        )
        
        print(f"✓ Successfully processed {resname}")
        print(f"  - SMILES: {smiles}")
        print(f"  - Auto-detected backbone type")
        print(f"  - Atoms: {len(parsed_residue.atoms)}")
        print(f"  - Bonds: {len(parsed_residue.bonds)}")
        
    except Exception as e:
        print(f"✗ Failed to process {resname}: {e}")


def example_custom_leaving_pattern():
    """Example: Use custom SMARTS patterns for leaving atoms."""
    print("\n=== Example 5: Custom SMARTS Patterns for Leaving Atoms ===")
    
    processor = CustomResidueProcessor()
    
    # Example: Complex molecule with custom leaving pattern
    smiles = "C(C(C(=O)O)N)C(C(=O)O)O"  # Aspartic acid
    resname = "ASP"
    
    # Define custom leaving pattern using SMARTS
    custom_leaving_pattern = {
        "[NX3]([H])([H])[CX4][CX3](=O)[O]": {1, 6},  # Remove N-terminal H's and C-terminal O
        "[CX3](=O)[O]": {1}  # Remove additional carboxyl O
    }
    
    try:
        parsed_residue = processor.process_custom_residue(
            smiles=smiles,
            resname=resname,
            output_path="asp_residue.pkl",
            backbone_type="protein",
            custom_leaving_pattern=custom_leaving_pattern,
            generate_3d=True,
            save_sdf=True  # Save SDF for debugging
        )
        
        print(f"✓ Successfully processed {resname}")
        print(f"  - SMILES: {smiles}")
        print(f"  - Backbone type: protein")
        print(f"  - Custom leaving pattern: {custom_leaving_pattern}")
        print(f"  - Atoms: {len(parsed_residue.atoms)}")
        print(f"  - Bonds: {len(parsed_residue.bonds)}")
        
    except Exception as e:
        print(f"✗ Failed to process {resname}: {e}")


def main():
    """Run all examples."""
    print("Custom Residue Processor Examples")
    print("=" * 50)
    
    # Run all examples
    example_protein_backbone()
    example_nucleic_acid_backbone()
    example_custom_backbone()
    example_auto_detection()
    example_custom_leaving_pattern()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the generated .pkl files for the processed residues.")
    print("SDF files have been generated for debugging purposes:")
    print("  - sep_residue.sdf (phosphoserine)")
    print("  - mad_residue.sdf (modified adenosine)")
    print("  - ben_residue.sdf (toluene)")
    print("  - ala_residue.sdf (alanine)")
    print("  - asp_residue.sdf (aspartic acid)")
    print("\nYou can open these SDF files in molecular viewers like:")
    print("  - PyMOL, VMD, Chimera, or online viewers")
    print("  - To verify 3D conformer generation and atom connectivity")


if __name__ == "__main__":
    main() 