#!/usr/bin/env python3
"""
Custom Residue Processor for Boltz

This script integrates chemtempgen.py with Boltz functions to process custom residues
from SMILES input to Boltz-compatible format.

Usage:
    python custom_residue_processor.py --smiles "CC1=CC=CC=C1" --name "BEN" --output "ben_residue.pkl"
"""

import sys
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import logging
import numpy as np

# Try to import Boltz modules, fallback to src path if not installed
try:
    from boltz.data.parse.mmcif import parse_ccd_residue, ParsedResidue, ParsedAtom, ParsedBond
    from boltz.data.parse.mmcif_with_constraints import parse_ccd_residue as parse_ccd_residue_with_constraints
    from boltz.data.const import token_ids
except ImportError:
    # Add Boltz src to path for imports when not installed as package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from boltz.data.parse.mmcif import parse_ccd_residue, ParsedResidue, ParsedAtom, ParsedBond
    from boltz.data.parse.mmcif_with_constraints import parse_ccd_residue as parse_ccd_residue_with_constraints
    from boltz.data.const import token_ids

# Try to import chemtempgen with fallbacks
try:
    # First try meeko.chemtempgen
    from meeko.chemtempgen import ChemicalComponent, ChemicalComponent_LoggingControler
except ImportError:
    try:
        # Then try local chemtempgen.py
        from chemtempgen import ChemicalComponent, ChemicalComponent_LoggingControler
    except ImportError:
        raise ImportError("Could not import chemtempgen. Please install meeko or ensure chemtempgen.py is in the current directory.")

# Import RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_constraints(constraints, key, transpose=False):
    """Extract constraint data from constraint objects."""
    import numpy as np
    
    if not constraints:
        return np.array([])
    
    result = np.array(list(
        map(
            lambda obj: getattr(obj, key),
            constraints
        )
    ))

    if transpose:
        return result.transpose()
    
    return result


def add_pickled_prop(mol, name, value):
    """Add a pickled property to an RDKit molecule."""
    mol.SetProp(
        name,
        pickle.dumps(value).hex()
    )


class CustomResidueProcessor:
    """
    A modular processor for converting SMILES to Boltz-compatible residue format.
    """
    
    def __init__(self):
        """Initialize the processor with default settings."""
        self.backbone_patterns = {
            "protein": {
                "smarts": "[NX3]([H])([H])[CX4][CX3](=O)[O]",
                "atom_names": ["N", "CA", "C", "O", "CB"],
                "leaving_atoms": set(),  # Don't remove by atom names - too broad
                "leaving_pattern": {"[NX3]([H])([H])[CX4][CX3](=O)[O]": {1, 6}}
            },
            "nucleic_acid": {
                "smarts": "[O][PX4](=O)([O])[OX2][CX4][CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]",
                "atom_names": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
                "leaving_atoms": set(),  # Don't remove by atom names - too broad
                "leaving_pattern": {
                    "[O][PX4](=O)([O])[OX2][CX4]": {0},  # Remove first O from phosphate
                    "[CX4]1[OX2][CX4][CX4][CX4]1[OX2][H]": {6}  # Remove terminal OH from sugar
                }
            }
        }
        

    
    def create_molecule_from_smiles(self, smiles: str, resname: str) -> ChemicalComponent:
        """
        Create a ChemicalComponent from SMILES string.
        
        Args:
            smiles: SMILES string
            resname: Residue name
            
        Returns:
            ChemicalComponent object
        """
        # Create RDKit molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens and sanitize with proper aromaticity handling
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        
        # Ensure explicit hydrogens are present
        mol = Chem.AddHs(mol, addCoords=True)
        
        # Initialize ring information
        Chem.GetSymmSSSR(mol)
        
        # Generate SMILES with explicit hydrogens and atom names
        smiles_exh = Chem.MolToSmiles(mol, allHsExplicit=True)
        
        # Generate atom names (simple sequential naming)
        atom_names = []
        for i, atom in enumerate(mol.GetAtoms()):
            element = atom.GetSymbol()
            atom_names.append(f"{element}{i+1}")
            atom.SetProp('atom_id', f"{element}{i+1}")
        
        return ChemicalComponent(mol, resname, smiles_exh, atom_names)
    
    def identify_backbone_type(self, cc: ChemicalComponent) -> str:
        """
        Identify if the molecule has protein or nucleic acid backbone patterns.
        
        Args:
            cc: ChemicalComponent object
            
        Returns:
            String indicating backbone type ("protein", "nucleic_acid", or "custom")
        """
        
        mol = cc.rdkit_mol
        
        # Check for protein backbone first (prioritize protein over nucleic acid)
        protein_pattern = Chem.MolFromSmarts(self.backbone_patterns["protein"]["smarts"])
        if mol.GetSubstructMatch(protein_pattern):
            return "protein"
        
        # Check for nucleic acid backbone only if no protein backbone found
        na_pattern = Chem.MolFromSmarts(self.backbone_patterns["nucleic_acid"]["smarts"])
        if mol.GetSubstructMatch(na_pattern):
            return "nucleic_acid"
        
        return "custom"
    

    
    def rename_backbone_atoms(self, cc: ChemicalComponent, backbone_type: str) -> ChemicalComponent:
        """
        Rename backbone atoms to match standard Boltz naming conventions.
        
        Args:
            cc: ChemicalComponent object
            backbone_type: Type of backbone ("protein", "nucleic_acid", "custom")
            
        Returns:
            Modified ChemicalComponent with renamed atoms
        """
        
        mol = cc.rdkit_mol
        
        # For custom backbones, don't rename atoms - keep original names
        if backbone_type == "custom":
            logger.info("Custom backbone detected - keeping original atom names")
            return cc
        
        # For nucleic acids, use generic backbone names
        if backbone_type == "nucleic_acid":
            # Use generic nucleic acid backbone names for all nucleotides
            standard_names = self.backbone_patterns[backbone_type]["atom_names"]
            logger.info(f"Using generic nucleic acid atom names for {cc.resname}")
        elif backbone_type in self.backbone_patterns:
            # For protein backbones, use standard names
            standard_names = self.backbone_patterns[backbone_type]["atom_names"]
        else:
            logger.warning(f"No backbone pattern found for {backbone_type}")
            return cc
        
        # Create a copy for modification
        rwmol = Chem.RWMol(mol)
        
        # Rename atoms to standard names
        for i, atom in enumerate(rwmol.GetAtoms()):
            if i < len(standard_names):
                atom.SetProp('atom_id', standard_names[i])
            else:
                # For additional atoms beyond standard names, keep original symbol
                atom.SetProp('atom_id', atom.GetSymbol())
        
        # Update the ChemicalComponent
        cc.rdkit_mol = rwmol.GetMol()
        cc.smiles_exh, cc.atom_name = self._get_smiles_with_atom_names(cc.rdkit_mol)
        
        return cc
    
    def _get_smiles_with_atom_names(self, mol) -> Tuple[str, List[str]]:
        """
        Get SMILES with atom names in the order of SMILES output.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Tuple of (SMILES string, list of atom names)
        """
        
        smiles_exh = Chem.MolToSmiles(mol, allHsExplicit=True)
        smiles_atom_output_order = mol.GetProp('_smilesAtomOutputOrder')
        
        # Parse atom output order
        delimiters = ('[', ']', ',')
        for delimiter in delimiters:
            smiles_atom_output_order = smiles_atom_output_order.replace(delimiter, ' ')
        smiles_output_order = [int(x) for x in smiles_atom_output_order.split()]
        
        # Create atom_id_dict with fallback for missing properties
        atom_id_dict = {}
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            # Try to get atom_id property, fallback to atom symbol + index
            try:
                atom_name = atom.GetProp('atom_id')
            except:
                atom_name = f"{atom.GetSymbol()}{atom_idx+1}"
            atom_id_dict[atom_idx] = atom_name
        
        atom_names = [atom_id_dict[atom_i] for atom_i in smiles_output_order]
        
        return smiles_exh, atom_names
    
    def define_leaving_atoms(self, cc: ChemicalComponent, backbone_type: str, 
                           custom_leaving_atoms: Optional[Set[str]] = None,
                           custom_leaving_pattern: Optional[Dict[str, Set[int]]] = None) -> ChemicalComponent:
        """
        Define leaving atoms for polymer connectivity.
        
        Args:
            cc: ChemicalComponent object
            backbone_type: Type of backbone
            custom_leaving_atoms: Custom set of leaving atom names
            custom_leaving_pattern: Custom leaving atom patterns
            
        Returns:
            Modified ChemicalComponent with leaving atoms defined
        """
        if backbone_type in self.backbone_patterns:
            pattern_info = self.backbone_patterns[backbone_type]
            leaving_atoms = custom_leaving_atoms or pattern_info["leaving_atoms"]
            leaving_pattern = custom_leaving_pattern or pattern_info["leaving_pattern"]
        else:
            leaving_atoms = custom_leaving_atoms or set()
            leaving_pattern = custom_leaving_pattern or {}
        
        # Mark leaving atoms in the molecule
        mol = cc.rdkit_mol
        for atom in mol.GetAtoms():
            atom_name = atom.GetProp('atom_id')
            if atom_name in leaving_atoms:
                atom.SetProp('pdbx_leaving_atom_flag', 'Y')
            else:
                atom.SetProp('pdbx_leaving_atom_flag', 'N')
        
        return cc
    
    def reassign_atom_names(self, mol):
        """Ensure all atoms have 'atom_id' and 'name' properties set."""
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetProp('atom_id', f"{atom.GetSymbol()}{i+1}")
            atom.SetProp('name', f"{atom.GetSymbol()}{i+1}")

    def truncate_residue(self, cc: ChemicalComponent, backbone_type: str,
                         custom_leaving_atoms: Optional[Set[str]] = None,
                         custom_leaving_pattern: Optional[Dict[str, Set[int]]] = None) -> ChemicalComponent:
        """
        Truncate the residue by removing leaving atoms.
        
        Args:
            cc: ChemicalComponent object
            backbone_type: Type of backbone
            custom_leaving_atoms: Custom leaving atom names
            custom_leaving_pattern: Custom leaving atom patterns
            
        Returns:
            Truncated ChemicalComponent
        """
        
        # Debug: Check molecule before truncation
        mol_before = cc.rdkit_mol
        logger.info(f"Before truncation: {mol_before.GetNumAtoms()} atoms, {mol_before.GetNumBonds()} bonds")
        
        # Check for rings before truncation
        rings_before = mol_before.GetRingInfo().NumRings()
        logger.info(f"Before truncation: {rings_before} rings")
        
        if backbone_type in self.backbone_patterns:
            pattern_info = self.backbone_patterns[backbone_type]
            allowed_smarts = pattern_info["smarts"]
            leaving_atoms = custom_leaving_atoms or pattern_info["leaving_atoms"]
            leaving_pattern = custom_leaving_pattern or pattern_info["leaving_pattern"]
        else:
            # For custom backbones, use the entire molecule as allowed pattern
            allowed_smarts = Chem.MolToSmarts(cc.rdkit_mol)
            leaving_atoms = custom_leaving_atoms or set()
            leaving_pattern = custom_leaving_pattern or {}
        
        # Truncate the residue
        with ChemicalComponent_LoggingControler():
            cc = cc.make_embedded(
                allowed_smarts=allowed_smarts,
                leaving_names=leaving_atoms,
                leaving_smarts_loc=leaving_pattern
            )
        
        # Debug: Check molecule after truncation
        mol_after = cc.rdkit_mol
        logger.info(f"After truncation: {mol_after.GetNumAtoms()} atoms, {mol_after.GetNumBonds()} bonds")
        
        # Re-initialize ring information after truncation
        try:
            Chem.GetSymmSSSR(mol_after)
            rings_after = mol_after.GetRingInfo().NumRings()
            logger.info(f"After truncation: {rings_after} rings")
        except Exception as e:
            logger.warning(f"Failed to initialize ring info after truncation: {e}")
            # Re-sanitize the molecule to fix ring info
            Chem.SanitizeMol(mol_after)
            Chem.GetSymmSSSR(mol_after)
            rings_after = mol_after.GetRingInfo().NumRings()
            logger.info(f"After re-sanitization: {rings_after} rings")
        
        # Check for problematic aromatic atoms after truncation
        problematic_atoms = []
        for i, atom in enumerate(mol_after.GetAtoms()):
            if atom.GetIsAromatic() and not atom.IsInRing():
                problematic_atoms.append((i, atom.GetSymbol()))
        
        if problematic_atoms:
            logger.warning(f"After truncation: Found {len(problematic_atoms)} atoms marked aromatic but not in rings: {problematic_atoms}")
        
        # After truncation, reassign atom names
        self.reassign_atom_names(cc.rdkit_mol)
        return cc
    
    def check_atom_ids(self, mol, context=""):
        missing = [i for i, atom in enumerate(mol.GetAtoms()) if not atom.HasProp('atom_id')]
        if missing:
            logger.warning(f"Missing atom_id on atoms at indices {missing} {context}")

    def generate_3d_conformer(self, cc: ChemicalComponent) -> ChemicalComponent:
        mol = cc.rdkit_mol

        # Debug aromaticity issues
        logger.info(f"Starting conformer generation for {cc.resname}")
        logger.info(f"Molecule has {mol.GetNumAtoms()} atoms")

        # Check for problematic aromatic atoms
        problematic_atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetIsAromatic() and not atom.IsInRing():
                problematic_atoms.append((i, atom.GetSymbol()))
        if problematic_atoms:
            logger.warning(f"Found {len(problematic_atoms)} atoms marked aromatic but not in rings: {problematic_atoms}")

        # Check atom_id before conformer generation
        self.check_atom_ids(mol, context='before conformer gen')

        # Generate 3D conformer with proper error handling
        try:
            # Don't add hydrogens - keep implicit hydrogens at broken ends
            logger.info("Keeping implicit hydrogens at broken ends (no AddHs)")
            # Reassign atom properties
            self.reassign_atom_names(mol)

            logger.info("Sanitizing molecule...")
            try:
                Chem.SanitizeMol(mol)
                logger.info("Successfully sanitized molecule")
                # Reassign atom properties after SanitizeMol
                self.reassign_atom_names(mol)
            except Exception as e:
                logger.error(f"Failed to sanitize molecule: {e}")
                raise

            logger.info("Initializing ring information...")
            try:
                Chem.GetSymmSSSR(mol)
                logger.info("Successfully initialized ring information")
            except Exception as e:
                logger.error(f"Failed to initialize ring information: {e}")
                raise

            logger.info("Attempting ETKDG conformer generation...")
            try:
                conf_id = AllChem.EmbedMolecule(mol, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
                if conf_id < 0:
                    raise ValueError("Standard embedding failed")
                logger.info(f"Successfully generated conformer with ID: {conf_id}")
                # Reassign atom properties after EmbedMolecule
                self.reassign_atom_names(mol)
            except Exception as e:
                logger.error(f"Failed ETKDG conformer generation: {e}")
                raise

            logger.info("Optimizing molecule...")
            try:
                AllChem.MMFFOptimizeMolecule(mol)
                logger.info("Successfully optimized with MMFF")
            except Exception as e:
                logger.warning(f"MMFF optimization failed: {e}, trying UFF...")
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                    logger.info("Successfully optimized with UFF")
                except Exception as e2:
                    logger.warning(f"UFF optimization also failed: {e2}")

            cc.rdkit_mol = mol
            cc.smiles_exh, cc.atom_name = self._get_smiles_with_atom_names(mol)
            logger.info(f"Successfully generated 3D conformer for {cc.resname}")
            
        except Exception as e:
            logger.warning(f"Failed to generate 3D conformer: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Try alternative embedding method
            try:
                logger.info("Trying alternative embedding method...")
                conf_id = AllChem.EmbedMolecule(mol, randomSeed=42, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
                if conf_id < 0:
                    raise ValueError("Alternative embedding failed")
                AllChem.UFFOptimizeMolecule(mol)
                cc.rdkit_mol = mol
                cc.smiles_exh, cc.atom_name = self._get_smiles_with_atom_names(mol)
                logger.info(f"Generated 3D conformer with alternative method for {cc.resname}")
            except Exception as e2:
                logger.error(f"All 3D conformer generation methods failed: {e2}")
                import traceback
                logger.error(f"Alternative method traceback: {traceback.format_exc()}")
                
                # Last resort: try basic embedding
                try:
                    logger.info("Trying basic embedding as last resort...")
                    conf_id = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100)
                    if conf_id >= 0:
                        logger.info(f"Generated basic 3D conformer for {cc.resname}")
                        cc.rdkit_mol = mol
                        cc.smiles_exh, cc.atom_name = self._get_smiles_with_atom_names(mol)
                    else:
                        logger.error("All conformer generation methods failed")
                except Exception as e3:
                    logger.error(f"Even basic embedding failed: {e3}")
                    import traceback
                    logger.error(f"Basic embedding traceback: {traceback.format_exc()}")
        
        # After conformer generation, reassign atom names and check again
        self.reassign_atom_names(cc.rdkit_mol)
        self.check_atom_ids(cc.rdkit_mol, context='after conformer gen')
        return cc
    
    def convert_to_parsed_residue(self, cc: ChemicalComponent) -> ParsedResidue:
        """
        Convert ChemicalComponent to Boltz ParsedResidue format with geometry constraints.
        
        Args:
            cc: ChemicalComponent object
            
        Returns:
            ParsedResidue object with geometry constraints
        """
        
        mol = cc.rdkit_mol
        
        # Get residue type from token_ids
        resname = cc.resname
        if resname in token_ids:
            residue_type = token_ids[resname]
        else:
            # Default to unknown token
            residue_type = token_ids.get("UNK", 0)
        
        # Use constraint-aware parsing to get geometry constraints
        try:
            # Ensure atoms have 'name' property for parse_ccd_residue
            for atom in mol.GetAtoms():
                if atom.HasProp('atom_id') and not atom.HasProp('name'):
                    atom.SetProp('name', atom.GetProp('atom_id'))
            
            # Ensure ring information is initialized for constraint computation
            Chem.GetSymmSSSR(mol)
            
            parsed_residue_with_constraints = parse_ccd_residue(resname, mol, 0)
            
            # Add geometry constraint properties to the molecule
            add_pickled_prop(mol, 'symmetries', mol.GetSubstructMatches(mol, uniquify=False))
            
            # Only add constraints if they exist
            if hasattr(parsed_residue_with_constraints, 'rdkit_bounds_constraints') and parsed_residue_with_constraints.rdkit_bounds_constraints:
                add_pickled_prop(mol, 'pb_edge_index', extract_constraints(parsed_residue_with_constraints.rdkit_bounds_constraints, 'atom_idxs', transpose=True))
                add_pickled_prop(mol, 'pb_lower_bounds', extract_constraints(parsed_residue_with_constraints.rdkit_bounds_constraints, 'lower_bound'))
                add_pickled_prop(mol, 'pb_upper_bounds', extract_constraints(parsed_residue_with_constraints.rdkit_bounds_constraints, 'upper_bound'))
                add_pickled_prop(mol, 'pb_bond_mask', extract_constraints(parsed_residue_with_constraints.rdkit_bounds_constraints, 'is_bond'))
                add_pickled_prop(mol, 'pb_angle_mask', extract_constraints(parsed_residue_with_constraints.rdkit_bounds_constraints, 'is_angle'))
            
            if hasattr(parsed_residue_with_constraints, 'chiral_atom_constraints') and parsed_residue_with_constraints.chiral_atom_constraints:
                add_pickled_prop(mol, 'chiral_atom_index', extract_constraints(parsed_residue_with_constraints.chiral_atom_constraints, 'atom_idxs', transpose=True))
                add_pickled_prop(mol, 'chiral_atom_is_r', extract_constraints(parsed_residue_with_constraints.chiral_atom_constraints, 'is_r'))
                add_pickled_prop(mol, 'chiral_atom_is_reference', extract_constraints(parsed_residue_with_constraints.chiral_atom_constraints, 'is_reference'))
            
            if hasattr(parsed_residue_with_constraints, 'stereo_bond_constraints') and parsed_residue_with_constraints.stereo_bond_constraints:
                add_pickled_prop(mol, 'stereo_bond_index', extract_constraints(parsed_residue_with_constraints.stereo_bond_constraints, 'atom_idxs', transpose=True))
                add_pickled_prop(mol, 'stereo_bond_is_e', extract_constraints(parsed_residue_with_constraints.stereo_bond_constraints, 'is_e'))
                add_pickled_prop(mol, 'stereo_bond_is_reference', extract_constraints(parsed_residue_with_constraints.stereo_bond_constraints, 'is_reference'))
            
            if hasattr(parsed_residue_with_constraints, 'planar_ring_5_constraints') and parsed_residue_with_constraints.planar_ring_5_constraints:
                add_pickled_prop(mol, 'aromatic_5_ring_index', extract_constraints(parsed_residue_with_constraints.planar_ring_5_constraints, 'atom_idxs', transpose=True))
            
            if hasattr(parsed_residue_with_constraints, 'planar_ring_6_constraints') and parsed_residue_with_constraints.planar_ring_6_constraints:
                add_pickled_prop(mol, 'aromatic_6_ring_index', extract_constraints(parsed_residue_with_constraints.planar_ring_6_constraints, 'atom_idxs', transpose=True))
            
            if hasattr(parsed_residue_with_constraints, 'planar_bond_constraints') and parsed_residue_with_constraints.planar_bond_constraints:
                add_pickled_prop(mol, 'planar_double_bond_index', extract_constraints(parsed_residue_with_constraints.planar_bond_constraints, 'atom_idxs', transpose=True))
            
            logger.info(f"Successfully added geometry constraints for {resname}")
            
        except Exception as e:
            logger.warning(f"Failed to add geometry constraints: {e}")
            # Continue without geometry constraints
            add_pickled_prop(mol, 'symmetries', mol.GetSubstructMatches(mol, uniquify=False))
        
        # Create atoms list
        atoms = []
        try:
            conf = mol.GetConformer()
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                
                # Get atom name - try different property names
                atom_name = None
                for prop_name in ['atom_id', 'name', 'atom_id']:
                    try:
                        atom_name = atom.GetProp(prop_name)
                        break
                    except:
                        continue
                
                # If no property found, use atom symbol + index
                if atom_name is None:
                    atom_name = f"{atom.GetSymbol()}{i+1}"
                
                parsed_atom = ParsedAtom(
                    name=atom_name,
                    coords=(float(pos.x), float(pos.y), float(pos.z)),
                    is_present=True,
                    bfactor=0.0  # Default B-factor
                )
                atoms.append(parsed_atom)
        except Exception as e:
            logger.warning(f"Failed to get conformer coordinates: {e}")
            # Use default coordinates if conformer fails
            for i, atom in enumerate(mol.GetAtoms()):
                # Get atom name - try different property names
                atom_name = None
                for prop_name in ['atom_id', 'name', 'atom_id']:
                    try:
                        atom_name = atom.GetProp(prop_name)
                        break
                    except:
                        continue
                
                # If no property found, use atom symbol + index
                if atom_name is None:
                    atom_name = f"{atom.GetSymbol()}{i+1}"
                
                parsed_atom = ParsedAtom(
                    name=atom_name,
                    coords=(0.0, 0.0, 0.0),  # Default coordinates
                    is_present=True,
                    bfactor=0.0  # Default B-factor
                )
                atoms.append(parsed_atom)
        
        # Create bonds list
        bonds = []
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # Map bond types
            bond_type = bond.GetBondType()
            if bond_type == Chem.BondType.SINGLE:
                bond_type_id = 1
            elif bond_type == Chem.BondType.DOUBLE:
                bond_type_id = 2
            elif bond_type == Chem.BondType.TRIPLE:
                bond_type_id = 3
            elif bond_type == Chem.BondType.AROMATIC:
                bond_type_id = 4
            else:
                bond_type_id = 1  # Default to single bond
            
            parsed_bond = ParsedBond(
                atom_1=begin_idx,
                atom_2=end_idx,
                type=bond_type_id
            )
            bonds.append(parsed_bond)
        
        # Create ParsedResidue
        parsed_residue = ParsedResidue(
            name=resname,
            type=residue_type,
            idx=0,  # Will be set by caller
            atoms=atoms,
            bonds=bonds,
            orig_idx=None,
            atom_center=0,  # Default center atom index
            atom_disto=0,   # Default distance atom index
            is_standard=False,  # Custom residue
            is_present=True
        )
        
        return parsed_residue
    
    def save_to_pickle(self, parsed_residue: ParsedResidue, mol_with_constraints, output_path: str):
        """
        Save ParsedResidue and molecule with geometry constraints to pickle file.
        
        Args:
            parsed_residue: ParsedResidue object
            mol_with_constraints: RDKit molecule with geometry constraint properties
            output_path: Output file path
        """
        
        # Set RDKit to pickle all properties
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        
        with open(output_path, 'wb') as f:
            pickle.dump(mol_with_constraints, f)
        logger.info(f"Saved molecule with geometry constraints to {output_path}")
    
    def save_to_sdf(self, mol, output_path: str):
        """
        Save molecule to SDF file for visualization and checking.
        
        Args:
            mol: RDKit molecule
            output_path: Output SDF file path
        """
        try:
            Chem.MolToMolFile(mol, output_path)
            logger.info(f"Saved molecule to SDF file: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save SDF file: {e}")
    
    def process_custom_residue(self, smiles: str, resname: str, output_path: str,
                             backbone_type: Optional[str] = None,
                             custom_leaving_atoms: Optional[Set[str]] = None,
                             custom_leaving_pattern: Optional[Dict[str, Set[int]]] = None,
                             generate_3d: bool = True,
                             save_sdf: bool = False) -> ParsedResidue:
        """
        Complete pipeline to process custom residue from SMILES to Boltz format.
        
        Args:
            smiles: SMILES string
            resname: Residue name (3-letter code)
            output_path: Output pickle file path
            backbone_type: Backbone type ("protein", "nucleic_acid", "custom")
            custom_leaving_atoms: Custom leaving atom names
            custom_leaving_pattern: Custom leaving atom patterns
            generate_3d: Whether to generate 3D conformer
            save_sdf: Whether to save the processed molecule as an SDF file
            
        Returns:
            ParsedResidue object
        """
        logger.info(f"Processing custom residue {resname} from SMILES: {smiles}")
        
        # Step 1: Create molecule from SMILES
        cc = self.create_molecule_from_smiles(smiles, resname)
        logger.info(f"Created molecule with {cc.rdkit_mol.GetNumAtoms()} atoms")
        
        # Step 2: Identify backbone type if not specified
        if backbone_type is None:
            backbone_type = self.identify_backbone_type(cc)
            logger.info(f"Identified backbone type: {backbone_type}")
        
        # Step 3: Rename backbone atoms
        cc = self.rename_backbone_atoms(cc, backbone_type)
        logger.info(f"Renamed backbone atoms")
        
        # Step 4: Define leaving atoms
        cc = self.define_leaving_atoms(cc, backbone_type, custom_leaving_atoms, custom_leaving_pattern)
        logger.info(f"Defined leaving atoms")
        
        # Step 5: Truncate residue (remove leaving atoms)
        cc = self.truncate_residue(cc, backbone_type, custom_leaving_atoms, custom_leaving_pattern)
        logger.info(f"Truncated residue")
        
        # Step 6: Generate 3D conformer if requested
        if generate_3d:
            cc = self.generate_3d_conformer(cc)
        
        # Step 7: Convert to ParsedResidue with geometry constraints
        parsed_residue = self.convert_to_parsed_residue(cc)
        logger.info(f"Converted to ParsedResidue format with geometry constraints")
        
        # Print the final SMILES string
        print(f"Final SMILES after processing: {cc.smiles_exh}")
        print(f"Number of atoms in final molecule: {cc.rdkit_mol.GetNumAtoms()}")
        
        # Step 8: Save to pickle
        self.save_to_pickle(parsed_residue, cc.rdkit_mol, output_path)
        
        # Step 9: Save to SDF if requested
        if save_sdf:
            self.save_to_sdf(cc.rdkit_mol, output_path.replace('.pkl', '.sdf'))
        
        return parsed_residue


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Process custom residue from SMILES to Boltz format")
    parser.add_argument("--smiles", required=True, help="SMILES string of the molecule")
    parser.add_argument("--name", required=True, help="Residue name (3-letter code)")
    parser.add_argument("--output", required=True, help="Output pickle file path")
    parser.add_argument("--backbone-type", choices=["protein", "nucleic_acid", "custom"], 
                       help="Backbone type (auto-detected if not specified)")
    parser.add_argument("--leaving-atoms", nargs="+", help="Custom leaving atom names")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D conformer generation")
    parser.add_argument("--save-sdf", action="store_true", help="Save processed molecule as SDF file")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CustomResidueProcessor()
    
    # Process the residue
    try:
        parsed_residue = processor.process_custom_residue(
            smiles=args.smiles,
            resname=args.name,
            output_path=args.output,
            backbone_type=args.backbone_type,
            custom_leaving_atoms=set(args.leaving_atoms) if args.leaving_atoms else None,
            generate_3d=not args.no_3d,
            save_sdf=args.save_sdf
        )
        
        print(f"Successfully processed {args.name} residue!")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to process residue: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 