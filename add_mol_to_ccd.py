#!/usr/bin/env python

import argparse
from pathlib import Path
import pickle
import urllib.request
import warnings
import rdkit.Chem
from CifFile import ReadCif
import numpy as np
from boltz.data.parse.mmcif_with_constraints import parse_ccd_residue

def orderToBondType(order: str):
    # Accept common CIF value_order variants like 'SING', 'SINGLE', 'DOUB', 'DOUBLE', 'TRIP', 'TRIPLE'
    if order is None:
        return rdkit.Chem.BondType.SINGLE
    s = str(order).strip().upper()
    if s.startswith('SING') or s == 'SINGLE':
        return rdkit.Chem.BondType.SINGLE
    if s.startswith('DOUB') or s == 'DOUBLE':
        return rdkit.Chem.BondType.DOUBLE
    if s.startswith('TRIP') or s == 'TRIPLE':
        return rdkit.Chem.BondType.TRIPLE
    # Aromatic fix thanks to ctakemoto
    if s.startswith('AROM'):
        return rdkit.Chem.BondType.AROMATIC
    # Unknown/unsupported order: warn and fall back to SINGLE
    warnings.warn(f"Unrecognized bond order '{order}' in CIF — falling back to SINGLE")
    return rdkit.Chem.BondType.SINGLE

def MolFromCIFFile(filename, resname):
    cif = ReadCif(filename)

    if len(cif) < 1:
        raise Exception("CIF does not contain any components!")
        # Prefer a block whose name matches the residue name (e.g. data_comp_NAME)
    cif_component = None
    try:
        keys = list(cif.keys())
    except Exception:
        keys = []

    resname_l = resname.lower() if resname is not None else None
    for k in keys:
        if resname_l and resname_l in k.lower():
            cif_component = cif[k]
            break

    # If no matching block found, pick the first block that isn't a comp_list
    if cif_component is None:
        for k in keys:
            if 'comp_list' not in k.lower():
                cif_component = cif[k]
                break

    # Final fallback
    if cif_component is None:
        cif_component = cif.first_block()

    # Get Atoms
    atom_types = cif_component['_chem_comp_atom.type_symbol']
    atom_names = cif_component['_chem_comp_atom.atom_id']
    atom_charges = list(map(int, cif_component['_chem_comp_atom.charge']))
    atom_leaving_flags = list(map(lambda flag: '1' if flag == 'Y' else '0', cif_component.get('_chem_comp_atom.pdbx_leaving_atom_flag', ['N'] * len(atom_names))))
    atom_positions = list(zip(
        map(float, cif_component['_chem_comp_atom.pdbx_model_Cartn_x_ideal']),
        map(float, cif_component['_chem_comp_atom.pdbx_model_Cartn_y_ideal']),
        map(float, cif_component['_chem_comp_atom.pdbx_model_Cartn_z_ideal']),
    ))

    if len(atom_positions) == 0:
        raise Exception("CIF must provide conformation in the form of atom positions!")

    # Get Bonds
    bond_begin_atoms = cif_component.get('_chem_comp_bond.atom_id_1', [])
    bond_end_atoms = cif_component.get('_chem_comp_bond.atom_id_2', [])
    bond_types = list(map(orderToBondType, cif_component.get('_chem_comp_bond.value_order', [])))
    bond_aromatic_flags = list(map(lambda flag: flag == 'Y', cif_component.get('_chem_comp_bond.pdbx_aromatic_flag', [])))
    
    rwMol = rdkit.Chem.RWMol()

    # Add Atoms
    atomMap = {}
    for atom_type, atom_name, atom_charge, atom_leaving_flag in zip(
        atom_types, atom_names, atom_charges, atom_leaving_flags
    ):
        atom = rdkit.Chem.Atom(atom_type.capitalize() if atom_type != 'X' else '*')
        atom.SetProp('name', atom_name)
        atom.SetProp('leaving_atom', atom_leaving_flag)
        atom.SetFormalCharge(atom_charge)
        residue_info = rdkit.Chem.AtomPDBResidueInfo()
        residue_info.SetName(atom_name.center(4))
        residue_info.SetIsHeteroAtom(True)
        residue_info.SetResidueName(resname)
        residue_info.SetResidueNumber(1)
        atom.SetPDBResidueInfo(residue_info)
        atom_index = rwMol.AddAtom(atom)
        atomMap[atom_name] = atom_index
    
    # Add bonds
    for bond_begin_atom, bond_end_atom, bond_type, bond_aromatic_flag in zip(
        bond_begin_atoms, bond_end_atoms, bond_types, bond_aromatic_flags
    ):
        number_of_bonds = rwMol.AddBond(
            atomMap[bond_begin_atom],
            atomMap[bond_end_atom],
            bond_type
        )
        rwMol.GetBondWithIdx(number_of_bonds - 1).SetIsAromatic(bond_aromatic_flag)

    # Add conformation
    if len(atom_positions) > 0:
        conformer = rdkit.Chem.Conformer(len(atom_positions))
        conformer.SetPositions(np.array(atom_positions))
        rwMol.AddConformer(conformer)

    return rwMol.GetMol()

def extractConstraints(constraints, key, transpose=False):
    result = np.array(list(
        map(
            lambda obj: getattr(obj, key),
            constraints
        )
    ))

    if transpose:
        return result.transpose()
    
    return result

def addPickledProp(mol, name, value):
    mol.SetProp(
        name,
        pickle.dumps(value).hex()
    )

def setPropsFromPDBResidueInfo(mol):
    for atom in mol.GetAtoms():
        residueInfo = atom.GetPDBResidueInfo()
        atom.SetProp("name", residueInfo.GetName().strip().upper())
        atom.SetIntProp("leaving_atom", 0)

def addMolToCCD(resname: str, mol: rdkit.Chem.Mol, boltz_path = Path().home() / '.boltz'):
    if mol is None:
        raise Exception("Mol can not be null")

    mol = rdkit.Chem.RemoveHs(mol)
    mol.UpdatePropertyCache(strict=False)
    rdkit.Chem.SanitizeMol(mol)
    rdkit.Chem.rdmolops.AssignStereochemistryFrom3D(mol)

    parsedResidue = parse_ccd_residue(resname, mol, 0)

    addPickledProp(mol, 'symmetries', mol.GetSubstructMatches(mol, uniquify=False))
    addPickledProp(mol, 'pb_edge_index', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'atom_idxs', transpose=True))
    addPickledProp(mol, 'pb_lower_bounds', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'lower_bound'))
    addPickledProp(mol, 'pb_upper_bounds', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'upper_bound'))
    addPickledProp(mol, 'pb_bond_mask', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'is_bond'))
    addPickledProp(mol, 'pb_angle_mask', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'is_angle')) # FIXME: This doesn't match the ref
    addPickledProp(mol, 'chiral_atom_index', extractConstraints(parsedResidue.chiral_atom_constraints, 'atom_idxs', transpose=True))
    addPickledProp(mol, 'chiral_check_mask', extractConstraints(parsedResidue.chiral_atom_constraints, 'is_reference'))
    addPickledProp(mol, 'chiral_atom_orientations', extractConstraints(parsedResidue.chiral_atom_constraints, 'is_r'))
    addPickledProp(mol, 'stereo_bond_index', extractConstraints(parsedResidue.stereo_bond_constraints, 'atom_idxs', transpose=True))
    addPickledProp(mol, 'stereo_check_mask', extractConstraints(parsedResidue.stereo_bond_constraints, 'is_check'))
    addPickledProp(mol, 'stereo_bond_orientations', extractConstraints(parsedResidue.stereo_bond_constraints, 'is_e'))
    addPickledProp(mol, 'aromatic_5_ring_index', extractConstraints(parsedResidue.planar_ring_5_constraints, 'atom_idxs', transpose=True))
    addPickledProp(mol, 'aromatic_6_ring_index', extractConstraints(parsedResidue.planar_ring_6_constraints, 'atom_idxs', transpose=True))
    addPickledProp(mol, 'planar_double_bond_index', extractConstraints(parsedResidue.planar_bond_constraints, 'atom_idxs', transpose=True))

    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
    with open(Path(boltz_path) / 'mols' / f'{resname}.pkl', 'wb') as f:
        pickle.dump(mol, f)

def addCIFToCCD(resname: str, filename = None, boltz_path = Path().home() / '.boltz'):
    if filename is None:
        try:
            filename, _ = urllib.request.urlretrieve(f"https://files.rcsb.org/ligands/download/{resname}.cif")
            return filename
        except:
            raise Exception(f"Can not retrieve {resname} from RCSB!")

    filepath = Path(filename)
    suffix = filepath.suffix.lower()
    if suffix == ".pdb":
        mol = rdkit.Chem.MolFromPDBFile(filename)
        setPropsFromPDBResidueInfo(mol)
    elif suffix == ".cif":
        mol = MolFromCIFFile(filename, resname)
    else:
        raise Exception(f"Unknown file extension {suffix}")
    
    if mol is None:
        raise Exception("Unknown error: Failed to load mol")
    
    addMolToCCD(resname, mol, boltz_path)

def validate_name(name):
    if len(name) < 1:
        raise argparse.ArgumentTypeError("name may not be empty")
    if len(name) > 5:
        raise argparse.ArgumentTypeError("name must not exceed 5 characters")
    return name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, type=validate_name, help="Name of molecule can not exceed 5 characters")
    parser.add_argument("-i", "--input", help="CIF file containing molecule to add to boltz's CCD")
    parser.add_argument("-b", "--boltz_path", default=Path().home() / '.boltz', help="Path to boltz directory")
    args = parser.parse_args()
    addCIFToCCD(args.name, args.input, args.boltz_path)

if __name__ == '__main__':
    main()
