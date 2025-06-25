#!/usr/bin/env python

import argparse
from pathlib import Path
import pickle
import meeko.chemtempgen
import rdkit.Chem
from CifFile import ReadCif
import numpy as np
import meeko

def orderToBondType(order: str):
    bondType = {
        'SING': rdkit.Chem.BondType.SINGLE,
        'DOUB': rdkit.Chem.BondType.DOUBLE,
        'TRIP': rdkit.Chem.BondType.TRIPLE,
    }
    return bondType[order]

def addCIFToCCD(resname, filename = None, boltz_path = Path().home() / '.boltz'):
    if filename == None:
        filename = meeko.chemtempgen.fetch_from_pdb(resname)
    cif = ReadCif(filename)

    if len(cif) < 1:
        raise Exception("CIF does not contain any components!")
    if len(cif) > 1:
        raise Exception("CIF may only contain ONE component!")
    
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

    rwMol.UpdatePropertyCache(strict=False)
    rdkit.Chem.SanitizeMol(rwMol)

    rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
    with open(Path(boltz_path) / 'mols' / f'{resname}.pkl', 'wb') as f:
        pickle.dump(rwMol.GetMol(), f)

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