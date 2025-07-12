#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
from boltz.data.parse.schema import parse_ccd_residue as schema_parse_ccd_residue
from boltz.data.parse.mmcif_with_constraints import parse_ccd_residue as mmcif_with_constraints_parse_ccd_residue
import pickle
import numpy as np
import argparse
from CifFile import ReadCif
import yaml
import copy
import itertools
import operator

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

def setPickledProp(mol, name, value):
    mol.SetProp(
        name,
        pickle.dumps(value).hex()
    )

def getPickledProp(mol, name):
    """Get a pickled property from an RDKit molecule."""
    return pickle.loads(
        bytes.fromhex(mol.GetProp(name))
    )

def backbone_from_smarts(smarts: str, atom_names: list[str | None], atom_idxs: list[int | None] | None, leaving_atoms: set[str]) -> Chem.Mol:
    mol = Chem.MolFromSmarts(smarts)

    if mol.GetNumAtoms() != len(atom_names):
        raise Exception("Backbone atom names do not match smarts!")
    
    for atom_name, atom_idx, atom in zip(atom_names, atom_idxs or itertools.repeat(None), mol.GetAtoms()):
        if atom_name:
            atom.SetProp("name", atom_name)
            residueInfo = atom.GetPDBResidueInfo() or Chem.AtomPDBResidueInfo()
            residueInfo.SetName(atom_name)
            atom.SetPDBResidueInfo(residueInfo)
            if atom_idx != None:
                atom.SetIntProp("idx", atom_idx)
        
        atom.SetIntProp("leaving_atom", int(atom_name in leaving_atoms))

    return mol

default_backbones = {
    "protein": backbone_from_smarts(
        "[NX3][CX4][CX3](=O)[O]",
        ["N", "CA", "C", "O", "OXT"],
        [0, 1, 2, 3, -1],
        set(["OXT"])
        
    ),
    "RNA": backbone_from_smarts(
        "[O][PX4](=O)([O])[OX2][CX4][CX4]([OX2]1)[CX4]([OX2])[CX4]([OX2])[CX4]1",
        ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],
        None,
        set(["OP3"])
    ),
    "DNA": backbone_from_smarts(
        "[O][PX4](=O)([O])[OX2][CX4][CX4]([OX2]1)[CX4]([OX2])[CX4][CX4]1",
        ["OP3", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"],
        None,
        set(["OP3"])
    ),
}

def getBackbone(mol: Chem.Mol, backbones: list[Chem.Mol]):
    for backbone in backbones.values():
        indicies = mol.GetSubstructMatch(backbone)
        if indicies:
            return backbone, indicies

    return None, None

def addBackboneProps(mol: Chem.Mol, backbones: list[Chem.Mol], props: set[str] = set(["name", "leaving_atom", "idx"])):
    backbone, indices = getBackbone(mol, backbones)

    if not backbone:
        return
    
    backbone_atom: Chem.Atom
    for atom_idx, backbone_atom in zip(indices, backbone.GetAtoms()):
        mol_atom = mol.GetAtomWithIdx(atom_idx)

        if backbone_atom.HasProp("name"):
            residueInfo = mol_atom.GetPDBResidueInfo() or Chem.AtomPDBResidueInfo()
            residueInfo.SetName(backbone_atom.GetProp("name"))
            mol_atom.SetPDBResidueInfo(residueInfo)
    
        for prop in props:
            if not backbone_atom.HasProp(prop):
                continue

            value = backbone_atom.GetProp(prop)
            mol_atom.SetProp(prop, value)
            
def renumberAtoms(mol: Chem.Mol) -> Chem.Mol:
    start_atoms = {}
    other_atoms = []
    end_atoms = {}

    for atom in mol.GetAtoms():
        if not atom.HasProp("idx"):
            other_atoms.append(atom.GetIdx())
            continue

        idx = atom.GetIntProp("idx")
        if idx >= 0:
            start_atoms[idx] = atom.GetIdx()
        else:
            end_atoms[idx] = atom.GetIdx()
    
    new_atom_order = list(itertools.chain(
        map(operator.itemgetter(1), sorted(start_atoms.items())),
        other_atoms,
        map(operator.itemgetter(1), sorted(end_atoms.items()))
    ))

    return Chem.RenumberAtoms(mol, new_atom_order)

def addBoltzParams(resname: str, mol: Chem.Mol, parse_ccd_residue):
    # Add missing atom names
    atom: Chem.Atom
    for atom in mol.GetAtoms():
        atom_name = f"{atom.GetSymbol()}{atom.GetIdx() + 1}"
        if atom.HasProp("name"):
            atom_name = atom.GetProp("name")
        else:
            atom.SetProp("name", atom_name)
        residueInfo = atom.GetPDBResidueInfo() or Chem.AtomPDBResidueInfo()
        residueInfo.SetName(atom_name)
        atom.SetPDBResidueInfo(residueInfo)

    parsedResidue = parse_ccd_residue(
        name=resname,
        ref_mol=mol,
        res_idx=0,
    )

    setPickledProp(mol, 'symmetries', mol.GetSubstructMatches(mol, uniquify=False))
    setPickledProp(mol, 'pb_edge_index', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'atom_idxs', transpose=True))
    setPickledProp(mol, 'pb_lower_bounds', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'lower_bound'))
    setPickledProp(mol, 'pb_upper_bounds', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'upper_bound'))
    setPickledProp(mol, 'pb_bond_mask', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'is_bond'))
    setPickledProp(mol, 'pb_angle_mask', extractConstraints(parsedResidue.rdkit_bounds_constraints, 'is_angle')) # FIXME: This doesn't match the ref
    setPickledProp(mol, 'chiral_atom_index', extractConstraints(parsedResidue.chiral_atom_constraints, 'atom_idxs', transpose=True))
    setPickledProp(mol, 'chiral_check_mask', extractConstraints(parsedResidue.chiral_atom_constraints, 'is_reference'))
    setPickledProp(mol, 'chiral_atom_orientations', extractConstraints(parsedResidue.chiral_atom_constraints, 'is_r'))
    setPickledProp(mol, 'stereo_bond_index', extractConstraints(parsedResidue.stereo_bond_constraints, 'atom_idxs', transpose=True))
    setPickledProp(mol, 'stereo_check_mask', extractConstraints(parsedResidue.stereo_bond_constraints, 'is_check'))
    setPickledProp(mol, 'stereo_bond_orientations', extractConstraints(parsedResidue.stereo_bond_constraints, 'is_e'))
    setPickledProp(mol, 'aromatic_5_ring_index', extractConstraints(parsedResidue.planar_ring_5_constraints, 'atom_idxs', transpose=True))
    setPickledProp(mol, 'aromatic_6_ring_index', extractConstraints(parsedResidue.planar_ring_6_constraints, 'atom_idxs', transpose=True))
    setPickledProp(mol, 'planar_double_bond_index', extractConstraints(parsedResidue.planar_bond_constraints, 'atom_idxs', transpose=True))

def orderToBondType(order: str):
    bondType = {
        'SING': Chem.BondType.SINGLE,
        'DOUB': Chem.BondType.DOUBLE,
        'TRIP': Chem.BondType.TRIPLE,
    }
    return bondType[order]

def boltzMolFromCIF(resname: str, filename: str | Path) -> Chem.Mol:
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
    
    rwMol = Chem.RWMol()

    # Add Atoms
    atomMap = {}
    for atom_type, atom_name, atom_charge, atom_leaving_flag in zip(
        atom_types, atom_names, atom_charges, atom_leaving_flags
    ):
        atom = Chem.Atom(atom_type.capitalize() if atom_type != 'X' else '*')
        atom.SetProp('name', atom_name)
        atom.SetProp('leaving_atom', atom_leaving_flag)
        atom.SetFormalCharge(atom_charge)
        residue_info = Chem.AtomPDBResidueInfo()
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
        conformer = Chem.Conformer(len(atom_positions))
        conformer.SetPositions(np.array(atom_positions))
        rwMol.AddConformer(conformer)

    mol = Chem.RemoveHs(rwMol.GetMol())
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    addBoltzParams(resname, mol, mmcif_with_constraints_parse_ccd_residue)
    return mol

def boltzMolFromSmiles(resname: str, smiles: str, backbones: list[Chem.Mol]) -> Chem.Mol:
    mol = AllChem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)

    options = AllChem.ETKDGv3()
    options.clearConfs = False

    conf_id = AllChem.EmbedMolecule(mol, options)

    if conf_id == -1:
        options.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, options)

    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    addBackboneProps(mol, backbones)
    mol = AllChem.RemoveHs(mol, sanitize=False)
    mol = renumberAtoms(mol)
    addBoltzParams(resname, mol, schema_parse_ccd_residue)
    return mol

def boltzMolFromPDB(resname: str, filename: str | Path, backbones: list[Chem.Mol]) -> Chem.Mol:
    mol = Chem.MolFromPDBFile(filename)

    for atom in mol.GetAtoms():
        residueInfo = atom.GetPDBResidueInfo()
        atom.SetProp("name", residueInfo.GetName())
        atom.SetIntProp("leaving_atom", 0)

    mol = AllChem.AddHs(mol)

    options = AllChem.ETKDGv3()
    options.clearConfs = False

    conf_id = AllChem.EmbedMolecule(mol, options)

    if conf_id == -1:
        options.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, options)

    AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    addBackboneProps(mol, backbones)
    mol = AllChem.RemoveHs(mol, sanitize=False)
    mol = renumberAtoms(mol)
    addBoltzParams(resname, mol, schema_parse_ccd_residue)
    return mol


def saveBoltzMol(mol: Chem.Mol, filename: str | Path):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    with open(filename, "wb") as f:
        pickle.dump(mol, f)

def printBoltzMolProps(mol):
    props = [
        'symmetries',
        'pb_edge_index',
        'pb_lower_bounds',
        'pb_upper_bounds',
        'pb_bond_mask',
        'pb_angle_mask',
        'chiral_atom_index',
        'chiral_check_mask',
        'chiral_atom_orientations',
        'stereo_bond_index',
        'stereo_check_mask',
        'stereo_bond_orientations',
        'aromatic_5_ring_index',
        'aromatic_6_ring_index',
        'planar_double_bond_index',
    ]
    for prop in props:
        if not mol.HasProp(prop):
            continue
        print(f"{prop}: {getPickledProp(mol, prop)}")

def validate_name(name):
    if len(name) < 1:
        raise argparse.ArgumentTypeError("name may not be empty")
    if len(name) > 5:
        raise argparse.ArgumentTypeError("name must not exceed 5 characters")
    return name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, type=validate_name, help="Name of molecule can not exceed 5 characters")
    parser.add_argument("-i", "--input", help="CIF or PDB file containing molecule to add to boltz's CCD")
    parser.add_argument("-s", "--smiles", help="SMILES to add to boltz's CCD")
    parser.add_argument("--backbones_filter", action="extend", nargs="+", help="Explicitly include a subset of backbones by name")
    parser.add_argument("--backbones", help="YAML specifying backbones smarts, atom_names and leaving_atoms")
    parser.add_argument("--save_pdb", help="Save pdb to path specified")
    parser.add_argument("-b", "--boltz_path", default=Path().home() / '.boltz', help="Path to boltz directory")
    args = parser.parse_args()

    backbones = copy.copy(default_backbones)
    if args.backbones:
        with open(args.backbones) as f:
            backbone_templates = yaml.safe_load(f)
        for backbone_name, backbone_template in backbone_templates.items():
            backbones[backbone_name] = backbone_from_smarts(
                backbone_template["smarts"],
                backbone_template["atom_names"],
                backbone_template["atom_idxs"],
                backbone_template["leaving_atoms"],
            )

    if args.backbones_filter:
        backbones = {backbone_name: backbones[backbone_name] for backbone_name in args.backbones_filter if backbone_name in backbones}

    mol = None
    if args.input:
        input_suffix = Path(args.input).suffix.lower()

        match input_suffix:
            case ".cif":
                mol = boltzMolFromCIF(args.name, args.input)
            case ".pdb":
                mol = boltzMolFromPDB(args.name, args.input, backbones)
            case _:
                raise Exception("Input has an unsupported file extension!")
    elif args.smiles:
        mol = boltzMolFromSmiles(args.name, args.smiles, backbones)
    else:
        raise Exception("No input or smiles provided!")

    if args.save_pdb:
        Chem.MolToPDBFile(mol, str(Path(args.save_pdb) / f"{args.name}.pdb"))

    saveBoltzMol(mol, Path(args.boltz_path) / "mols" / f"{args.name}.pkl")

if __name__ == "__main__":
    main()