#!/usr/bin/env python

import numpy as np
import pickle
from pathlib import Path
import json
import yaml
import rdkit.Chem
import pandas as pd
import argparse
import boltz.data.types
import torch
from boltz.data.module.inferencev2 import PredictionDataset
from boltz.data.types import Manifest, RDKitBoundsConstraint, ChiralAtomConstraint, StereoBondConstraint, PlanarBondConstraint, PlanarRing5Constraint, PlanarRing6Constraint, TokenV2, TokenBondV2
import funcy

def get_paths(project: Path, file: Path, output: Path, suffix: str = None):
    output_path = output / file.relative_to(project).with_suffix(suffix or "")
    if suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    return file, output_path

def dump_json(project: Path, file: Path, output: Path):
    input_path, output_path = get_paths(project, file, output, suffix=".yaml")
    print(f"Generating: {output_path}")
    with open(input_path) as f:
        data = json.load(f)
    
    with open(output_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def dump_np_dict(np_dict: np.array, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    for key, value in np_dict.items():
        if type(value) not in [np.ndarray, torch.Tensor]:
            print(f"Skipping: {key} {type(value)}")
            continue
        
        output_file = output_path / f"{key}.csv"
        print(f"Generating: {output_file}")

        try:
            df = None
            if hasattr(value.dtype, "names") and value.dtype.names:
                df = pd.DataFrame.from_records(value.tolist(), columns=value.dtype.names)
            else:
                df = pd.DataFrame(value.tolist())
            df.to_csv(output_file)
        except e:
            print(f"Failed to generate: \"{output_file}\"")
            raise e

def getPickledProp(mol, name):
    """Get a pickled property from an RDKit molecule."""
    return pickle.loads(
        bytes.fromhex(mol.GetProp(name))
    )

def dump_pkl(project: Path, file: Path, output: Path):
    input_path, output_path = get_paths(project, file, output)
    with open(input_path, "rb") as f:
        mols = pickle.load(f)
    if type(mols) is rdkit.Chem.Mol:
        mols = {file.stem: mols}
    for mol_name, mol in mols.items():
        output_file = output_path / f"{mol_name}.pdb"
        print(f"Generating: {output_file}")

        atom: rdkit.Chem.Atom
        for atom in mol.GetAtoms():
            residueInfo = atom.GetPDBResidueInfo() or rdkit.Chem.AtomPDBResidueInfo()
            residueInfo.SetName(atom.GetProp("name"))
            atom.SetPDBResidueInfo(residueInfo)
        
        rdkit.Chem.MolToPDBFile(mol, str(output_file))

        constraints = {}
        if mol.HasProp("pb_edge_index"):
            pb_edge_index = getPickledProp(mol, "pb_edge_index").T
            pb_lower_bounds = getPickledProp(mol, "pb_lower_bounds")
            pb_upper_bounds = getPickledProp(mol, "pb_upper_bounds")
            pb_bond_mask = getPickledProp(mol, "pb_bond_mask")
            pb_angle_mask = getPickledProp(mol, "pb_angle_mask")
            constraints["rdkit_bounds_constraints"] = np.array(
                list(zip(
                    pb_edge_index,
                    pb_bond_mask,
                    pb_angle_mask,
                    pb_upper_bounds,
                    pb_lower_bounds
                )),
                dtype=boltz.data.types.RDKitBoundsConstraint
            )
        
        if mol.HasProp("chiral_atom_index"):
            chiral_atom_index = getPickledProp(mol, 'chiral_atom_index').T
            chiral_check_mask = getPickledProp(mol, 'chiral_check_mask')
            chiral_atom_orientations = getPickledProp(mol, 'chiral_atom_orientations')
            constraints["chiral_atom_constraints"] = np.array(
                list(zip(
                    chiral_atom_index,
                    chiral_check_mask,
                    chiral_atom_orientations,
                )),
                dtype=boltz.data.types.ChiralAtomConstraint
            )
        
        if mol.HasProp("stereo_bond_index"):
            stereo_bond_index = getPickledProp(mol, 'stereo_bond_index').T
            stereo_check_mask = getPickledProp(mol, 'stereo_check_mask')
            stereo_bond_orientations = getPickledProp(mol, 'stereo_bond_orientations')
            constraints["stereo_bond_constraints"] = np.array(
                list(zip(
                    stereo_bond_index,
                    stereo_check_mask,
                    stereo_bond_orientations,
                )),
                dtype=boltz.data.types.StereoBondConstraint
            )
        
        if mol.HasProp("aromatic_5_ring_index"):
            aromatic_5_ring_index = getPickledProp(mol, 'aromatic_5_ring_index')
            constraints["planar_ring_5_constraints"] = np.array(
                [(index,) for index in aromatic_5_ring_index.T],
                dtype=boltz.data.types.PlanarRing5Constraint
            )
        
        if mol.HasProp("aromatic_6_ring_index"):
            aromatic_6_ring_index = getPickledProp(mol, 'aromatic_6_ring_index')
            constraints["planar_ring_6_constraints"] = np.array(
                [(index,) for index in aromatic_6_ring_index.T],
                dtype=boltz.data.types.PlanarRing6Constraint
            )
        
        if mol.HasProp("planar_double_bond_index"):
            planar_double_bond_index = getPickledProp(mol, 'planar_double_bond_index')
            constraints["planar_bond_constraints"] = np.array(
                [(index,) for index in planar_double_bond_index.T],
                dtype=boltz.data.types.PlanarBondConstraint
            )

        if constraints:
            dump_np_dict(
                constraints,
                output_path / f"{mol_name}_constraints"
            )
        
def dump_npz(project: Path, file: Path, output: Path):
    input_path, output_path = get_paths(project, file, output)
    dump_np_dict(
        np.load(input_path, allow_pickle=True),
        output_path
    )

def dump_features(project: Path, output: Path, boltz_path = Path().home() / '.boltz'):
    processed = project / "processed"
    dataset = PredictionDataset(
        Manifest.load(processed / "manifest.json"),
        processed / "structures",
        processed / "msa",
        Path(boltz_path) / "mols",
        processed / "constraints",
        processed / "templates",
        processed / "mols",
    )

    for features in dataset:
        output_path = output / processed.relative_to(project) / "features" / features["record"].id
        output_path.mkdir(parents=True, exist_ok=True)

        rdkit_bounds_constraints = np.array(
            list(zip(
                features["rdkit_bounds_index"].T,
                features["rdkit_bounds_bond_mask"],
                features["rdkit_bounds_angle_mask"],
                features["rdkit_upper_bounds"],
                features["rdkit_lower_bounds"],
            )),
            dtype=RDKitBoundsConstraint
        )

        chiral_atom_constraints = np.array(
            list(zip(
                features["chiral_atom_index"].T,
                features["chiral_reference_mask"],
                features["chiral_atom_orientations"],
            )),
            dtype=ChiralAtomConstraint
        )

        stereo_bond_constraints = np.array(
            list(zip(
                features["stereo_bond_index"].T,
                features["stereo_reference_mask"],
                features["stereo_bond_orientations"],
            )),
            dtype=StereoBondConstraint
        )

        planar_bond_constraints = np.array(
            [(index,) for index in features["planar_bond_index"].T],
            dtype=PlanarBondConstraint
        )

        planar_ring_5_constraints = np.array(
            [(index,) for index in features["planar_ring_5_index"].T],
            dtype=PlanarRing5Constraint
        )

        planar_ring_6_constraints = np.array(
            [(index,) for index in features["planar_ring_6_index"].T],
            dtype=PlanarRing6Constraint
        )

        token_data = np.array(
            list(zip(
                features["token_index"],
                # atom_idx
                # atom_num
                features["residue_index"],
                torch.argmax(features["res_type"], dim=1),
                # res_name
                features["sym_id"],
                features["asym_id"],
                features["entity_id"],
                features["mol_type"],
                torch.argmax(features["token_to_center_atom"], dim=1),
                torch.argmax(features["token_to_rep_atom"], dim=1),
                # center_coords
                features["disto_center"],
                features["token_resolved_mask"],
                features["token_disto_mask"],
                features["modified"],
                # frame_rot
                # frame_t
                # frame_mask
                features["cyclic_period"],
                features["affinity_token_mask"],
            )),
            dtype=[
                ("token_idx", np.dtype("i4")),
                # ("atom_idx", np.dtype("i4")),
                # ("atom_num", np.dtype("i4")),
                ("res_idx", np.dtype("i4")),
                ("res_type", np.dtype("i4")),
                # ("res_name", np.dtype("<U8")),
                ("sym_id", np.dtype("i4")),
                ("asym_id", np.dtype("i4")),
                ("entity_id", np.dtype("i4")),
                ("mol_type", np.dtype("i4")),
                ("center_idx", np.dtype("i4")),
                ("disto_idx", np.dtype("i4")),
                # ("center_coords", np.dtype("3f4")),
                ("disto_coords", np.dtype("3f4")),
                ("resolved_mask", np.dtype("?")),
                ("disto_mask", np.dtype("?")),
                ("modified", np.dtype("?")),
                # ("frame_rot", np.dtype("9f4")),
                # ("frame_t", np.dtype("3f4")),
                # ("frame_mask", np.dtype("i4")),
                ("cyclic_period", np.dtype("i4")),
                ("affinity_mask", np.dtype("?")),
            ]
        )

        features = {
            **funcy.omit(
                features,
                [
                    "rdkit_bounds_index",
                    "rdkit_bounds_bond_mask",
                    "rdkit_bounds_angle_mask",
                    "rdkit_upper_bounds",
                    "rdkit_lower_bounds",
                    "chiral_atom_index",
                    "chiral_reference_mask",
                    "chiral_atom_orientations",
                    "stereo_bond_index",
                    "stereo_reference_mask",
                    "stereo_bond_orientations",
                    "planar_bond_index",
                    "planar_ring_5_index",
                    "planar_ring_6_index",

                    "token_index",
                    "residue_index",
                    "res_type",
                    "sym_id",
                    "asym_id",
                    "entity_id",
                    "mol_type",
                    "token_to_center_atom",
                    "token_to_rep_atom",
                    "disto_center",
                    "token_resolved_mask",
                    "token_disto_mask",
                    "modified",
                    "cyclic_period",
                    "affinity_token_mask",
                ]
            ),
            "rdkit_bounds_constraints": rdkit_bounds_constraints,
            "chiral_atom_constraints": chiral_atom_constraints,
            "stereo_bond_constraints": stereo_bond_constraints,
            "planar_bond_constraints": planar_bond_constraints,
            "planar_ring_5_constraints": planar_ring_5_constraints,
            "planar_ring_6_constraints": planar_ring_6_constraints,
            "token_data": token_data,
        }

        dump_np_dict(
            features,
            output_path
        )

def dump_project(project: Path, output: Path, boltz_path = Path().home() / '.boltz'):
    for file in project.rglob("*.json"):
        dump_json(project, file, output)
    for file in project.rglob("*.pkl"):
        dump_pkl(project, file, output)
    for file in project.rglob("*.npz"):
        dump_npz(project, file, output)
    dump_features(project, output, boltz_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", help="Path to project to dump")
    parser.add_argument("--ccd", help="CCD ligand name to dump")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-b", "--boltz_path", default=Path().home() / '.boltz', help="Path to boltz directory")
    args = parser.parse_args()

    if not args.project and not args.ccd:
        parser.print_help()
        exit()

    boltz_path = Path(args.boltz_path)
    if args.project:
        project = Path(args.project)
        dump_project(project, Path(args.output or f"{project.stem}_dump"), boltz_path)
    
    mols_path = boltz_path / "mols"
    if args.ccd:
        dump_pkl(mols_path, mols_path / f"{args.ccd}.pkl", Path(args.output or f"{args.ccd}_dump"))

if __name__ == "__main__":
    main()