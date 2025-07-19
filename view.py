#!/usr/bin/env python3

import argparse
from pathlib import Path
import pymol
from pymol import cmd

def display_models(models: dict[str, Path], reference: str = None):
    hide_models = reference is not None

    if reference is None:
        reference = next(iter(models), None)

    for model_index, (model_name, model) in enumerate(models.items()):
        cmd.load(
            model,
            model_name
        )
        if hide_models or model_index > 0:
            cmd.align(model_name, reference)
            cmd.disable(model_name)

    return reference

def script(args):
    reference = display_models({reference.stem: reference for reference in map(Path, args.references or [])})
    for batch_pattern in args.batches:
        for batch in sorted(Path().glob(f"boltz_results_{batch_pattern}")):
            batch_predictions = batch / "predictions"
            targets = {target.stem: target for target in sorted(batch_predictions.rglob("*.cif"))}
            reference = display_models(targets, reference=reference)


def main(args):
    pymol.launch(["-l", __file__, "--",
                  *(["-r", *args.references] if args.references else []),
                  *(["-b", *args.batches] if args.batches else [])
    ])

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--references", nargs="+", action="extend", help="Reference model files (pdb, cif, etc.)")
parser.add_argument("-b", "--batches", nargs="+", action="extend", help="List of batches to open")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
else:
    script(args)