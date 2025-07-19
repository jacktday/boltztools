#!/usr/bin/env python3

import argparse
from pathlib import Path
import pymol
from pymol import cmd

def script(args):
    first_target_record_id = None
    for batch_index, batch in enumerate([batch for batch_pattern in args.batches for batch in Path().glob(f"boltz_results_{batch_pattern}")]):
        batch_predictions = batch / "predictions"
        for target_index, target in enumerate(batch_predictions.rglob("*_model_0.cif")):
            target_record_id = target.parts[-2]

            if first_target_record_id is None:
                first_target_record_id = target_record_id

            cmd.load(
                target,
                target_record_id
            )
            if batch_index > 0 or target_index > 0:
                cmd.align(target_record_id, first_target_record_id)
                cmd.disable(target_record_id)

def main(args):
    pymol.launch(["-l", __file__, "--",
                  *(["-b", *args.batches] if args.batches else [])
    ])

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batches", nargs="+", action="extend", help="List of batches to open")
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
else:
    script(args)