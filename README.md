> [!WARNING]
> Boltz 2 tokenizes residues at the residue level. This means Boltz 2 does not know about intramolecular bonds. As a result Boltz 2 relies on accurate atom naming of residues to infer bonding. This means any modified residue must conform to the atom naming of the base residue in the ccd. It also likely restricts the complexity of modified residues. Boltz 1 does not appear to have this limitation.

## Introduction

This repository provides tools that enable users to add custom molecules to the CCD. This allows the user to specify atom names and leaving atom flags. Leaving atom flags are not supported for pdb inputs.

## Examples add_mol_to_ccd.py

Adding CCD from rcsb. The CIF will be automatically downloaded from rcsb. Note name must not exceed 5 characters.

```
./add_mol_to_ccd.py -n CLA
```

Adding custom CIF or PDB.

```
./add_mol_to_ccd.py -n MLE -i MLE.cif
```

Specify boltz path manually if you have moved ~/.boltz

```
./add_mol_to_ccd.py -n CLA --boltz_path /opt/boltz
```

## Example view.py
This tool opens and aligns cifs from multiple boltz batches.

Opening boltz_results_batch_A_01, boltz_results_batch_A_02, and boltz_results_batch_B_05.

```
./view.py -b batch_A_01 batch_A_02 batch_B_05
```

Using patterns to avoid fully specifying the name.

```
./view.py -b batch_A_\* batch_B_05
```

Add a reference structures (cif, pdb, etc.).

```
./view.py -r 9ccd_A.cif 9ccd_B.cif -b batch_A_\* batch_B_05
```

## Examples dump.py

Dumping boltz_results_stuff to boltz_results_stuff_dump.

```
./dump.py -p ./boltz_results_stuff
```

Dumping boltz_results_stuff to stuff_dump.

```
./dump.py -p ./boltz_results_stuff -o ./stuff_dump
```

Dumping ccd TRP to TRP_dump.

```
./dump.py --ccd TRP
```

Dumping ccd TRP to TRP.

```
./dump.py --ccd TRP -o ./TRP
```
