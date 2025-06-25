## Introduction

Boltz-2 currently restricts constraints to CCD molecules. This repository provides tools that enable users to add custom molecules to the CCD, allowing constraints on arbitrary molecules.

## Examples

Adding CCD from rcsb. The CIF will be automatically downloaded from rcsb. Note name must not exceed 5 characters.

```
./add_cif_to_ccd.py -n CLA
```

Adding custom CIF.

```
./add_cif_to_ccd.py -n MLE -i MLE.cif
```

Specify boltz path manually if you have moved ~/.boltz

```
./add_cif_to_ccd.py -n CLA --boltz_path /opt/boltz
```