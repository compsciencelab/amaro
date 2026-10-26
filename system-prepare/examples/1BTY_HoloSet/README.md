# Prepared 1BTY_HoloSet example

This directory contains an already prepared NOH protein–ligand system. According
to the included [metadata](metadata.json), preparation used pH 7.45, excluded
residue name `CA`, and removed water. The final system has 1638 NOH beads: 1629
protein beads and 9 ligand beads. Molecular charge is +9 e (+8 protein, +1 other
components).

| File | Meaning |
| --- | --- |
| `noh.pdb` | NOH coordinates and atom order |
| `noh.psf` | Matching topology |
| `noh-embedding.npy` | One NOH species id per simulated particle |
| `total-charge.npy` | Scalar molecular charge, +9 e |
| `noh-charge.npy` | Compatibility vector: molecular charge in entry zero, zeros elsewhere |
| `protein-protonation.csv`, `protein-protonation.json` | Saved protein protonation/pKa records |
| `metadata.json` | Preparation settings, counts, charge bookkeeping, and source provenance |

The original all-atom HoloSet input referenced in metadata is not included.
The prepared files support an execution example; regenerating them requires the
original input and preparation environment. Do not treat the compatibility charge
vector as per-atom physical partial charges.

From the release root, validate without running MD:

```bash
python scripts/amaro.py validate-inputs system-prepare/examples/1BTY_HoloSet
```

Then run the [short YAML example](../../../openmm-pythonforce/examples/openmm_base_input.yaml):

```bash
python openmm-pythonforce/run.py --conf openmm-pythonforce/examples/openmm_base_input.yaml
```

This uses the included `amaro_v2.ckpt`. It is not a manuscript production protocol.
