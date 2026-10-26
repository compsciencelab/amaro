# NOH System Preparation

This utility prepares a PDB plus optional PSF or PRMTOP topology for a NOH OpenMM run. It uses HTMD `systemPrepare` to select protein protonation states, removes hydrogens into NOH bead types, retains non-water non-protein components, and writes a matching NOH PDB, PSF, embedding, and total charge.

`prepare_noh_system.py` and `noh_utils.py` are simulation-input utilities. They are separate from the Hugging Face release code because they create a new prepared system rather than convert an existing all-atom parquet record.

## Run

Run from the release root in an environment providing MoleculeKit's
`systemPrepare` and its preparation dependencies. This can be separate from the
OpenMM inference environment. The supplied prepared example does not require
rerunning protonation.

```bash
python system-prepare/prepare_noh_system.py \
  --pdb /path/to/structure.pdb \
  --topology /path/to/structure.psf \
  --sys-id my-system --ph 7.45 \
  --output-root system-prepare/noh-data
```

Omit `--topology` for PDB-only preparation. Provide ligand templates where required
by the chemistry rules below.

`--ph` is required. It is a preparation-time protonation assumption, stored in `metadata.json`, and should be chosen for the intended system and experiment. Do not use it to relabel an already prepared mdCATH or KDeep topology: the current Parquet conversion copies their existing protonation states and has no source-level pH field.

`--filter-str` accepts a MoleculeKit atom selection and defaults to `all`.
The input is loaded and selected once, before protein and non-protein preparation,
so the same selection applies to protein chains, ligands, ions, and waters.
For example:

```bash
# All protein chains, without ligands or ions:
... --filter-str 'protein'

# Selected chains, excluding a particular component:
... --filter-str '(chain A B) and not resname BEN'
```

Select complete residues with the heavy atoms required for preparation. Protein
hydrogens are regenerated in the default pKa mode. `--exclude-resnames` removes
additional residues after selection; waters are removed unless `--keep-water`
is set. Empty selections raise an error. Existing protein chain IDs are retained;
MoleculeKit `autoSegment` assigns IDs only to protein atoms whose chain is blank.

The default `--protein-charge-method pka` resets input protein protonation variants
and uses `systemPrepare` to predict states at the requested pH. Predicted side-chain
charges are checked against the selected states. `STATE_CHARGES` remains necessary
for this check and for the optional `residue-states` method: this MoleculeKit version
does not populate usable atomic charges in `systemPrepare`'s returned molecule.
Standard terminal charges are counted from the prepared N/OXT bonds for each chain,
including neutral termini, rather than assuming all terminal charges cancel.
The protonation CSV/JSON reports describe side chains; metadata records the terminal
contribution separately as `protein_terminal_charge`.

Without `--topology`, non-protein residues are prepared separately using MoleculeKit
`templateResidueFromSmiles` to assign bond orders, formal charges, and hydrogens.
Single-atom templates such as water use RDKit hydrogen addition. BEN uses a
benzamidinium pKa of 11.6 to select its dominant aqueous protonation state. At
pH 7.45 it has charge +1 and two NH2 beads (embedding 7). This is an explicit
BEN rule, not a general ligand pKa predictor. Heavy-atom coordinates are retained.
Generic names such as `MOL` or `LIG` do not imply benzamidine chemistry.

Monatomic ions use `ION_CHARGES`, restricted to elements in `NOH_MAP_DICT`:
Na/K +1, Ca/Mg/Zn +2, and F/Cl/Br/I −1. Explicit PDB formal charges take
precedence; Co/Cu require such a charge or a supplied topology. Calcium remains
+2 when coordinated to water. With `--keep-water`, water hydrogens are added
and the oxygen maps to embedding 13. Waters are otherwise removed.

Other ligands require a SMILES template specifying the desired protonation
state, for example `--ligand-smiles 'LIG=NC(=[NH2+])c1ccccc1'`. This option can
be repeated and can override BEN's built-in rule. Unknown residues cause an
error instead of silently receiving missing hydrogens or zero charge.
With `--topology`, existing non-protein hydrogens, bonds, and partial charges
are preserved; supply a fully prepared all-atom structure. Template overrides
cannot be combined with `--topology`.

For the original 3PTB PDB at pH 7.45, the existing command needs no extra
options: BEN contributes +1 and calcium +2, giving a non-protein charge of +3.
PDB-only non-protein charges are formal charges for total-charge bookkeeping,
not force-field partial charges.

Bead summaries and metadata use MoleculeKit selections to distinguish `protein`,
`ligands` (non-polymer molecules, including cofactors), and `others` (ions, water,
nucleic acids, and lipids). Classification does not depend on ligand names.
Preparation still requires a template or supplied topology for
non-protein components, and all exported elements must be supported by `NOH_MAP_DICT`.

The output directory contains `noh.pdb`, `noh.psf`, `noh-embedding.npy`, `noh-charge.npy`, `total-charge.npy`, and `metadata.json`. Pass all four simulation inputs, including the embedding, to PythonForce.
`noh-charge.npy` stores the molecular charge in its first entry and zeros elsewhere;
it is not a vector of physical partial charges.

```bash
python scripts/amaro.py validate-inputs system-prepare/noh-data/my-system
python scripts/amaro.py run system-prepare/noh-data/my-system \
  --checkpoint ckpts/amaro_v2.ckpt --outdir outputs/my-system --steps 10
```

## Included 1BTY example

[`examples/1BTY_HoloSet/`](examples/1BTY_HoloSet/README.md) contains prepared
PDB/PSF coordinates, species ids, molecular charge, and protonation records. Its
metadata records pH 7.45, excluded residue name `CA`, removed water, 1638 NOH beads,
and total charge +9 e. Use these files directly for the release's runnable example.
The original all-atom HoloSet source is not included; the recorded input path is
provenance, not a runtime dependency.

See the [release README](../README.md) for installation and the
[runner README](../openmm-pythonforce/README.md) for simulation options.
