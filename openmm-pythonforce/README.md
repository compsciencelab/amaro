# OpenMM PythonForce

Run TorchMD-Net checkpoints with OpenMM's `PythonForce`. This runner imports
`torchmdnet` from the active Python environment. Follow the repository's
installation instructions to install the AMARO version. It does not require
the Acellera TensorNet plugin.
See the [repository README](../README.md) for installation,
checkpoint inventory, and prepared inputs. Use `--model-ckpt` (`-m`) to select a checkpoint.

## Installation

First follow the [Python 3.13 / CUDA 12.6 environment recipe](../README.md#install)
to create `amaro_v2`. From the repository root, reactivate it and check the runner:

```bash
mamba activate amaro_v2
python -c 'import torchmdnet, openmm; print(torchmdnet.__file__); print(openmm.__version__); assert hasattr(openmm, "PythonForce")'
```

The check prints the installed `torchmdnet` location so you can confirm the
active environment uses the intended AMARO version.

## Running

Run from the repository root. Supply a checkpoint trained for the input species
and their ordering; a NOH model needs NOH species ids, not atomic numbers.

```bash
python openmm-pythonforce/run.py \
  --conf openmm-pythonforce/examples/openmm_base_input.yaml
```

The example runs 10 unrestrained, nonperiodic steps with the downloaded `ckpts/amaro_v2.ckpt`
and the included 1BTY_HoloSet NOH inputs. It uses a 1 fs step, 300 K, and no
minimization. This is an execution example, not an equilibration or manuscript
production protocol. Paths in YAML are relative to the
working directory. Put `--conf` before CLI overrides. YAML uses argparse names
such as `model_ckpt`, `output_freq`, and `constrainHBonds`. `--external` accepts
additional `load_model` arguments as a YAML mapping, for example
`--external '{max_num_neighbors: 128}'`.

CUDA is the default for both OpenMM and PyTorch. For a CPU check, add
`--device cpu --platform Reference` (or `--platform CPU`). `--compile` enables
optional PyTorch compilation after an eager warmup; it requires static shapes
and can incur substantial startup cost. Eager inference is the default.

The runner preserves checkpoint architecture, including charge conditioning
and NQE settings. `--charges` supplies a floating-point molecular charge or the
sum of a `.npy` charge array; omitted charges mean zero. It does not change the
model architecture. Forces are computed with autograd from energy, using the
same convention as `TMDNETCalculator`.

Model energies default to kcal/mol and coordinates to angstroms. The runner
converts energies to kJ/mol, coordinates to nm, and forces to kJ/mol/nm.
`--model-energy-unit` and `--model-length-unit` describe the checkpoint's units;
they do not change its learned physics. A periodic box is passed to the model
in model length units, and `--force-nonperiodic` disables it.

`--coulomb-cutoff` is in nm; zero retains the checkpoint's cutoff, while a
positive value overrides it in model length units. `--reaction-field-dielectric`
requires a cutoff Coulomb output head. `--include-atomref-energy` enables the
checkpoint's reference-energy contribution through `load_model`; by default
Atomref priors are disabled. An enabled prior must exist in the checkpoint to
restore reference energies. The released AMARO v2 checkpoints have a Scalar
output head and no Atomref prior: Coulomb-specific controls and restoration of
reference energies do not apply to these checkpoints.

NOH filtering, topology masses or increased NOH masses, X–H constraints,
hydrogen mass repartitioning, and optional CA/ligand restraints are applied to
one system. OpenMM minimization includes the same forces and constraints as MD.
The random seed is passed to NumPy, PyTorch, the Langevin integrator, and initial
velocities; backend implementations can still produce different trajectories.

## Outputs

At `output_freq`, the runner writes:

- `nvt_metrics_<outname>.csv`: step, time, energies, temperature, speed, and remaining/elapsed time.
- `traj_<outname>.xtc`: coordinates without periodic wrapping.
- `forces_<outname>.csv`: `step,time_ps,atom_index,fx_kJ_mol_nm,fy_kJ_mol_nm,fz_kJ_mol_nm`.
  Indices are zero-based in topology order. Forces include restraints and exclude
  constraint impulses and thermostat kicks.

It also writes `input.yaml` and `params.txt`,
`<outname>_structure_minimized.pdb` when minimization is requested, and
`<outname>_final_state.xml` with `--save-final-state`.
Use either `--steps` or `--comp-time` (hours); clock-time runs report elapsed time.

## Lightweight tests

With pytest installed, run the small CPU regression suite from the repository root:

```bash
OMP_NUM_THREADS=1 python -m pytest openmm-pythonforce/tests -q
```

Tests use the installed `torchmdnet` package and local molecular fixtures.
They compare PythonForce against its ASE calculator and
finite-difference forces, check unit conversion, fractional charge, periodic
boxes, NOH masses, CLI parity, and short-run outputs. They do not benchmark CUDA
or require the Acellera plugin.

## Additional run options

Use a new output directory for each run: the runner overwrites report files.
Use `python openmm-pythonforce/run.py --help` for the full interface, including
restraints, periodic boxes, minimization, and clock-time runs.

For example, from the root, a user-chosen 1000-step run with minimization is:

```bash
python openmm-pythonforce/run.py \
  --conf openmm-pythonforce/examples/openmm_base_input.yaml \
  --steps 1000 --output-freq 100 --minimize-steps 100 \
  --outdir outputs/1BTY-1000 --save-final-state
```

These settings illustrate the CLI; they do not specify a validated sampling protocol.
