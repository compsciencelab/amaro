import argparse
import os

import yaml

from system import LENGTH_SCALE


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.name.endswith((".yaml", ".yml")):
            raise ValueError("Configuration file must end with yaml or yml")
        with values as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            parser.error("Configuration must be a YAML mapping.")
        known = {action.dest for action in parser._actions}
        unknown = sorted(key for key in config if key not in known)
        if unknown:
            raise ValueError(
                f"Unknown argument(s) in config file: {', '.join(unknown)}"
            )
        namespace.__dict__.update(config)


def optional_path(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null"}:
        return None
    return value


def save_argparse(args, filename, exclude=None):
    exclude = set(exclude or [])
    data = {key: value for key, value in vars(args).items() if key not in exclude}
    with open(filename, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run OpenMM MD with TorchMD-Net PythonForce"
    )
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )
    parser.add_argument(
        "--model-ckpt", "-m", type=str, default=None, help="TorchMD-Net checkpoint"
    )
    parser.add_argument(
        "--model-energy-unit",
        choices=("kcal/mol", "kcal_per_mol", "kcal", "eV", "ev"),
        default="kcal/mol",
        help="Energy unit used by the ML model; forces use energy per model length unit.",
    )
    parser.add_argument(
        "--model-length-unit",
        choices=tuple(LENGTH_SCALE),
        default="angstrom",
        help="Length unit used by the ML model.",
    )
    parser.add_argument("--pdb", "-p", type=str, help="PDB file of the system")
    parser.add_argument(
        "--psf", type=str, default=None, help="PSF file of the system (optional)"
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=str,
        default=None,
        help="Per-particle species id .npy file",
    )
    parser.add_argument(
        "--charges",
        "-q",
        type=str,
        default=None,
        help="Total charge or .npy charge array.",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        default=".",
        help="Output directory for the simulation",
    )
    parser.add_argument(
        "--outname",
        type=str,
        default="output",
        help="Output name for the simulation files",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=None,
        help="Total steps performed in the simulation",
    )
    parser.add_argument(
        "--output-freq", "-f", type=int, default=1000, help="Output frequency in steps"
    )
    parser.add_argument(
        "--time-step", "-t", type=float, default=3.0, help="Time step in femtoseconds"
    )
    parser.add_argument(
        "--temperature", "-T", type=float, default=350.0, help="Temperature in Kelvin"
    )
    parser.add_argument(
        "--friction-coeff",
        "-g",
        type=float,
        default=1.0,
        help="Friction coefficient in ps^-1",
    )
    parser.add_argument(
        "--minimize-steps",
        type=int,
        default=0,
        help="OpenMM minimization iterations before MD",
    )
    parser.add_argument(
        "--noh",
        type=str,
        default=None,
        help="MoleculeKit filter used to build the NNP system.",
    )
    parser.add_argument(
        "--use-increased-masses",
        action="store_true",
        help="Use NOH bead masses instead of topology masses.",
    )
    parser.add_argument(
        "--init-coords",
        type=str,
        default=None,
        help="Initial coordinates for the simulation",
    )
    parser.add_argument(
        "--comp-time",
        type=float,
        default=None,
        help="Computational time in hours; exclusive with --steps",
    )
    parser.add_argument(
        "--save-final-state",
        action="store_true",
        help="Save the final OpenMM state XML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Langevin noise and velocities",
    )
    parser.add_argument(
        "--force-nonperiodic",
        action="store_true",
        help="Ignore any periodic box vectors",
    )
    parser.add_argument(
        "--periodic-box",
        type=str,
        default=None,
        help="Periodic box lengths/vectors: x,y,z, scalar/3-value .npy, 3x3 .npy, or MoleculeKit-style 3xN .npy.",
    )
    parser.add_argument(
        "--periodic-box-unit", choices=tuple(LENGTH_SCALE), default="angstrom"
    )
    parser.add_argument(
        "--coulomb-cutoff",
        type=float,
        default=0.0,
        help="Coulomb cutoff in nm; zero uses the checkpoint setting",
    )
    parser.add_argument("--reaction-field-dielectric", type=float, default=None)
    parser.add_argument(
        "--include-atomref-energy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--bound-pdb",
        type=str,
        default=None,
        help="Bound-state PDB used for restraint positions and pocket center. Defaults to --pdb.",
    )
    parser.add_argument(
        "--ligand-sel",
        type=str,
        default="resname LIG and noh",
        help="MoleculeKit ligand selection",
    )
    parser.add_argument(
        "--k-harmonic",
        type=float,
        default=0.0,
        help="CA harmonic restraint in kcal/mol/A^2",
    )
    parser.add_argument(
        "--ca-harmonic-pocket-cutoff",
        type=float,
        default=5.0,
        help="Do not restrain CA atoms from residues within this Angstrom cutoff of the bound ligand.",
    )
    parser.add_argument(
        "--k-flat-bottom",
        type=float,
        default=0.0,
        help="Ligand flat-bottom k in kcal/mol/A^2",
    )
    parser.add_argument(
        "--flat-bottom-box-size",
        type=str,
        default="50.0,50.0,50.0",
        help="Full flat-bottom box size in Angstrom: scalar, x,y,z, or a .npy path.",
    )
    parser.add_argument(
        "--flat-bottom-pocket-cutoff",
        type=float,
        default=5.0,
        help="Maximum Angstrom cutoff for residues used to define the pocket centroid.",
    )
    parser.add_argument(
        "--flat-bottom-pocket-residues",
        type=int,
        default=5,
        help="Number of closest ligand-contacting residues used to define the pocket centroid.",
    )
    parser.add_argument(
        "--constrainHBonds",
        action="store_true",
        help="Constrain all bonds involving hydrogen",
    )
    parser.add_argument(
        "--hydrogen-mass",
        type=float,
        default=None,
        help="Hydrogen mass in amu for hydrogen mass repartitioning. Requires --constrainHBonds.",
    )

    parser.add_argument(
        "--external",
        type=yaml.safe_load,
        default={},
        help="Extra load_model arguments as a YAML mapping",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the energy model with torch.compile",
    )
    parser.add_argument(
        "--device", default="cuda", help="PyTorch device, e.g. cuda:0 or cpu"
    )
    parser.add_argument(
        "--platform", choices=("CUDA", "CPU", "Reference"), default="CUDA"
    )

    args = parser.parse_args(argv)
    # argparse does not validate choices supplied by a YAML action.
    for action in parser._actions:
        if (
            action.choices is not None
            and getattr(args, action.dest) not in action.choices
        ):
            parser.error(f"Invalid {action.dest}: {getattr(args, action.dest)!r}")
    args.model_ckpt = optional_path(args.model_ckpt)
    args.pdb = optional_path(args.pdb)
    if args.model_ckpt is None:
        parser.error("--model-ckpt is required.")
    if not isinstance(args.external, dict):
        parser.error("--external must be a YAML mapping.")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative.")
    if args.comp_time is not None and args.comp_time <= 0:
        parser.error("--comp-time must be positive.")
    if args.pdb is None:
        raise SystemExit("--pdb is required.")
    if args.steps is not None and args.comp_time is not None:
        raise SystemExit("Please specify either --comp-time or --steps, but not both.")
    if args.steps is None and args.comp_time is None:
        raise SystemExit("Please specify either --comp-time or --steps.")
    if args.output_freq <= 0:
        raise SystemExit("--output-freq must be positive.")
    if args.time_step <= 0:
        raise SystemExit("--time-step must be positive.")
    if args.seed is not None and args.seed <= 0:
        raise SystemExit("--seed must be a positive integer.")
    if args.hydrogen_mass is not None:
        if args.hydrogen_mass <= 0:
            raise SystemExit("--hydrogen-mass must be positive when supplied.")
        if not args.constrainHBonds:
            raise SystemExit("--hydrogen-mass requires --constrainHBonds.")
    if args.coulomb_cutoff < 0:
        raise SystemExit("--coulomb-cutoff must be non-negative.")
    if (
        args.reaction_field_dielectric is not None
        and args.reaction_field_dielectric <= 0
    ):
        parser.error("--reaction-field-dielectric must be positive.")
    if args.k_harmonic < 0:
        raise SystemExit("--k-harmonic must be non-negative.")
    if args.ca_harmonic_pocket_cutoff <= 0:
        raise SystemExit("--ca-harmonic-pocket-cutoff must be positive.")
    if args.k_flat_bottom < 0:
        raise SystemExit("--k-flat-bottom must be non-negative.")
    if args.flat_bottom_pocket_cutoff <= 0:
        raise SystemExit("--flat-bottom-pocket-cutoff must be positive.")
    if args.flat_bottom_pocket_residues <= 0:
        raise SystemExit("--flat-bottom-pocket-residues must be positive.")

    args.pdb = optional_path(args.pdb)
    args.psf = optional_path(args.psf)
    args.embedding = optional_path(args.embedding)
    args.init_coords = optional_path(args.init_coords)
    args.bound_pdb = optional_path(args.bound_pdb)
    args.periodic_box = optional_path(args.periodic_box)
    args.noh = optional_path(args.noh)
    args.outdir = args.outdir or "."
    os.makedirs(args.outdir, exist_ok=True)
    save_argparse(args, os.path.join(args.outdir, "input.yaml"), exclude=["conf"])
    return args.outdir, args
