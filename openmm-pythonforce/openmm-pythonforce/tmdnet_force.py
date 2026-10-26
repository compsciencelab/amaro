"""OpenMM PythonForce adapter for TorchMD-Net energy and force models."""

import numpy as np
import openmm as mm
import torch
from openmm import unit


class _ComputeTorchMDNet:
    """Evaluate a TorchMD-Net energy model in OpenMM units.

    The model may return forces itself, or just energy (as used by
    TMDNETCalculator). Autograd must remain enabled in either case.
    """

    def __init__(
        self,
        model,
        z,
        q,
        batch,
        length_scale,
        energy_scale,
        periodic,
        compile=False,
    ):
        self.model = model.eval()
        self.compiled_model = None
        self.z = z
        self.q = q
        self.batch = batch
        self.length_scale = length_scale
        self.energy_scale = energy_scale
        self.periodic = periodic
        self.compile = compile
        self.dtype = next(model.parameters(), torch.empty(0)).dtype
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, state):
        pos = (
            torch.tensor(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=self.dtype,
                device=self.z.device,
            )
            / self.length_scale
        )
        pos.requires_grad_(True)
        box = None
        if self.periodic:
            box = (
                torch.tensor(
                    state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                        unit.nanometer
                    ),
                    dtype=self.dtype,
                    device=self.z.device,
                )
                / self.length_scale
            )

        with torch.enable_grad():
            model = (
                self.compiled_model if self.compiled_model is not None else self.model
            )
            energy, forces = model(
                z=self.z, pos=pos, batch=self.batch, q=self.q, box=box
            )
            if forces is None or forces.numel() == 0:
                forces = -torch.autograd.grad(energy.sum(), pos)[0]

        energy = energy.detach().sum().item() * self.energy_scale
        forces = forces.detach().cpu().numpy() * (self.energy_scale / self.length_scale)
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise ValueError("TorchMD-Net returned non-finite energy or forces.")
        if forces.shape != (len(self.z), 3):
            raise ValueError(
                f"Expected forces with shape ({len(self.z)}, 3), got {forces.shape}."
            )

        # The eager call initializes cached batch sizes before full-graph tracing.
        # Copy results to host before another call can reuse CUDA graph storage.
        if self.compile and self.compiled_model is None:
            self.compiled_model = torch.compile(
                self.model,
                backend="inductor",
                dynamic=False,
                fullgraph=True,
                mode="reduce-overhead",
            )
        return energy, forces


def add_python_force(system, species_ids, model_ckpt, args, periodic):
    """Attach a checkpoint potential without rebuilding particles or masses."""
    from torchmdnet.models.model import load_model
    import torchmdnet

    from noh import total_charge_from_arg
    from system import ENERGY_SCALE, LENGTH_SCALE

    if not hasattr(mm, "PythonForce"):
        raise RuntimeError(
            "This runner requires an OpenMM build with PythonForce (amaro-pl-2)."
        )
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use --device cpu --platform CPU for a CPU run."
        )
    kwargs = dict(args.external)
    # Do not change checkpoint architecture based on the presence of --charges.
    # Compute forces from energy, following TMDNETCalculator.
    kwargs["derivative"] = False
    kwargs["device"] = device
    kwargs.setdefault("max_num_neighbors", 128)
    kwargs.setdefault("static_shapes", args.compile)
    if args.compile and not kwargs["static_shapes"]:
        raise ValueError("--compile requires external.static_shapes=true.")
    remove_ref_energy = not args.include_atomref_energy
    if (
        "remove_ref_energy" in kwargs
        and kwargs["remove_ref_energy"] != remove_ref_energy
    ):
        raise ValueError(
            "Use --include-atomref-energy to control reference energies; external.remove_ref_energy conflicts."
        )
    kwargs["remove_ref_energy"] = remove_ref_energy
    if args.coulomb_cutoff > 0:
        kwargs["coulomb_cutoff"] = (
            args.coulomb_cutoff * LENGTH_SCALE[args.model_length_unit]
        )
    if args.reaction_field_dielectric is not None:
        kwargs["coulomb_epsilon_solvent"] = args.reaction_field_dielectric
    model = load_model(model_ckpt, **kwargs).eval()
    if args.reaction_field_dielectric is not None:
        output_model = model.output_model
        if not hasattr(output_model, "epsilon_solvent"):
            raise ValueError(
                "--reaction-field-dielectric requires a model with cutoff Coulomb interactions."
            )
        output_model.epsilon_solvent = float(args.reaction_field_dielectric)
    dtype = next(model.parameters()).dtype
    z = torch.tensor(species_ids, dtype=torch.long, device=device)
    batch = torch.zeros_like(z)
    q = torch.tensor([total_charge_from_arg(args.charges)], dtype=dtype, device=device)
    compute = _ComputeTorchMDNet(
        model,
        z,
        q,
        batch,
        1.0 / LENGTH_SCALE[args.model_length_unit],
        ENERGY_SCALE[args.model_energy_unit],
        periodic,
        compile=args.compile,
    )
    force = mm.PythonForce(compute)
    force.setForceGroup(0)
    force.setUsesPeriodicBoundaryConditions(periodic)
    system.addForce(force)
    print(f"TorchMD-Net source: {torchmdnet.__file__}")
    print(
        f"PythonForce: model={model_ckpt}, particles={len(z)}, total charge={q.item()} e, "
        f"energy={args.model_energy_unit}, length={args.model_length_unit}, periodic={periodic}"
    )
    return force
