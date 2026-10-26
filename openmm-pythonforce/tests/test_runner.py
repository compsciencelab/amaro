import ast
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm as mm
import openmm.app as app
import pytest
import torch
import yaml
from ase.io import read
from openmm import unit
from torchmdnet.calculators import TMDNETCalculator
from torchmdnet.models.model import create_model

from config import get_args
from noh import NOH_MASSES, NOH_MAP_REVERSE
from system import build_full_nnp_system, configure_periodic_box
from tmdnet_force import _ComputeTorchMDNet, add_python_force

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).with_name("data")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    args = dict(
        model="tensornet2",
        activation="silu",
        atom_filter=-1,
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        embedding_dimension=8,
        equivariance_invariance_group="O(3)",
        max_z=128,
        num_layers=1,
        num_rbf=4,
        q_dim=2,
        precision=32,
        prior_model=None,
        derivative=False,
        static_shapes=False,
        output_model="Scalar",
        rbf_type="expnorm",
        reduce_op="add",
        trainable_rbf=False,
        max_num_neighbors=64,
        remove_ref_energy=False,
        charge=False,
    )
    torch.manual_seed(123)
    model = create_model(args)
    path = tmp_path_factory.mktemp("model") / "model.ckpt"
    torch.save({"hyper_parameters": args, "state_dict": model.state_dict()}, path)
    return path


def test_common_arguments_match_native():
    def arguments(path):
        tree = ast.parse(path.read_text())
        return {
            node.args[0].value: {
                kw.arg: ast.dump(kw.value)
                for kw in node.keywords
                if kw.arg in {"default", "action", "type", "choices"}
            }
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        }

    native_path = ROOT.parent / "openmm-tensornetforce" / "config.py"
    if not native_path.exists():
        pytest.skip("Native runner is not included in this standalone release")
    native = arguments(native_path)
    python = arguments(ROOT / "openmm-pythonforce" / "config.py")
    native.pop("--model-xml")
    for key, value in native.items():
        assert python[key] == value, key


def test_yaml_and_validation(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model_ckpt": "test.ckpt",
                "pdb": "test.pdb",
                "steps": 20,
                "outdir": str(tmp_path),
                "external": {"max_num_neighbors": 32},
                "use_increased_masses": True,
            }
        )
    )
    _, args = get_args(["--conf", str(config), "--steps", "4"])
    assert args.steps == 4
    assert args.external == {"max_num_neighbors": 32}
    assert args.use_increased_masses
    with pytest.raises(SystemExit):
        get_args(["--conf", str(config), "--comp-time", "0.5"])
    with pytest.raises(SystemExit):
        get_args(["--conf", str(config), "--hydrogen-mass", "3"])


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize(
    "energy_unit,scale", [("kcal/mol", 4.184), ("eV", 96.48533212331002)]
)
def test_checkpoint_matches_calculator_and_finite_difference(
    checkpoint, periodic, energy_unit, scale
):
    atoms = read(DATA / "caffeine.pdb")
    atoms.info["charge"] = 0.5
    atoms.set_cell(np.eye(3) * 40)
    atoms.set_pbc(periodic)
    atoms.calc = TMDNETCalculator(str(checkpoint), max_num_neighbors=128)
    expected_energy = atoms.get_potential_energy() * scale
    expected_forces = atoms.get_forces() * scale * 10
    system = mm.System()
    for mass in atoms.get_masses():
        system.addParticle(float(mass))
    system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 4 * unit.nanometer))
    args = SimpleNamespace(
        seed=1,
        device="cpu",
        external={},
        compile=False,
        include_atomref_energy=False,
        model_length_unit="angstrom",
        model_energy_unit=energy_unit,
        coulomb_cutoff=0,
        reaction_field_dielectric=None,
        charges="0.5",
    )
    add_python_force(system, atoms.numbers, str(checkpoint), args, periodic)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    positions = atoms.positions * 0.1
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    assert state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    ) == pytest.approx(expected_energy, rel=1e-5)
    np.testing.assert_allclose(
        state.getForces(asNumpy=True)._value, expected_forces, rtol=2e-4, atol=2e-4
    )
    delta = 1e-4
    energies = []
    for sign in [-1, 1]:
        displaced = positions.copy()
        displaced[0, 0] += sign * delta
        context.setPositions(displaced)
        energies.append(context.getState(getEnergy=True).getPotentialEnergy()._value)
    assert -(energies[1] - energies[0]) / (2 * delta) == pytest.approx(
        expected_forces[0, 0], rel=0.04, abs=0.01 * scale
    )
    del context, integrator


@pytest.mark.parametrize("returned_forces", [False, True])
@pytest.mark.parametrize("length_scale", [0.1, 1.0])
def test_callback_force_modes_and_length_units(returned_forces, length_scale):
    class Harmonic(torch.nn.Module):
        def forward(self, z, pos, batch, q, box):
            assert torch.is_grad_enabled()
            energy = (pos.square().sum() + q.sum()).reshape(1, 1)
            return energy, -2 * pos if returned_forces else None

    class State:
        def getPositions(self, asNumpy):
            return np.array([[0.2, 0.3, 0.4]]) * unit.nanometer

    compute = _ComputeTorchMDNet(
        Harmonic(),
        torch.tensor([6]),
        torch.tensor([0.5]),
        torch.tensor([0]),
        length_scale,
        4.184,
        False,
    )
    with torch.no_grad():
        energy, forces = compute(State())
    pos = np.array([[0.2, 0.3, 0.4]]) / length_scale
    assert energy == pytest.approx((np.square(pos).sum() + 0.5) * 4.184)
    np.testing.assert_allclose(forces, -2 * pos * 4.184 / length_scale, rtol=1e-6)


def test_noh_selection_and_increased_masses():
    system, top, z, pos, box = build_full_nnp_system(
        str(DATA / "caffeine.pdb"),
        noh_mode="noh and element C",
        use_increased_masses=True,
    )
    assert len(z) == len(pos) == top.getNumAtoms() == system.getNumParticles()
    for index, species_id in enumerate(z):
        assert (
            system.getParticleMass(index)._value
            == NOH_MASSES[NOH_MAP_REVERSE[int(species_id)]]
        )
    args = SimpleNamespace(
        periodic_box="40", periodic_box_unit="angstrom", force_nonperiodic=False
    )
    assert configure_periodic_box(system, top, args, box)
    np.testing.assert_allclose(
        [vector.value_in_unit(unit.nanometer) for vector in system.getDefaultPeriodicBoxVectors()],
        np.eye(3) * 4,
    )
    args.force_nonperiodic = True
    assert not configure_periodic_box(system, top, args, box)
    assert top.getPeriodicBoxVectors() is None


@pytest.mark.parametrize("clock_time", [False, True])
def test_runner_outputs(checkpoint, tmp_path, monkeypatch, clock_time):
    import run

    argv = [
        "run.py",
        "--model-ckpt",
        str(checkpoint),
        "--pdb",
        str(DATA / "caffeine.pdb"),
        "--device",
        "cpu",
        "--platform",
        "Reference",
        "--output-freq",
        "1",
        "--time-step",
        "0.1",
        "--seed",
        "42",
        "--force-nonperiodic",
        "--outdir",
        str(tmp_path),
        "--save-final-state",
        "--minimize-steps",
        "1",
        "--constrainHBonds",
        "--hydrogen-mass",
        "3",
    ]
    argv += ["--comp-time", "0.000001"] if clock_time else ["--steps", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    run.main()
    for name in [
        "input.yaml",
        "params.txt",
        "output_structure_minimized.pdb",
        "output_final_state.xml",
        "traj_output.xtc",
    ]:
        assert (tmp_path / name).stat().st_size > 0
    with (tmp_path / "nvt_metrics_output.csv").open() as handle:
        metrics = list(csv.reader(handle))
    assert (
        "Elapsed Time (s)" in metrics[0]
        if clock_time
        else "Time Remaining" in metrics[0]
    )
    with (tmp_path / "forces_output.csv").open() as handle:
        forces = list(csv.DictReader(handle))
    if not clock_time:
        assert len(forces) == 3 * 24
        assert {int(row["step"]) for row in forces} == {1, 2, 3}
        assert {int(row["atom_index"]) for row in forces} == set(range(24))
        assert np.isfinite(
            [
                [
                    float(row[key])
                    for key in ["fx_kJ_mol_nm", "fy_kJ_mol_nm", "fz_kJ_mol_nm"]
                ]
                for row in forces
            ]
        ).all()
