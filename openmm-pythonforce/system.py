import os
import tempfile

import numpy as np
import openmm as mm
import openmm.app as app
from moleculekit.molecule import Molecule
from moleculekit.periodictable import periodictable
from openmm.unit import dalton, nanometer

from noh import (
    NOH_MAP_REVERSE,
    NOH_MASSES,
    species_ids_for_system,
    total_charge_from_arg,
)

KCAL_TO_KJ = 4.184
EV_TO_KJ_PER_MOL = 96.48533212331002
ENERGY_SCALE = {
    "kcal/mol": KCAL_TO_KJ,
    "kcal_per_mol": KCAL_TO_KJ,
    "kcal": KCAL_TO_KJ,
    "eV": EV_TO_KJ_PER_MOL,
    "ev": EV_TO_KJ_PER_MOL,
}
LENGTH_SCALE = {
    "angstrom": 10.0,
    "Angstrom": 10.0,
    "A": 10.0,
    "nm": 1.0,
    "nanometer": 1.0,
}


def compute_dof(system):
    dof = 0
    for i in range(system.getNumParticles()):
        if system.getParticleMass(i) > 0 * dalton:
            dof += 3
    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        if (
            system.getParticleMass(p1) > 0 * dalton
            or system.getParticleMass(p2) > 0 * dalton
        ):
            dof -= 1
    if any(
        isinstance(system.getForce(i), mm.CMMotionRemover)
        for i in range(system.getNumForces())
    ):
        dof -= 3
    return dof


def box_vectors_from_array(values, box_unit):
    values = np.asarray(values, dtype=np.float64).squeeze()
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    if values.ndim == 0:
        values = np.repeat(values, 3)
    if values.shape == (3,):
        matrix = np.diag(values)
    elif values.shape == (3, 1):
        matrix = np.diag(values[:, 0])
    elif values.ndim == 2 and values.shape[0] == 3 and values.shape[1] != 3:
        matrix = np.diag(values[:, 0])
    elif values.ndim == 2 and values.shape[1] == 3 and values.shape[0] != 3:
        matrix = np.diag(values[0])
    elif values.shape == (3, 3):
        matrix = values
    else:
        raise ValueError(
            "Periodic box must be lengths [a,b,c], a 3x3 vector matrix, or a MoleculeKit-style 3xN box array."
        )

    if box_unit in {"angstrom", "Angstrom", "A"}:
        matrix = 0.1 * matrix
    elif box_unit not in {"nm", "nanometer"}:
        raise ValueError(f"Unsupported box unit: {box_unit}")
    if np.any(np.linalg.norm(matrix, axis=1) <= 0):
        return None
    return tuple(mm.Vec3(*row) for row in matrix) * nanometer


def read_box_vectors(path_or_value, box_unit):
    value = str(path_or_value).strip()
    if os.path.exists(value):
        values = np.load(value)
    else:
        values = [
            float(item)
            for item in value.replace("[", "").replace("]", "").split(",")
            if item.strip()
        ]
    return box_vectors_from_array(values, box_unit)


def moleculekit_box_vectors(mol):
    box = getattr(mol, "box", None)
    if box is None:
        return None
    try:
        return box_vectors_from_array(box, "angstrom")
    except ValueError:
        return None


def build_full_nnp_system(
    structure,
    psf=None,
    init_coords=None,
    noh_mode=None,
    embedding=None,
    use_increased_masses=False,
):
    mol = Molecule(structure)
    print(f"unique elements: {np.unique(mol.element)}")
    if psf is not None:
        mol.read(psf)
    species_ids = species_ids_for_system(
        mol, noh_mode=noh_mode, embedding_path=embedding
    )

    if init_coords is not None:
        print(f"Reading coordinates from {init_coords}")
        mol.read(init_coords)
    else:
        print("Using coordinates from the structure file")
    if noh_mode is not None:
        ## THIS TRY-EXCEPT BLOCK IS A HACK TO WORK AROUND A BUG IN MOLECULEKIT 0.13.0
        ## TODO: ask the MoleculeKit developers to fix this in a future release
        try:
            mol.filter(noh_mode, guessBonds=False)
        except TypeError as exc:
            if "unexpected keyword argument 'guessBonds'" not in str(exc):
                raise
            readme = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", "tmdnet-amaro-v2", "examples",
                "openmm_pythonforce", "README.md",
            ))
            raise RuntimeError(
                "Installed MoleculeKit does not support guessBonds in Molecule.filter(). "
                f"See {readme}, section 'MoleculeKit compatibility warning', "
                "for the local fix, then restart Python and rerun."
            ) from exc

    masses = np.asarray(mol.masses, dtype=np.float64)
    if np.any(masses == 0):
        masses = np.asarray(
            [periodictable[element].mass for element in mol.element], dtype=np.float64
        )
    if use_increased_masses:
        masses = np.asarray(
            [
                NOH_MASSES[NOH_MAP_REVERSE[int(species_id)]]
                for species_id in species_ids
            ],
            dtype=np.float64,
        )

    system = mm.System()
    for mass in masses:
        system.addParticle(float(mass) * dalton)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "system.pdb")
        mol.write(pdb_path, writebonds=True)
        pdb = app.PDBFile(pdb_path)
        topology = pdb.topology
        positions = pdb.positions

    box_vectors = topology.getPeriodicBoxVectors() or moleculekit_box_vectors(mol)
    if box_vectors is not None:
        topology.setPeriodicBoxVectors(box_vectors)
    if len(species_ids) != system.getNumParticles():
        raise ValueError(
            f"Species id count ({len(species_ids)}) does not match particle count "
            f"({system.getNumParticles()}). For NOH, --embedding must be the per-particle "
            "species id array for the filtered no-H system."
        )
    if len(positions) != system.getNumParticles():
        raise ValueError(
            f"Position count ({len(positions)}) does not match particle count ({system.getNumParticles()})."
        )
    return system, topology, species_ids, positions, box_vectors


def atom_is_hydrogen(atom):
    return atom.element is not None and atom.element.symbol == "H"


def constrain_hydrogen_bonds(system, topology, positions):
    constrained = {
        frozenset(
            (system.getConstraintParameters(i)[0], system.getConstraintParameters(i)[1])
        )
        for i in range(system.getNumConstraints())
    }
    added = 0
    for atom1, atom2 in topology.bonds():
        if atom_is_hydrogen(atom1) == atom_is_hydrogen(atom2):
            continue
        key = frozenset((atom1.index, atom2.index))
        if key in constrained:
            continue
        pos1 = np.asarray(
            positions[atom1.index].value_in_unit(nanometer), dtype=np.float64
        )
        pos2 = np.asarray(
            positions[atom2.index].value_in_unit(nanometer), dtype=np.float64
        )
        system.addConstraint(
            atom1.index, atom2.index, float(np.linalg.norm(pos1 - pos2)) * nanometer
        )
        constrained.add(key)
        added += 1
    return added


def repartition_hydrogen_masses(system, topology, hydrogen_mass_amu):
    target = float(hydrogen_mass_amu) * dalton
    changed = 0
    for atom1, atom2 in topology.bonds():
        if atom_is_hydrogen(atom1) == atom_is_hydrogen(atom2):
            continue
        hydrogen = atom1 if atom_is_hydrogen(atom1) else atom2
        heavy = atom2 if atom_is_hydrogen(atom1) else atom1
        old_h_mass = system.getParticleMass(hydrogen.index)
        delta = target - old_h_mass
        if abs(delta.value_in_unit(dalton)) < 1e-12:
            continue
        heavy_mass = system.getParticleMass(heavy.index) - delta
        if heavy_mass <= 0 * dalton:
            raise ValueError(
                f"Hydrogen mass repartitioning would make atom {heavy.index} mass non-positive."
            )
        system.setParticleMass(hydrogen.index, target)
        system.setParticleMass(heavy.index, heavy_mass)
        changed += 1
    return changed


def configure_periodic_box(system, topology, args, box_vectors):
    if args.periodic_box is not None:
        box_vectors = read_box_vectors(args.periodic_box, args.periodic_box_unit)
        topology.setPeriodicBoxVectors(box_vectors)
    periodic = (not args.force_nonperiodic) and box_vectors is not None
    if periodic:
        system.setDefaultPeriodicBoxVectors(*box_vectors)
        vectors_nm = [vector.value_in_unit(nanometer) for vector in box_vectors]
        print(f"Periodic boundary conditions: enabled, box vectors={vectors_nm} nm")
    else:
        topology.setPeriodicBoxVectors(None)
        print("Periodic boundary conditions: disabled")
    return periodic
