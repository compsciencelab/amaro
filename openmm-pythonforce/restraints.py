import warnings

import numpy as np
import openmm as mm
import openmm.app as app
from moleculekit.molecule import Molecule
from openmm.unit import angstrom

KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2 = 4.184 / 0.01
POCKET_PROTEIN_SELECTION = "protein and noh"


def load_reference_molecule(pdb_path, psf_path, noh_mode):
    mol = Molecule(pdb_path)
    if psf_path is not None:
        mol.read(psf_path)
    if noh_mode is not None:
        mol.filter(noh_mode, guessBonds=False)
    return mol


def atom_indices(mol, selection, label):
    indices = np.flatnonzero(mol.atomselect(selection)).astype(np.int64)
    if len(indices) == 0:
        raise ValueError(f"Selection '{selection}' matched zero atoms for {label}.")
    return indices


def residue_key(mol, index):
    return (
        mol.segid[index],
        mol.chain[index],
        mol.resid[index],
        mol.insertion[index],
        mol.resname[index],
    )


def residue_label(key):
    segid, chain, resid, insertion, resname = key
    label_parts = []
    if segid:
        label_parts.append(str(segid))
    if chain:
        label_parts.append(str(chain))
    label_parts.append(f"{resid}{insertion}:{resname}")
    return ":".join(label_parts)


def min_distance_to_ligand(mol, atom_indices_, ligand_indices):
    atom_coords = mol.coords[atom_indices_, :, 0].astype(np.float64)
    ligand_coords = mol.coords[ligand_indices, :, 0].astype(np.float64)
    diff = atom_coords[:, None, :] - ligand_coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def closest_pocket_residues(mol, ligand_indices, cutoff_a, max_residues):
    protein_indices = atom_indices(mol, POCKET_PROTEIN_SELECTION, "pocket protein")
    distances = min_distance_to_ligand(mol, protein_indices, ligand_indices)
    residue_distances = {}
    for atom_index, distance in zip(protein_indices, distances):
        key = residue_key(mol, atom_index)
        residue_distances[key] = min(
            float(distance), residue_distances.get(key, np.inf)
        )
    residues = [
        (key, distance)
        for key, distance in residue_distances.items()
        if distance <= cutoff_a
    ]
    residues.sort(key=lambda item: item[1])
    residues = residues[:max_residues]
    if len(residues) == 0:
        raise ValueError(
            f"No protein residues found within {cutoff_a:.2f} A of the ligand. "
            "Increase --flat-bottom-pocket-cutoff or check --ligand-sel."
        )
    return [key for key, _ in residues]


def ca_indices_for_residues(mol, residues):
    ca_indices = []
    for index in atom_indices(mol, POCKET_PROTEIN_SELECTION, "pocket protein"):
        if mol.name[index] == "CA" and residue_key(mol, index) in residues:
            ca_indices.append(int(index))
    if len(ca_indices) == 0:
        raise ValueError("The selected pocket residues contain no CA atoms.")
    return np.asarray(ca_indices, dtype=np.int64)


def ca_indices_outside_pocket(mol, ligand_indices, cutoff_a):
    protein_indices = atom_indices(
        mol, POCKET_PROTEIN_SELECTION, "CA restraint protein"
    )
    distances = min_distance_to_ligand(mol, protein_indices, ligand_indices)
    pocket_residues = {
        residue_key(mol, index) for index in protein_indices[distances <= cutoff_a]
    }
    all_ca_atoms_count = sum(mol.name[index] == "CA" for index in protein_indices)
    ca_indices = [
        int(index)
        for index in protein_indices
        if mol.name[index] == "CA" and residue_key(mol, index) not in pocket_residues
    ]
    if len(ca_indices) == 0:
        raise ValueError(
            "Automatic CA restraint selection found zero CA atoms outside the pocket. "
            "Decrease --ca-harmonic-pocket-cutoff."
        )
    return np.asarray(ca_indices, dtype=np.int64), pocket_residues, all_ca_atoms_count


def parse_flat_bottom_box_size(value):
    value = str(value).strip()
    if value.endswith(".npy"):
        values = np.asarray(np.load(value), dtype=np.float64).reshape(-1)
    else:
        values = np.asarray(
            [float(item.strip()) for item in value.split(",") if item.strip()],
            dtype=np.float64,
        )
    if len(values) == 1:
        values = np.repeat(values, 3)
    if len(values) != 3 or np.any(values <= 0):
        raise ValueError(
            "--flat-bottom-box-size must be a positive scalar, x,y,z list, or .npy with 1 or 3 values."
        )
    return values


def positions_to_angstrom_array(positions):
    if hasattr(positions, "value_in_unit"):
        return np.asarray(positions.value_in_unit(angstrom), dtype=np.float64)
    return np.asarray(positions, dtype=np.float64)


def format_vector(values):
    return ",".join(f"{value:.3f}" for value in values)


def check_ligand_centroid_in_flat_bottom_box(
    initial_positions, ligand_indices, center_a, box_size_a
):
    initial_coords_a = positions_to_angstrom_array(initial_positions)
    ligand_centroid_a = initial_coords_a[ligand_indices].mean(axis=0)
    half_box_a = box_size_a / 2.0
    delta_a = ligand_centroid_a - center_a
    if np.any(np.abs(delta_a) > half_box_a):
        required_size_a = 2.0 * np.abs(delta_a)
        warnings.warn(
            "Initial ligand centroid is outside the flat-bottom box. "
            f"centroid={format_vector(ligand_centroid_a)} A, center={format_vector(center_a)} A, "
            f"delta={format_vector(delta_a)} A, box_size={format_vector(box_size_a)} A. "
            f"Use at least {format_vector(required_size_a)} A for --flat-bottom-box-size, "
            "or regenerate initial coordinates closer to the pocket."
        )
    return ligand_centroid_a, delta_a


def write_pdb_with_original_ids(topology, positions, path):
    with open(path, "w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)


def add_ca_harmonic_restraint(system, reference_mol, ca_indices, k_kcal_mol_a2):
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k_kcal_mol_a2 * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    reference_coords_nm = reference_mol.coords[:, :, 0].astype(np.float64) * 0.1
    for index in ca_indices:
        force.addParticle(int(index), reference_coords_nm[int(index)].tolist())
    force.setForceGroup(1)
    system.addForce(force)
    print(f"Added CA harmonic restraint on {len(ca_indices)} atom(s).")


def add_ligand_flat_bottom_restraint(
    system, ligand_indices, center_a, box_size_a, k_kcal_mol_a2
):
    half_box_nm = box_size_a * 0.05
    center_nm = center_a * 0.1
    energy = (
        "0.5*k*(dx^2+dy^2+dz^2);"
        "dx=select(step(abs(x1-x0)-hx), abs(x1-x0)-hx, 0);"
        "dy=select(step(abs(y1-y0)-hy), abs(y1-y0)-hy, 0);"
        "dz=select(step(abs(z1-z0)-hz), abs(z1-z0)-hz, 0)"
    )
    force = mm.CustomCentroidBondForce(1, energy)
    force.addPerBondParameter("k")
    force.addPerBondParameter("hx")
    force.addPerBondParameter("hy")
    force.addPerBondParameter("hz")
    force.addPerBondParameter("x0")
    force.addPerBondParameter("y0")
    force.addPerBondParameter("z0")
    group = force.addGroup([int(index) for index in ligand_indices])
    force.addBond(
        [group],
        [
            k_kcal_mol_a2 * KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2,
            *half_box_nm.tolist(),
            *center_nm.tolist(),
        ],
    )
    force.setForceGroup(2)
    system.addForce(force)
    print(
        "Added ligand centroid flat-bottom restraint: "
        f"{len(ligand_indices)} atom(s), center {center_a} A, box {box_size_a} A."
    )


def add_restraints(system, args, initial_positions):
    reference_mol = load_reference_molecule(
        args.bound_pdb or args.pdb, args.psf, args.noh
    )
    if reference_mol.numAtoms != system.getNumParticles():
        raise ValueError(
            f"Reference PDB atom count ({reference_mol.numAtoms}) does not match "
            f"OpenMM system particle count ({system.getNumParticles()})."
        )
    ligand_indices = atom_indices(reference_mol, args.ligand_sel, "ligand")
    info = {
        "ligand_selection": args.ligand_sel,
        "ligand_restraint_atoms": len(ligand_indices),
    }

    if args.k_harmonic > 0:
        ca_indices, excluded_residues, all_ca_atoms_count = ca_indices_outside_pocket(
            reference_mol, ligand_indices, args.ca_harmonic_pocket_cutoff
        )
        add_ca_harmonic_restraint(system, reference_mol, ca_indices, args.k_harmonic)
        info["all_ca_atoms_count"] = all_ca_atoms_count
        info["ca_harmonic_atoms"] = len(ca_indices)
        info["ca_harmonic_k_kcal_mol_a2"] = args.k_harmonic
        info["ca_harmonic_excluded_residues"] = len(excluded_residues)

    if args.k_flat_bottom > 0:
        pocket_residues = closest_pocket_residues(
            reference_mol,
            ligand_indices,
            args.flat_bottom_pocket_cutoff,
            args.flat_bottom_pocket_residues,
        )
        pocket_ca_indices = ca_indices_for_residues(reference_mol, pocket_residues)
        center_a = (
            reference_mol.coords[pocket_ca_indices, :, 0]
            .mean(axis=0)
            .astype(np.float64)
        )
        box_size_a = parse_flat_bottom_box_size(args.flat_bottom_box_size)
        ligand_centroid_a, ligand_delta_a = check_ligand_centroid_in_flat_bottom_box(
            initial_positions, ligand_indices, center_a, box_size_a
        )
        add_ligand_flat_bottom_restraint(
            system, ligand_indices, center_a, box_size_a, args.k_flat_bottom
        )
        info["flat_bottom_k_kcal_mol_a2"] = args.k_flat_bottom
        info["flat_bottom_box_size_a"] = format_vector(box_size_a)
        info["flat_bottom_center_a"] = format_vector(center_a)
        info["flat_bottom_initial_ligand_centroid_a"] = format_vector(ligand_centroid_a)
        info["flat_bottom_initial_ligand_delta_a"] = format_vector(ligand_delta_a)
        info["flat_bottom_min_required_box_size_a"] = format_vector(
            2.0 * np.abs(ligand_delta_a)
        )
        info["flat_bottom_initial_ligand_inside_box"] = bool(
            np.all(np.abs(ligand_delta_a) <= box_size_a / 2.0)
        )
        info["flat_bottom_pocket_residues"] = len(pocket_residues)
        info["flat_bottom_pocket_residue_ids"] = ",".join(
            residue_label(key) for key in pocket_residues
        )
        info["flat_bottom_pocket_ca_atoms"] = len(pocket_ca_indices)
    return info
