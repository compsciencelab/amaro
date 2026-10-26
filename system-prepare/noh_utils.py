import warnings

import numpy as np

# Use the same component definitions for bead summaries and output metadata.
# "ligands" includes cofactors and other non-polymer molecules.
COMPONENT_SELECTIONS = {
    "protein": "protein",
    "ligands": "not (protein or nucleic or lipids or water or ions)",
    "others": "nucleic or lipids or water or ions",
}

# fmt: off
NOH_MAP_DICT = {
    "CH0": 1, "CH1": 2, "CH2": 3, "CH3": 4,
    "NH0": 5, "NH1": 6, "NH2": 7, "NH3": 8,
    "OH0": 9, "OH1": 10, "SH0": 11, "SH1": 12,
    "OH2": 13,
    "B": 14, "F": 15, "Si": 16, "P": 17,
    "Cl": 18, "Br": 19, "I": 20,
    "Na": 21, "K": 22, "Ca": 23, "Mg": 24,
    "Co": 25, "Cu": 26, "Zn": 27,
}

# fmt: on


def get_constraints(mol):
    element = np.asarray(mol.element)
    bonds = np.asarray(mol.bonds, dtype=np.int64)
    if bonds.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    hydrogen_to_heavy = {}
    hh_bonds = 0
    for bond in bonds:
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= mol.numAtoms or j >= mol.numAtoms:
            continue

        ei, ej = element[i], element[j]
        if ei != "H" and ej != "H":
            continue
        if ei == "H" and ej == "H":
            hh_bonds += 1
            continue

        heavy_idx, hydrogen_idx = (j, i) if ei == "H" else (i, j)
        previous_heavy = hydrogen_to_heavy.get(hydrogen_idx)
        if previous_heavy is not None and previous_heavy != heavy_idx:
            raise ValueError(
                f"Hydrogen atom {hydrogen_idx} is bonded to multiple heavy atoms in "
                f"{mol.viewname or 'molecule'}."
            )
        hydrogen_to_heavy[hydrogen_idx] = heavy_idx

    if hh_bonds > 0:
        warnings.warn(
            f"Found {hh_bonds} hydrogen-hydrogen bonds in {mol.viewname or 'molecule'}; ignoring them.",
            RuntimeWarning,
        )

    if not hydrogen_to_heavy:
        return np.empty((0, 2), dtype=np.int64)

    return np.array(
        sorted((heavy_idx, hydrogen_idx) for hydrogen_idx, heavy_idx in hydrogen_to_heavy.items()),
        dtype=np.int64,
    )


def noh_map(mol, constraints=None):
    """Label atoms by element and attached H count, reusing known constraints."""
    hydrogen_counts = np.zeros(mol.numAtoms, dtype=int)
    if constraints is None:
        constraints = get_constraints(mol)
    if constraints.size:
        heavy_idxs, _ = constraints.T
        np.add.at(hydrogen_counts, heavy_idxs, 1)

    # C/N/O/S always encode H count. Other elements pass through when unbound
    # to H; NOH_MAP_DICT is the single list of supported bead species.
    return np.array([
        f"{element}H{count}" if element in {"C", "N", "O", "S"} or count else element
        for element, count in zip(mol.element, hydrogen_counts)
    ])


def build_noh_embedding(mol):
    noh_idxs = mol.get("index", sel="noh", guessBonds=False)
    constraints = get_constraints(mol)
    atomtype = noh_map(mol, constraints)
    try:
        noh_emb = np.array([NOH_MAP_DICT[at] for at in atomtype[noh_idxs]], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(
            f"Species not defined in NOH_MAP_DICT for {mol.viewname or 'molecule'}: {exc.args[0]}"
        ) from exc
    for component, selection in COMPONENT_SELECTIONS.items():
        indices = mol.get("index", sel=f"({selection}) and noh", guessBonds=False)
        if len(indices):
            labels = atomtype[indices]
            print(f"NOH beads ({component}, {len(indices)}): {labels[:5]} ... {labels[-5:]}")
    return noh_idxs, constraints, atomtype, noh_emb
