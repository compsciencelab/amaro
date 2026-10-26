import numpy as np
from moleculekit.periodictable import periodictable

# fmt: off
NOH_MAP = {
    "CH0": 1, "CH1": 2, "CH2": 3, "CH3": 4,
    "NH0": 5, "NH1": 6, "NH2": 7, "NH3": 8,
    "OH0": 9, "OH1": 10, "SH0": 11, "SH1": 12,
    "OH2": 13, "B": 14, "F": 15, "Si": 16,
    "P": 17, "Cl": 18, "Br": 19, "I": 20,
    "Na": 21, "K": 22, "Ca": 23, "Mg": 24,
    "Co": 25, "Cu": 26, "Zn": 27,
}

NOH_MAP_REVERSE = {value: key for key, value in NOH_MAP.items()}
NOH_MASSES = {
    "CH3": 15.03452, "CH2": 14.02658, "CH1": 13.01864, "CH0": 12.0107,
    "NH3": 17.03052, "NH2": 16.02258, "NH1": 15.01464, "NH0": 14.0067,
    "OH1": 17.00194, "OH0": 15.994, "SH1": 33.07294, "SH0": 32.065,
    "OH2": 18.00988, "B": 10.811, "F": 18.998, "Si": 28.0855,
    "P": 30.9738, "Cl": 35.453, "Br": 79.904, "I": 126.90447,
    "Na": 22.989769, "K": 39.0983, "Ca": 40.078, "Mg": 24.305,
    "Co": 58.9332, "Cu": 63.546, "Zn": 65.38,
}
# fmt: on


def noh_atomtypes(mol):
    if len(mol.bonds) == 0:
        mol.guessBonds()
    hydrogen_counts = np.zeros(mol.numAtoms, dtype=np.int32)
    for i, j in mol.bonds:
        if mol.element[i] == "H" and mol.element[j] != "H":
            hydrogen_counts[j] += 1
        elif mol.element[j] == "H" and mol.element[i] != "H":
            hydrogen_counts[i] += 1

    atomtypes = []
    for i, element in enumerate(mol.element):
        if element == "H":
            atomtypes.append("H")
        elif element in {
            "B",
            "Si",
            "F",
            "Cl",
            "Br",
            "I",
            "P",
            "Na",
            "K",
            "Ca",
            "Mg",
            "Co",
            "Cu",
            "Zn",
        }:
            atomtypes.append(element)
        else:
            atomtypes.append(f"{element}H{hydrogen_counts[i]}")
    return np.asarray(atomtypes)


def species_ids_for_system(mol, noh_mode, embedding_path):
    selected = (
        mol.atomselect(noh_mode)
        if noh_mode is not None
        else np.ones(mol.numAtoms, dtype=bool)
    )
    if embedding_path is not None:
        species_ids = np.asarray(np.load(embedding_path))
        if species_ids.ndim != 1 or len(species_ids) != int(selected.sum()):
            raise ValueError(
                "--embedding must be a one-dimensional array with one id per selected particle."
            )
        if not np.all(np.isfinite(species_ids)) or np.any(
            species_ids != species_ids.astype(np.int64)
        ):
            raise ValueError("--embedding must contain finite integer species ids.")
        return species_ids.astype(np.int32)
    if noh_mode is not None:
        atomtypes = noh_atomtypes(mol.copy())
        if np.any(mol.element[selected] == "H"):
            raise ValueError(
                "--noh must select only heavy atoms when deriving NOH species ids."
            )
        return np.asarray(
            [NOH_MAP[atomtype] for atomtype in atomtypes[selected]], dtype=np.int32
        )
    return np.asarray(
        [periodictable[element].number for element in mol.element], dtype=np.int32
    )


def total_charge_from_arg(value):
    if value is None:
        return 0.0
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        if not value.endswith(".npy"):
            raise ValueError("--charges must be a number or a .npy file")
        return float(np.load(value).sum(dtype=np.float64))
