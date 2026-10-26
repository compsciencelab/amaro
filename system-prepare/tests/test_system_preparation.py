import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from moleculekit.molecule import Molecule
from openmm.app import CharmmPsfFile

from noh_utils import build_noh_embedding
from prepare_noh_system import (
    assign_pka_charges,
    compute_complex_stats_for_metadata,
    get_mol_prepared_charges,
    get_protein_terminal_charge,
    load_input_molecule,
    prepare_nonprotein,
    prepare_nonprotein_residue,
    prepare_protein,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def peptide(tmp_path):
    # ILE16–GLY18 heavy atoms from 1BTY; no local PDB download is required.
    mol = Molecule().empty(19, numFrames=1)
    mol.record[:] = "ATOM"
    mol.chain[:] = "A"
    mol.resname[:] = ["ILE"] * 8 + ["VAL"] * 7 + ["GLY"] * 4
    mol.resid[:] = [16] * 8 + [17] * 7 + [18] * 4
    mol.name[:] = ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1",
                   "N", "CA", "C", "O", "CB", "CG1", "CG2", "N", "CA", "C", "O"]
    mol.element[:] = [name[0] for name in mol.name]
    mol.coords[:, :, 0] = [
        [35.700, 19.589, 20.234], [35.550, 20.497, 19.066],
        [36.807, 20.237, 18.234], [37.894, 20.256, 18.772],
        [35.544, 21.989, 19.514], [34.399, 22.321, 20.484],
        [35.560, 22.968, 18.278], [33.034, 22.335, 19.785],
        [36.640, 20.021, 16.964], [37.785, 19.760, 16.052],
        [37.896, 21.020, 15.170], [36.905, 21.499, 14.639],
        [37.466, 18.517, 15.170], [38.603, 18.296, 14.156],
        [37.202, 17.225, 16.050], [39.101, 21.479, 15.085],
        [39.440, 22.677, 14.271], [38.928, 24.015, 14.824],
        [38.710, 24.947, 14.072],
    ]
    pdb = tmp_path / "peptide.pdb"
    mol.write(str(pdb))
    return Molecule(str(pdb))


@pytest.fixture
def small_ligand():
    mol = Molecule().empty(3, numFrames=1)
    mol.record[:] = "HETATM"
    mol.chain[:] = "A"
    mol.resname[:] = "MOL"
    mol.resid[:] = 100
    mol.name[:] = ["C1", "C2", "N1"]
    mol.element[:] = ["C", "C", "N"]
    mol.coords[:, :, 0] = [[0, 0, 0], [1.5, 0, 0], [2.0, 1.4, 0]]
    return mol


@pytest.fixture
def complex_pdb(peptide, small_ligand, tmp_path):
    mol = Molecule()
    for i, chain in enumerate("ABCXYZ"):
        part = peptide.copy()
        part.append(small_ligand)
        ion = Molecule().empty(1, numFrames=1)
        ion.record[:] = "HETATM"
        ion.element[:] = "Ca"
        ion.name[:] = "CA"
        ion.resname[:] = "CA"
        ion.resid[:] = 101
        ion.coords[:] = 10
        part.append(ion)
        water = ion.copy()
        water.element[:] = "O"
        water.name[:] = "O"
        water.resname[:] = "HOH"
        water.resid[:] = 102
        water.coords[:] = 15
        part.append(water)
        part.chain[:] = chain
        part.coords[:, 0, :] += 60 * i
        mol.append(part)
    pdb = tmp_path / "complex.pdb"
    mol.write(str(pdb))
    return pdb


@pytest.mark.parametrize("selection,chains", [("all", {"A", "B", "C", "X", "Y", "Z"}),
                                               ("chain X Y Z", {"X", "Y", "Z"})])
def test_complex_chain_selection(complex_pdb, selection, chains):
    mol = load_input_molecule(complex_pdb, None, selection, [], False)
    assert set(mol.chain) == chains
    assert set(mol.get("resname", sel="not protein")) == {"MOL", "CA"}
    # Metadata consumes an assigned topology, as it does after preparation.
    mol.guessBonds()
    stats = compute_complex_stats_for_metadata(mol, complex_pdb, 0, "test")
    assert stats["num_atoms_ligands"] == len(mol.get("index", sel="resname MOL"))
    assert stats["num_atoms_others"] == len(mol.get("index", sel="resname CA"))
    assert sum(stats[f"num_atoms_{c}"] for c in ("protein", "ligands", "others")) == mol.numAtoms


def test_empty_selection(complex_pdb):
    with pytest.raises(ValueError, match="No atoms remain"):
        load_input_molecule(complex_pdb, None, "chain Q", [], False)


@pytest.mark.parametrize("first_chain,second_chain", [("A", "B"), ("A", ""), ("", "")])
def test_multichain_protein_preparation(peptide, first_chain, second_chain):
    peptide.chain[:] = first_chain
    second = peptide.copy()
    second.chain[:] = second_chain
    second.coords[:, 0, :] += 50
    peptide.append(second)
    prepared, details = prepare_protein(peptide, 7.45, reset_protonation=True)
    assert set(prepared.chain) == {"A", "B"}
    assert len(details) == 6
    assert assign_pka_charges(prepared, details, 7.45)[0] == 0
    assert get_mol_prepared_charges(prepared) == 0
    assert get_protein_terminal_charge(prepared) == 0
    assert len(build_noh_embedding(prepared)[1]) == np.sum(prepared.element == "H")
    # Preparation must not rename chains on the original input.
    assert set(peptide.chain) == {first_chain, second_chain}


def test_residue_charge_identities():
    mol = Molecule().empty(4)
    mol.resname[:] = "LYS"
    mol.resid[:] = 1
    mol.chain[:] = ["A", "B", "B", "B"]
    mol.insertion[:] = ["", "", "A", "A"]
    mol.segid[:] = ["P1", "P1", "P1", "P2"]
    assert get_mol_prepared_charges(mol) == 4


@pytest.mark.parametrize("nh,oh,charge", [(3, 0, 0), (2, 0, -1), (3, 1, 1), (2, 1, 0)])
def test_terminal_charges(nh, oh, charge):
    # Separate backbone amine and carboxyl groups, with independently neutral
    # or charged termini. The calculation must not assume they cancel.
    mol = Molecule().empty(4 + nh + oh)
    mol.name[:4] = ["N", "CA", "C", "OXT"]
    mol.element[:4] = ["N", "C", "C", "O"]
    mol.element[4:] = "H"
    bonds = [(0, 1), (2, 3)] + [(0, i) for i in range(4, 4 + nh)]
    if oh:
        bonds.append((3, mol.numAtoms - 1))
    mol.bonds = np.array(bonds, dtype=np.uint32)
    assert get_protein_terminal_charge(mol) == charge


def test_generic_ligands_and_duplicate_residue_numbers(small_ligand):
    mol = small_ligand
    with pytest.raises(ValueError, match="No protonation template"):
        prepare_nonprotein_residue(mol, 7.45, {})
    second = mol.copy()
    second.chain[:] = "B"
    second.coords[:, 0, :] += 50
    mol.append(second)
    prepared, _, charge = prepare_nonprotein(mol, None, 7.45, {"MOL": "CC[NH3+]"})
    assert charge == 2
    assert set(prepared.chain) == {"A", "B"}
    stats = compute_complex_stats_for_metadata(prepared, "input.pdb", charge, "test")
    assert stats["num_atoms_protein"] == 0
    assert stats["num_atoms_ligands"] == prepared.numAtoms


@pytest.mark.parametrize("selection,keep_water,protein,ligand,other", [
    ("chain A and protein", False, True, False, False),
    ("chain A", False, True, True, True),
    ("chain A and resname CA", False, False, False, True),
    ("chain A and (resname CA or water)", True, False, False, True),
])
def test_cli_exports(complex_pdb, tmp_path, selection, keep_water, protein, ligand, other):
    subprocess.run([
        sys.executable, str(ROOT / "prepare_noh_system.py"),
        "--pdb", str(complex_pdb), "--ph", "7.45", "--ligand-smiles", "MOL=CC[NH3+]",
        "--filter-str", selection, "--sys-id", "test", "--output-root", str(tmp_path),
    ] + (["--keep-water"] if keep_water else []), check=True, capture_output=True, text=True)
    output = tmp_path / "test"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["filter_str"] == selection
    stats = metadata["complex_stats"]
    for component, present in [("protein", protein), ("ligands", ligand), ("others", other)]:
        assert bool(stats[f"num_atoms_{component}"]) == present
    noh = Molecule(str(output / "noh.pdb"))
    topology = CharmmPsfFile(str(output / "noh.psf")).topology
    assert topology.getNumAtoms() == noh.numAtoms
    assert noh.numAtoms == len(np.load(output / "noh-embedding.npy")) == stats["num_noh_atoms_total"]
    assert np.load(output / "noh-charge.npy").sum() == np.load(output / "total-charge.npy")[0]
    assert metadata["nonprotein_charge"] == (3 if ligand else 2 if other else 0)
