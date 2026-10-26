import numpy as np
import pytest
from moleculekit.molecule import Molecule

from noh_utils import NOH_MAP_DICT, build_noh_embedding
from prepare_noh_system import (
    ION_CHARGES,
    prepare_nonprotein,
    parse_ligand_smiles,
    prepare_nonprotein_residue,
)


@pytest.fixture
def benzamidine():
    # Heavy-atom coordinates from 3PTB BEN A:1, without hydrogen/bond-order data.
    mol = Molecule()
    mol.empty(9, numFrames=1)
    mol.record[:] = "HETATM"
    mol.resname[:] = "BEN"
    mol.resid[:] = 1
    mol.chain[:] = "A"
    mol.element[:] = ["C"] * 7 + ["N", "N"]
    mol.name[:] = ["C1", "C2", "C3", "C4", "C5", "C6", "C", "N1", "N2"]
    mol.coords[:, :, 0] = [
        [-1.853, 14.311, 16.658], [-2.107, 15.653, 16.758],
        [-1.774, 16.341, 17.932], [-1.175, 15.662, 19.005],
        [-0.914, 14.295, 18.885], [-1.257, 13.634, 17.708],
        [-2.193, 13.627, 15.496], [-2.797, 14.235, 14.491],
        [-1.762, 12.391, 15.309],
    ]
    return mol


@pytest.mark.parametrize("ph,charge,nitrogen_beads", [(7.45, 1, [7, 7]), (13, 0, [6, 7])])
def test_benzamidine_states(benzamidine, ph, charge, nitrogen_beads):
    original_coords = benzamidine.coords.copy()
    mol = prepare_nonprotein_residue(benzamidine, ph, {})
    beads = build_noh_embedding(mol)[-1]
    assert sorted(beads[-2:]) == nitrogen_beads
    assert mol.charge.sum() == charge
    assert np.sum(mol.element == "H") == 8 + charge
    np.testing.assert_array_equal(mol.coords[:9], original_coords)
    np.testing.assert_array_equal(mol.name[:9], benzamidine.name)
    assert set(mol.resname) == {"BEN"}


def test_explicit_state_and_repreparation(benzamidine):
    mol = prepare_nonprotein_residue(benzamidine, 7.45, {"BEN": "NC(=N)c1ccccc1"})
    assert mol.charge.sum() == 0
    mol = prepare_nonprotein_residue(mol, 7.45, {})
    assert mol.charge.sum() == 1
    assert list(build_noh_embedding(mol)[-1][-2:]) == [7, 7]


@pytest.mark.parametrize("element,charge", [(e, c) for e, c in ION_CHARGES.items() if c is not None])
def test_supported_ions(element, charge):
    mol = Molecule()
    mol.empty(1, numFrames=1)
    mol.element[:] = element
    mol.resname[:] = element.upper()
    mol = prepare_nonprotein_residue(mol, 7.45, {})
    assert mol.charge[0] == charge
    assert list(build_noh_embedding(mol)[-1]) == [NOH_MAP_DICT[element]]


def test_variable_oxidation_and_water():
    mol = Molecule()
    mol.empty(1, numFrames=1)
    mol.element[:] = "Cu"
    mol.resname[:] = "CU"
    with pytest.raises(ValueError, match="oxidation state"):
        prepare_nonprotein_residue(mol, 7.45, {})
    mol.formalcharge[:] = 1
    assert prepare_nonprotein_residue(mol, 7.45, {}).charge[0] == 1
    mol.element[:] = "O"
    mol.resname[:] = "HOH"
    mol.formalcharge[:] = 0
    mol = prepare_nonprotein_residue(mol, 7.45, {})
    assert mol.numAtoms == 3
    assert mol.charge.sum() == 0
    assert list(build_noh_embedding(mol)[-1]) == [13]


def test_unknown_and_mismatched_templates(benzamidine):
    benzamidine.resname[:] = "LIG"
    with pytest.raises(ValueError, match="No protonation template"):
        prepare_nonprotein_residue(benzamidine, 7.45, {})
    with pytest.raises(ValueError, match="Heavy atoms"):
        prepare_nonprotein_residue(benzamidine, 7.45, {"LIG": "CC"})
    assert parse_ligand_smiles(["BEN=NC(=[NH2+])c1ccccc1"])["BEN"] == "NC(=[NH2+])c1ccccc1"


def test_supplied_topology_preserved(benzamidine, tmp_path):
    mol = prepare_nonprotein_residue(benzamidine, 7.45, {})
    mol.charge[:] = 0
    mol.charge[7:9] = [0.4, 0.6]
    pdb, psf = tmp_path / "prepared.pdb", tmp_path / "prepared.psf"
    mol.write(str(pdb))
    mol.write(str(psf))
    input_mol = Molecule(str(pdb))
    input_mol.read(str(psf))
    result, source, charge = prepare_nonprotein(input_mol, psf, 13, {})
    np.testing.assert_allclose(result.charge, mol.charge)
    assert charge == pytest.approx(1)
    assert result.numAtoms == mol.numAtoms
    assert list(build_noh_embedding(result)[-1][-2:]) == [7, 7]
    assert source.startswith("topology:")


@pytest.mark.parametrize("element,num_hydrogens", [("F", 1), ("S", 2)])
def test_unsupported_bead_species(element, num_hydrogens):
    mol = Molecule().empty(1 + num_hydrogens, numFrames=1)
    mol.element[0] = element
    mol.element[1:] = "H"
    mol.bonds = np.array([(0, i) for i in range(1, mol.numAtoms)], dtype=np.uint32)
    with pytest.raises(ValueError, match=f"{element}H{num_hydrogens}"):
        build_noh_embedding(mol)
