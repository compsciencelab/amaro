#!/usr/bin/env python3

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["HTMD_NONINTERACTIVE"] = "1"

import numpy as np
from moleculekit.tools.preparation import systemPrepare
from moleculekit.molecule import Molecule
from moleculekit.tools.autosegment import autoSegment

from noh_utils import COMPONENT_SELECTIONS, NOH_MAP_DICT, build_noh_embedding, get_constraints


# Charged side-chain states only; other supported states are neutral. This is
# used by residue-states mode and to check pKa charges against systemPrepare.
STATE_CHARGES = {
    "LYS": +1, "ARG": +1, "HIP": +1, "HSP": +1,
    "ASP": -1, "GLU": -1, "CYM": -1, "TYM": -1,
}

# Per-group chemistry; pKa VALUES are predicted separately for each residue.
# At pH == pKa choose the less protonated state (a deterministic 50/50 tie).
PKA_GROUPS = {
    "ASP": "acid",
    "GLU": "acid",
    "CYS": "acid",
    "TYR": "acid",
    "HIS": "base",
    "LYS": "base",
    "ARG": "base",
}
NEUTRAL_RESIDUES = {"ALA", "ASN", "GLN", "GLY", "ILE", "LEU", "MET",
                    "PHE", "PRO", "SER", "THR", "TRP", "VAL", "CYX"}

# Only isolated monatomic ions, never atoms within a ligand. Co/Cu need an
# explicit PDB formal charge (or a supplied topology) to select oxidation state.
ION_CHARGES = {
    "Na": +1, "K": +1, "Ca": +2, "Mg": +2, "Zn": +2,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
    "Co": None, "Cu": None,
}
assert set(ION_CHARGES).issubset(NOH_MAP_DICT)

# Aqueous benzamidinium pKa; this selects a dominant state, not a binding-site
# pKa prediction. Source: "Solvation-Enhanced Salt Bridges", JACS 146 (2024).
BENZAMIDINIUM_PKA = 11.6

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "noh-data"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare AMARO NOH reference data from a structure. The script computes the "
            "protein charge from HTMD systemPrepare and writes noh.pdb, noh.psf, "
            "noh-embedding.npy, and noh-charge.npy into noh-data/<sys_id>."
        )
    )
    parser.add_argument("--pdb", required=True, help="Input PDB file.")
    parser.add_argument(
        "--filter-str",
        default="all",
        help=("MoleculeKit atom selection applied once before preparation, for example "
              "'protein' or 'chain X Y Z'. Defaults to all chains and components. "
              "Select whole residues; excluding waters is controlled by --keep-water."),
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="Optional topology file, for example PSF or PRMTOP.",
    )
    parser.add_argument(
        "--sys-id",
        default=None,
        help="System identifier used for the output directory. Defaults to the PDB stem.",
    )
    parser.add_argument(
        "--ph",
        type=float,
        required=True,
        help="pH for protein preparation and the built-in BEN protonation rule.",
    )
    parser.add_argument(
        "--protein-charge-method",
        choices=("pka", "residue-states"),
        default="pka",
        help=("Default pka: reset protein input states, predict per-residue pKas, "
              "select states at --ph, and check state/charge agreement before export. "
              "residue-states: legacy state-name charge accounting."),
    )
    parser.add_argument(
        "--reset-protein-protonation",
        action="store_true",
        help=("Remove protein hydrogens and normalize protonation-specific residue "
              "names before predicting states. Automatic with the pka method; "
              "optional with residue-states. May change histidine tautomers."),
    )
    parser.add_argument(
        "--ligand-smiles",
        action="append",
        default=[],
        metavar="RESNAME=SMILES", # controls the placeholder shown in `--help`
        help=(
            "Ligand template with the intended protonation and formal charge at --ph. "
            "Repeat for multiple residue names. Overrides the built-in BEN rule. "
            "Cannot be combined with --topology, which preserves supplied states."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where noh-data/<sys_id> will be created.",
    )
    parser.add_argument(
        "--keep-water",
        action="store_true",
        help="Keep waters in the NOH export. By default, waters are removed.",
    )
    parser.add_argument(
        "--exclude-resnames",
        default="",
        help=(
            "Comma-separated residue names to remove before preparation and final "
            "NOH export, for example P0PC,CLA,SOD."
        ),
    )
    return parser.parse_args()


def parse_exclude_resnames(exclude_resnames):
    parsed = []
    for raw_resname in exclude_resnames.split(","):
        resname = raw_resname.strip().upper()
        if not resname:
            continue
        parsed.append(resname)
        if len(resname) > 4:
            parsed.append(resname[:4])
    return list(dict.fromkeys(parsed))


def load_input_molecule(pdb_path, topology_path, filter_str, exclude_resnames, keep_water):
    """Apply input selections once, before splitting protein from other residues."""
    mol = Molecule(str(pdb_path))
    if topology_path:
        mol.read(str(topology_path))
    selected = mol.atomselect(filter_str)
    if exclude_resnames:
        selected &= ~mol.atomselect("resname " + " ".join(exclude_resnames), guessBonds=False)
    if not keep_water:
        selected &= ~mol.atomselect("water", guessBonds=False)
    mol.filter(selected, guessBonds=False)
    if mol.numAtoms == 0:
        raise ValueError("No atoms remain after --filter-str, residue exclusions, and water filtering.")
    print(f"Selected input: {mol.numAtoms} atoms; chains {sorted(set(mol.chain))}")
    return mol


def get_mol_prepared_charges(mol_p):
    """Legacy state-name charge accounting; fresh pKa mode uses assign_pka_charges."""
    _, residues = mol_p.getResidues(fields=("segid", "chain", "resid", "insertion"))
    return float(sum(STATE_CHARGES.get(mol_p.resname[idx[0]], 0) for idx in residues))


def charge_from_pka(group, pka, ph_value):
    """Integer charge of the dominant fixed state, not an ensemble average."""
    if group not in PKA_GROUPS or not np.isfinite(pka) or not np.isfinite(ph_value):
        raise ValueError(f"Cannot assign pKa charge for {group}: pKa={pka}, pH={ph_value}")
    protonated = ph_value < pka
    return (0 if protonated else -1) if PKA_GROUPS[group] == "acid" else int(protonated)


def assign_pka_charges(mol, details, ph_value):
    """Use systemPrepare's fresh pKas and check its selected residue states.

    systemPrepare owns protonation and hydrogen placement. This function only
    assigns integer side-chain charges and rejects inconsistent/missing results.
    Terminal charges are counted separately from the prepared bonds.
    """
    report = details.copy()
    charges, sources = [], []
    for row in report.itertuples(index=False):
        group, state = str(row.resname), str(row.protonation)
        if state == "CYX":  # Disulfides are not titratable thiols.
            q, source = 0, "disulfide_not_titratable"
        elif group in PKA_GROUPS:
            q = charge_from_pka(group, float(row.pKa), ph_value)
            source = "predicted_pka_vs_ph"
        elif group in NEUTRAL_RESIDUES:
            q, source = 0, "non_titratable"
        else:
            raise ValueError(f"No pKa charge model for protein residue {group} {row.resid}")
        if q != STATE_CHARGES.get(state, 0):
            raise ValueError(
                f"pKa/state mismatch at {row.chain}:{row.resid}{row.insertion}: "
                f"pH/pKa requires charge {q:+d}, systemPrepare selected {state}. "
                "No simulation inputs exported."
            )
        charges.append(q)
        sources.append(source)
    # The table carries full residue identities, so insertion codes remain distinct.
    fields = ["segid", "chain", "resid", "insertion"]
    actual = {tuple(key): str(state) for *key, state in zip(
        mol.segid, mol.chain, mol.resid, mol.insertion, mol.resname)}
    reported = {tuple(getattr(row, f) for f in fields): str(row.protonation)
                for row in report.itertuples(index=False)}
    if len(reported) != len(report) or actual != reported:
        raise ValueError("Protonation table does not match the prepared protein residues")
    report["predictor_pka_raw"] = report["pKa"]
    report.loc[report["protonation"] == "CYX", "pKa"] = np.nan
    report["state_sidechain_charge"] = charges
    report["charge_assignment_source"] = sources
    return float(sum(charges)), report


def get_protein_terminal_charge(mol):
    """Count standard charged termini, including internal chain breaks and Pro.

    A protonated backbone N has four single bonds. A terminal OXT without
    a bonded hydrogen carries -1; a protonated OXT is neutral. Capped and
    cyclic peptides have neither charged terminal group.
    """
    degree = np.bincount(mol.bonds.ravel(), minlength=mol.numAtoms)
    constraints = get_constraints(mol)
    protonated = np.zeros(mol.numAtoms, dtype=bool)
    protonated[constraints[:, 0]] = True
    nterm = (mol.name == "N") & (mol.element == "N") & (degree == 4)
    cterm = (mol.name == "OXT") & (mol.element == "O") & ~protonated
    return float(nterm.sum() - cterm.sum())


def compute_complex_stats_for_metadata(mol, pdb_path, fallback_total_charge, charge_source):
    """Summarize a prepared molecule using its assigned bonds for selections."""
    pdb_path = Path(pdb_path)
    elements = np.asarray(mol.element, dtype=object)
    noh_mask = mol.atomselect("noh", guessBonds=False)
    unique_elements = sorted({str(element) for element in elements.tolist() if element})

    stats = {
        "system": pdb_path.parent.name,
        "complex_name": pdb_path.stem.removesuffix("_aa"),
        "total_charge": float(fallback_total_charge),
        "num_atoms_total": int(elements.shape[0]),
        "num_noh_atoms_total": int(np.sum(noh_mask)),
        "unique_elements": ",".join(unique_elements),
        "num_bonds_total": len(mol.bonds),
        "chains": sorted(set(mol.chain)),
        "charge_source": charge_source,
    }
    for component, selection in COMPONENT_SELECTIONS.items():
        mask = mol.atomselect(selection, guessBonds=False)
        stats[f"num_atoms_{component}"] = int(mask.sum())
        stats[f"num_noh_atoms_{component}"] = int((mask & noh_mask).sum())
    return stats


def prepare_protein(input_mol, ph_value, reset_protonation=False):
    """Prepare all selected protein chains; non-protein components stay separate."""
    mol = input_mol.copy()
    mol.filter("protein")
    if mol.numAtoms == 0:
        return mol, None

    mol.box = None
    mol.boxangles = None
    mol.time = None
    mol.step = None
    # Fill only missing chain IDs; keep every existing chain unchanged.
    if np.any(mol.chain == ""):
        mol = autoSegment(mol, sel=mol.chain == "", fields=("chain",))
    if reset_protonation:
        mol.filter("noh", guessBonds=False)
        # Preserve CYX disulfide identities; reset titratable side-chain variants.
        for old, new in {
            "HIP": "HIS", "HID": "HIS", "HIE": "HIS",
            "HSP": "HIS", "HSD": "HIS", "HSE": "HIS",
            "GLH": "GLU", "ASH": "ASP", "LYN": "LYS",
            "CYM": "CYS", "TYM": "TYR", "AR0": "ARG",
        }.items():
            mol.resname[mol.resname == old] = new
    num_hydrogens = np.sum(mol.element == "H")
    print(f"Preparing molecule for charge computation: {mol.numAtoms} atoms, {num_hydrogens} hydrogens, pH={ph_value}")
    mol_p, details = systemPrepare(
        mol,
        pH=ph_value,
        verbose=0,
        return_details=True,
        hydrophobic_thickness=None,
        ignore_ns=True,
    )
    print(f"Prepared molecule: {mol_p.numAtoms} atoms, {np.sum(mol_p.element == 'H')} hydrogens.")
    # This MoleculeKit version does not export PDB2PQR bonds.
    if len(mol_p.bonds) == 0:
        mol_p.guessBonds()
        keep_nearest_hydrogen_bonds(mol_p)
    return mol_p, details


def write_protonation_report(details, ph_value, output_dir):
    """Keep the predictor's evidence separate from discrete charge bookkeeping."""
    report = details.copy()
    report["preparation_ph"] = ph_value
    if "state_sidechain_charge" not in report:
        report["state_sidechain_charge"] = report["protonation"].map(
            lambda state: STATE_CHARGES.get(state, 0)
        )
        report["charge_assignment_source"] = "legacy_residue_state_name"
    if "pKa" not in report:
        report["pKa"] = np.nan
    report["pka_available"] = np.isfinite(report["pKa"])
    report["ph_minus_pka"] = ph_value - report["pKa"]
    report["near_ph"] = report["pka_available"] & (report["ph_minus_pka"].abs() < 1)
    report["pka_status"] = np.where(
        report["pka_available"], "predicted",
        "not_returned; do_not_infer_a_pka_from_the_selected_state",
    )
    report.loc[report["protonation"] == "CYX", "pka_status"] = "not_applicable_disulfide"
    report.to_csv(output_dir / "protein-protonation.csv", index=False)
    records = json.loads(report.to_json(orient="records"))
    (output_dir / "protein-protonation.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Protein protonation/pKa report: {output_dir / 'protein-protonation.csv'}")
    near = report[report["near_ph"]]
    if not near.empty:
        print("Residues with a reported pKa within 1 pH unit:")
        print(near[["chain", "resid", "insertion", "protonation", "pKa"]].to_string(index=False))
    ionizable = {"HIS", "HID", "HIE", "HIP", "ASP", "ASH", "GLU", "GLH",
                 "LYS", "LYN", "ARG", "AR0", "TYR", "TYM", "CYS", "CYM"}
    missing = report[report["resname"].isin(ionizable) & ~report["pka_available"]
                     & (report["protonation"] != "CYX")]
    if not missing.empty:
        warnings.warn(
            f"No pKa returned for {len(missing)} ionizable protein residues. "
            "Review protein-protonation.csv; existing variant names can preserve "
            "input states. Use --reset-protein-protonation only when a fresh "
            "prediction is intended."
        )
    return {"csv": "protein-protonation.csv", "json": "protein-protonation.json",
            "residues": len(records), "ionizable_without_pka": len(missing),
            "near_ph": int(report["near_ph"].sum()),
            "charge_note": "Report charges cover side chains. Standard terminal charges are counted separately from prepared bonds and included in protein_charge."}


def keep_nearest_hydrogen_bonds(mol):
    """Resolve ambiguous guessed H bonds; never apply this to supplied topology."""
    bonds = mol.bonds
    if bonds.size == 0:
        return

    element = mol.element
    coords = mol.coords[:, :, mol.frame]
    heavy_heavy_bonds = []
    hydrogen_candidates = {}

    for bond in bonds:
        i, j = int(bond[0]), int(bond[1])
        if i < 0 or j < 0 or i >= mol.numAtoms or j >= mol.numAtoms:
            continue

        ei, ej = element[i], element[j]
        if ei != "H" and ej != "H":
            heavy_heavy_bonds.append((i, j))
            continue
        if ei == "H" and ej == "H":
            continue

        heavy_idx, hydrogen_idx = (j, i) if ei == "H" else (i, j)
        dist2 = float(np.sum((coords[heavy_idx] - coords[hydrogen_idx]) ** 2))
        best = hydrogen_candidates.get(hydrogen_idx)
        if best is None or dist2 < best[0]:
            hydrogen_candidates[hydrogen_idx] = (dist2, heavy_idx)

    hydrogen_bonds = sorted(
        (heavy_idx, hydrogen_idx)
        for hydrogen_idx, (_, heavy_idx) in hydrogen_candidates.items()
    )
    new_bonds = np.array(heavy_heavy_bonds + hydrogen_bonds, dtype=np.int64).reshape(-1, 2)
    mol.bonds = new_bonds.astype(bonds.dtype, copy=False)
    if hasattr(mol, "bondtype"):
        mol.bondtype = np.array(["un"] * len(new_bonds), dtype=object)


def parse_ligand_smiles(entries):
    templates = {}
    for entry in entries:
        resname, sep, smiles = entry.partition("=")
        if not sep or not resname.strip() or not smiles.strip():
            raise ValueError("--ligand-smiles must have the form RESNAME=SMILES")
        templates[resname.strip().upper()] = smiles.strip()
    return templates


def prepare_nonprotein_residue(residue, ph_value, ligand_smiles, sel="all"):
    """Prepare one PDB residue, without treating metal coordination as bonds."""
    from rdkit import Chem

    residue = residue.copy(sel=sel)
    resname = str(residue.resname[0]).upper()
    if residue.numAtoms == 1 and residue.element[0] in ION_CHARGES:
        element = residue.element[0]
        explicit_charge = int(residue.formalcharge[0])
        charge = explicit_charge or ION_CHARGES[element]
        if charge is None:
            raise ValueError(
                f"Ion {resname} ({element}) needs an explicit PDB formal charge "
                "or --topology to specify its oxidation state."
            )
        residue.formalcharge[:] = charge
        residue.charge[:] = charge
        return residue

    smiles = ligand_smiles.get(resname)
    if smiles is None and resname == "BEN":
        if abs(ph_value - BENZAMIDINIUM_PKA) < 1:
            warnings.warn("pH is near the BEN pKa; check the intended protonation state.")
        smiles = "NC(=[NH2+])c1ccccc1" if ph_value < BENZAMIDINIUM_PKA else "NC(=N)c1ccccc1"
    if smiles is None and np.all(residue.atomselect("water", guessBonds=False)):
        smiles = "O"
    if smiles is None:
        raise ValueError(
            f"No protonation template for non-protein residue {resname}. "
            "Supply --ligand-smiles 'RESNAME=SMILES' with the desired protonation "
            "state, or a prepared hydrogenated PDB with --topology."
        )

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES template for {resname}: {smiles}")
    template = Chem.RemoveHs(template)
    residue.filter("noh", guessBonds=False)
    if sorted(residue.element) != sorted(a.GetSymbol() for a in template.GetAtoms()):
        raise ValueError(f"Heavy atoms of {resname} do not match its SMILES template.")
    # Guess connectivity within this residue only; PDB CONECT lacks bond orders.
    # A one-atom template (e.g. water) has no bonds to guess before adding Hs.
    if residue.numAtoms == 1:
        template.AddConformer(Chem.Conformer(1))
        template.GetConformer().SetAtomPosition(0, residue.coords[0, :, 0].tolist())
        prepared = Molecule.fromRDKitMol(Chem.AddHs(template, addCoords=True))
        for field in ("resname", "resid", "chain", "segid", "insertion"):
            getattr(prepared, field)[:] = getattr(residue, field)[0]
        prepared.name[0] = residue.name[0]
        residue = prepared
    else:
        residue.templateResidueFromSmiles(
            "all", smiles, addHs=True, guessBonds=True, _logger=False,
        )
    heavy = residue.element != "H"
    heavy_bonds = np.all(heavy[residue.bonds], axis=1)
    if heavy_bonds.sum() != template.GetNumBonds():
        raise ValueError(f"Connectivity of {resname} does not match its SMILES template.")
    # These are formal charges for total-charge bookkeeping, not force-field
    # partial charges. The templated molecule's .charge is otherwise all zeros.
    residue.charge[:] = residue.formalcharge
    return residue


def prepare_nonprotein(input_mol, topology_path, ph_value, ligand_smiles):
    """Prepare selected non-protein residues, or retain their supplied topology."""
    mol = input_mol.copy()
    mol.filter("not protein")
    source = f"topology:{Path(topology_path).name}" if topology_path else "input_pdb"
    if mol.numAtoms == 0:
        return None, source, 0.0

    if not topology_path:
        # Residue identities include chain and insertion code; resid alone is
        # not unique. Splitting also removes metal coordination CONECT records.
        _, residues = mol.getResidues(fields=("segid", "chain", "resid", "insertion", "resname"))
        prepared = Molecule()
        for indices in residues:
            prepared.append(prepare_nonprotein_residue(mol, ph_value, ligand_smiles, sel=indices))
        mol = prepared
        source += "+ligand_templates_and_ion_formal_charges"
    elif not np.all(np.isfinite(mol.charge)):
        raise ValueError("Non-protein topology contains non-finite charges.")

    return mol, source, float(mol.charge.sum())


def write_outputs(mol, noh_emb, total_charge, output_dir, metadata):
    noh_mol = mol.copy()
    noh_mol.filter("noh", guessBonds=False)
    if noh_mol.numAtoms == 0:
        raise ValueError("NOH filtering produced an empty structure.")

    noh_charge = np.zeros(noh_emb.shape[0], dtype=np.float32)
    noh_charge[0] = np.float32(total_charge)
    total_charge_array = np.array([total_charge], dtype=np.float32)

    noh_pdb_path = output_dir / "noh.pdb"
    noh_psf_path = output_dir / "noh.psf"
    noh_embedding_path = output_dir / "noh-embedding.npy"
    noh_charge_path = output_dir / "noh-charge.npy"
    total_charge_path = output_dir / "total-charge.npy"
    metadata_path = output_dir / "metadata.json"

    noh_mol.write(str(noh_pdb_path))
    noh_mol.write(str(noh_psf_path))
    np.save(noh_embedding_path, noh_emb)
    np.save(noh_charge_path, noh_charge)
    np.save(total_charge_path, total_charge_array)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return {
        "noh_pdb": noh_pdb_path,
        "noh_psf": noh_psf_path,
        "noh_embedding": noh_embedding_path,
        "noh_charge": noh_charge_path,
        "total_charge": total_charge_path,
        "metadata": metadata_path,
    }


def main():
    args = parse_args()
    ligand_smiles = parse_ligand_smiles(args.ligand_smiles)
    if args.topology and ligand_smiles:
        raise ValueError("--ligand-smiles cannot be combined with --topology.")
    if not np.isfinite(args.ph):
        raise ValueError("--ph must be finite.")

    pdb_path = Path(args.pdb).resolve()
    topology_path = Path(args.topology).resolve() if args.topology else None
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if topology_path and not topology_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topology_path}")

    sys_id = args.sys_id or pdb_path.stem
    output_dir = Path(args.output_root).resolve() / sys_id
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_resnames = parse_exclude_resnames(args.exclude_resnames)
    input_mol = load_input_molecule(
        pdb_path, topology_path, args.filter_str, exclude_resnames, args.keep_water,
    )
    reset_protonation = args.reset_protein_protonation or args.protein_charge_method == "pka"
    ref_mol, protonation_details = prepare_protein(
        input_mol, args.ph, reset_protonation=reset_protonation,
    )
    protein_charge, terminal_charge = 0.0, 0.0
    protonation_report = None
    topology_source = ""
    if protonation_details is not None:
        if args.protein_charge_method == "pka":
            protein_charge, protonation_details = assign_pka_charges(ref_mol, protonation_details, args.ph)
        else:
            protein_charge = get_mol_prepared_charges(ref_mol)
        terminal_charge = get_protein_terminal_charge(ref_mol)
        protein_charge += terminal_charge
        protonation_report = write_protonation_report(protonation_details, args.ph, output_dir)
        topology_source = "systemPrepare"

    extra_mol, extra_source, nonprotein_charge = prepare_nonprotein(
        input_mol, topology_path, args.ph, ligand_smiles,
    )
    if extra_mol is not None:
        ref_mol.append(extra_mol)
        topology_source = f"{topology_source};nonprotein:{extra_source}".lstrip(";")
    total_charge = float(protein_charge + nonprotein_charge)
    charge_sources = []
    if protonation_details is not None:
        charge_sources.append("per_residue_predicted_pka_vs_ph"
                              if args.protein_charge_method == "pka" else "systemPrepare_protein_residue_states")
        charge_sources.append("prepared_protein_termini")
    if extra_mol is not None:
        charge_sources.append("topology_nonprotein_charges" if topology_path
                              else "nonprotein_template_and_ion_formal_charges")
    charge_source = "+".join(charge_sources)

    _, constraints, _, noh_emb = build_noh_embedding(ref_mol)
    if constraints.shape[0] != np.sum(ref_mol.element == "H"):
        raise ValueError(
            "Every prepared hydrogen must be bonded to exactly one heavy atom. "
            "Check the supplied topology or guessed protein bonds."
        )

    metadata = {
        "sys_id": sys_id,
        "input_pdb": str(pdb_path),
        "input_topology": str(topology_path) if topology_path else None,
        "filter_str": args.filter_str,
        "exclude_resnames": exclude_resnames,
        "topology_source": topology_source,
        "ph": args.ph,
        "reset_protein_protonation": reset_protonation,
        "protein_charge_method": args.protein_charge_method,
        "protein_pka_scope": "protein_only; ligand_and_ions_prepared_separately",
        "pka_tie_rule": "pH_equal_pKa_selects_less_protonated_state",
        "protein_protonation_report": protonation_report,
        "ligand_smiles": ligand_smiles,
        "benzamidinium_pka": (BENZAMIDINIUM_PKA if not topology_path
                              and "BEN" in input_mol.resname and "BEN" not in ligand_smiles else None),
        "protein_charge": float(protein_charge),
        "protein_terminal_charge": terminal_charge,
        "nonprotein_charge": float(nonprotein_charge),
        "total_charge": float(total_charge),
        "charge_source": charge_source,
        "charge_file_semantics": {
            "noh-charge.npy": "Compatibility NOH charge vector with total charge stored at index 0 and zeros elsewhere.",
            "total-charge.npy": "Preferred scalar charge file for charge-conditioned TensorNet2 Coulomb models.",
        },
        "num_constraints": int(constraints.shape[0]),
        "keep_water": bool(args.keep_water),
        "complex_stats": compute_complex_stats_for_metadata(
            mol=ref_mol,
            pdb_path=pdb_path,
            fallback_total_charge=total_charge,
            charge_source=charge_source,
        ),
    }
    outputs = write_outputs(ref_mol, noh_emb, total_charge, output_dir, metadata)

    print(f"System ID: {sys_id}")
    print(f"pH: {args.ph}")
    print(f"Computed protein charge: {protein_charge}")
    print(f"Computed nonprotein charge: {nonprotein_charge}")
    print(f"Computed total charge: {total_charge}")
    print(f"Topology source for NOH export: {topology_source}")
    print(f"NOH beads: {noh_emb.shape[0]}")
    print(f"Constraint count: {constraints.shape[0]}")
    print(f"Output directory: {output_dir}")
    print(f"Saved NOH PDB: {outputs['noh_pdb']}")
    print(f"Saved NOH PSF: {outputs['noh_psf']}")
    print(f"Saved NOH embedding: {outputs['noh_embedding']}")
    print(f"Saved NOH charge: {outputs['noh_charge']}")
    print(f"Saved total charge: {outputs['total_charge']}")
    print(f"Saved metadata: {outputs['metadata']}")


if __name__ == "__main__":
    main()
