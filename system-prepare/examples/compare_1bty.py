#!/usr/bin/env python3
"""Compare the prepared 1BTY examples using NumPy.

Run: python compare_1bty.py [left_directory right_directory]
Rows show differences only; delta charge is right minus left.
"""

import argparse
from io import BytesIO
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noh_utils import NOH_MAP_DICT

BEAD_TYPES = {value: name for name, value in NOH_MAP_DICT.items()}


def load_array(path):
    data = path.read_bytes()
    # These PlayMoleculeAI files have a UTF-8 replacement character in place
    # of NumPy's initial 0x93 byte. Correct only that known prefix in memory.
    if data.startswith(b"\xef\xbf\xbdNUMPY"):
        warnings.warn(f"{path}: damaged NumPy magic prefix; correcting in memory", stacklevel=2)
        data = b"\x93" + data[3:]
    return np.load(BytesIO(data), allow_pickle=False)


def load(directory):
    """Use PDB identity rather than array index to align the two preparations."""
    atoms = {}
    for line in (directory / "noh.pdb").read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key = (line[21:22].strip(), line[22:27].strip(), line[12:16].strip())
        if key in atoms:
            raise ValueError(f"{directory}: duplicate bead identity {key}")
        atoms[key] = line[17:20].strip()

    charge = load_array(directory / "noh-charge.npy")
    embedding = load_array(directory / "noh-embedding.npy")
    for name, values in (("charge", charge), ("embedding", embedding)):
        if values.shape != (len(atoms),):
            raise ValueError(f"{directory}: {name} shape {values.shape}, expected {(len(atoms),)}")
    if not np.all(np.isin(embedding, list(BEAD_TYPES))):
        raise ValueError(f"{directory}: unknown embedding values")
    total = load_array(directory / "total-charge.npy")
    if total.size != 1:
        raise ValueError(f"{directory}: expected one total charge")
    return atoms, charge, embedding, float(total.item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("left", nargs="?", type=Path, default=here / "1BTY_HoloSet")
    parser.add_argument("right", nargs="?", type=Path, default=here / "1BTY_PlayMoleculeAI")
    parser.add_argument("--atol", type=float, default=1e-6, help="absolute charge tolerance (default: 1e-6)")
    args = parser.parse_args()
    if args.atol < 0:
        parser.error("--atol must be nonnegative")
    left, ql, el, tl = load(args.left)
    right, qr, er, tr = load(args.right)
    il = {key: i for i, key in enumerate(left)}
    ir = {key: i for i, key in enumerate(right)}
    common = [key for key in left if key in right]
    li = np.array([il[key] for key in common], dtype=int)
    ri = np.array([ir[key] for key in common], dtype=int)
    charge_diff = ~np.isclose(ql[li], qr[ri], atol=args.atol, rtol=0)
    emb_diff = el[li] != er[ri]
    residue_diff = np.array([left[key] != right[key] for key in common], dtype=bool)

    print(f"Left:  {args.left} ({len(left)} beads)")
    print(f"Right: {args.right} ({len(right)} beads)")
    print(f"Total charge: {tl:.9g} -> {tr:.9g} e; delta = {tr - tl:+.9g} e")
    print("noh-charge.npy is a compatibility vector, not per-bead physical partial charges.")
    print(f"Matched beads: {len(common)}; left only: {len(left) - len(common)}; "
          f"right only: {len(right) - len(common)}")
    print(f"Matched differences: charge={np.count_nonzero(charge_diff)}, "
          f"embedding/bead-type={np.count_nonzero(emb_diff)}, "
          f"residue-name={np.count_nonzero(residue_diff)}")

    print("\nBead-type counts (embedding):       left  right  delta")
    for emb in np.union1d(el, er):
        nl, nr = np.count_nonzero(el == emb), np.count_nonzero(er == emb)
        print(f"{BEAD_TYPES[emb]:>4} ({int(emb):2d})                      {nl:5d} {nr:6d} {nr - nl:+6d}")

    print("\nDifferent or unmatched beads; indices are 1-based PDB row positions.")
    print(f"{'chain:resid:atom':<20} {'index L/R':<13} {'resname L/R':<13} "
          f"{'type(emb) L/R':<24} {'charge L/R':<23} {'delta q':>10}")

    def side(key, atoms, indices, charge, embedding):
        if key not in indices:
            return "-", "-", "-", "-"
        i = indices[key]
        emb = int(embedding[i])
        return str(i + 1), atoms[key], f"{BEAD_TYPES[emb]}({emb})", f"{charge[i]:.9g}"

    changed = {key for key, different in zip(common, charge_diff | emb_diff | residue_diff) if different}
    rows = 0
    for key in list(left) + [key for key in right if key not in left]:
        if key in il and key in ir and key not in changed:
            continue
        a = side(key, left, il, ql, el)
        b = side(key, right, ir, qr, er)
        delta = f"{qr[ir[key]] - ql[il[key]]:+.6g}" if key in il and key in ir else "-"
        label = ":".join(part or "_" for part in key)
        print(f"{label:<20} {a[0] + '/' + b[0]:<13} {a[1] + '/' + b[1]:<13} "
              f"{a[2] + '/' + b[2]:<24} {a[3] + '/' + b[3]:<23} {delta:>10}")
        rows += 1
    if not rows:
        print("No differences.")


if __name__ == "__main__":
    main()
