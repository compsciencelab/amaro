import numpy as np
from glob import glob
from os.path import join as opj
from moleculekit.molecule import Molecule

# Map of noh atom types to embedding values
nohcgmap = {"CH0": 1, "CH1": 2, "CH2": 3, "CH3": 4,
            "NH0": 5, "NH1": 6, "NH2": 7, "NH3": 8,
            "OH0": 9, "OH1": 10, "SH0": 11, "SH1": 12,
        } 

if __name__ == '__main__':
    name = "cln"
    outdir = "."
    pdb = glob(f"../noh/{name}*.pdb")
    psf = glob(f"../noh/{name}*.psf")
    
    mol = Molecule(pdb) 
    mol.read(psf)
    
    emb = np.array([nohcgmap[i] for i in mol.atomtype])
    np.save(opj(outdir, f"{name}_noh_embeddings.npy"), emb)