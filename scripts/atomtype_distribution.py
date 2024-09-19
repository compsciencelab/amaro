import os
import h5py
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

fsize = 20
plt.rcParams.update({'font.size': fsize, 
                'axes.labelsize': fsize, 
                'legend.fontsize': fsize-2, 
                'xtick.labelsize': fsize-2, 
                'ytick.labelsize': fsize-2, 
                'lines.linewidth': 2.5,
                'font.style': 'normal',
                })

nohcgmap = {"CH0": 1, "CH1": 2, "CH2": 3, "CH3": 4,
            "NH0": 5, "NH1": 6, "NH2": 7, "NH3": 8,
            "OH0": 9, "OH1": 10, "SH0": 11, "SH1": 12,
        } 

def get_atomtype_distribution(embedding, cgmap):
    return [cgmap[atom] for atom in embedding]

def readPDBs(pdbList):
    if isinstance(pdbList, list):
        return pdbList
    pdblist = []
    with open(pdbList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)


if __name__ == "__main__":
    data_dir = '/PATH/TO/NOHMDCATH/DIR'
    mdcath_dom_list = '/PATH/TO/DOMAIN/LIST/FILE.txt'
    
    output_dir = '.'
    rcgmap = {v:k for k,v in nohcgmap.items()}
    
    domains = readPDBs(mdcath_dom_list)
    atomtypes = []
    for d in tqdm(domains):
        h5file = os.path.join(data_dir, f'mdcath_noh_dataset_{d}.h5')
        with h5py.File(h5file, 'r') as f:
            embeddings = f[d]['z'][:]
            at_comp = get_atomtype_distribution(embeddings, rcgmap)
            atomtypes.extend(at_comp)

    atomtypes = np.array(atomtypes)
    plt.figure(figsize=(12, 8))
    N, bins, patches = plt.hist(atomtypes, bins=12, label=rcgmap.values(), ec='black')
    
    # annotate the value over the bars
    for i in range(len(N)):
        plt.text(bins[i], N[i], str(int(N[i])), fontsize=14, verticalalignment='bottom')
    plt.xlabel('Atom type')
    plt.ylabel('Counts')
    cmap = plt.get_cmap('tab20', len(rcgmap))

    for i in range(len(rcgmap)):
        patches[i].set_facecolor(cmap(i))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'atomtype_distribution.png'))
    