"""Benchmarking script for tmdnet model inference. This script utilizes `torch.compile` to compile 
a specific model and records the time (measured in milliseconds) required to complete both a
forward and a backward pass for a designated molecule."""

import os
import yaml
import time
import torch
import numpy as np
from glob import glob
from tabulate import tabulate
from moleculekit.molecule import Molecule
from torchmdnet.calculators import External
from moleculekit.periodictable import periodictable
from torchmdnet.models.model import create_model

# ingnore user warnings
import warnings
warnings.filterwarnings("ignore")

def build_args(**kwargs):
    base_yaml = "../train/amaro_tmdnet.yaml"
    with open(base_yaml, "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, val in kwargs.items():
        assert key in args, f"Broken test! Unknown key '{key}'."
        args[key] = val
    
    return args
    

class GpuTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # synchronize CUDA device
        self.end = time.perf_counter()
        self.interval = (self.end - self.start) * 1000  # Convert to milliseconds

def benchmark_pdb(pdb_file, **kwargs):
    device = "cuda"
    torch.cuda.empty_cache() # clear cuda cache

    molecule = Molecule(pdb_file)
    atomic_numbers = torch.tensor(
        [periodictable[symbol].number for symbol in molecule.element],
        dtype=torch.long,
        device=device,
    )
    positions = torch.tensor(
        molecule.coords[:, :, 0], dtype=torch.float32, device=device
    ).to(device)
    molecule = None
    torch.cuda.nvtx.range_push("Initialization") 
    
    args = build_args(derivative=True,
                      max_z=int(atomic_numbers.max() + 1),
                      check_errors=False,
                      static_shapes=True,
                      **kwargs)
    
    model = create_model(args)
    numparams = sum(p.numel() for p in model.parameters())
    z = atomic_numbers
    pos = positions
    batch = torch.zeros_like(z).to("cuda")
    model = model.to("cuda")
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Warmup")
    for i in range(3):
        energy, force = model(z, pos, batch)
    torch.cuda.synchronize()
    model = External(
        model,
        embeddings=z.unsqueeze(0),
        device="cuda",
        use_cuda_graph=True,
    )

    if not model.use_cuda_graph:
        print("Warning: CUDA graph not supported, disabling some features")
        model.model.representation_model.distance.check_errors = True
        model.model.representation_model.distance.resize_to_fit = True

    for i in range(10):
        energy, force = model.calculate(pos, None)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("Benchmark")
    nbench = 100
    times = np.zeros(nbench)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    if model.use_cuda_graph:
        with torch.no_grad():
            with GpuTimer() as timer:
                with torch.cuda.stream(stream):
                    for i in range(nbench):
                        energy, force = model.calculate(pos, None)
                torch.cuda.synchronize()
    else:
        with GpuTimer() as timer:
            with torch.cuda.stream(stream):
                for i in range(nbench):
                    energy, force = model.calculate(pos, None)
            torch.cuda.synchronize()

    return len(atomic_numbers), numparams, timer.interval / nbench


def benchmark_all(to_benchmark, pdbs):
    column_labels = []
    results = {}
    for pdb_file in pdbs:
        pdb_name = os.path.basename(pdb_file)
        molecule = Molecule(pdb_file)
        natoms = len(molecule.element)
        print("Found %s, with %d atoms" % (os.path.basename(pdb_file), natoms))

        # results structure: {case: {'numParameters':numparams, 'pdb_numAtoms': time}}
        num_atoms = 0
        for name, kwargs in to_benchmark.items():
            torch.cuda.empty_cache()
            num_atoms, numparams, time = benchmark_pdb(pdb_file, **kwargs)
            pdb_label = pdb_name.split(".")[0] + '_' + str(num_atoms)
            if pdb_label not in column_labels:
                column_labels.append(pdb_label)
            if name not in results.keys():
                results[name] = {'numParameters': numparams}
            results[name][pdb_label] = time
    
    # sort column labels by number of atoms
    column_labels = sorted(column_labels, key=lambda x: int(x.split('_')[2]))
    # Print results as table
    table_data = []
    for case, values in results.items():
        case_data = [case, values["numParameters"]]
        for pdb_numAtoms in column_labels:
            case_data.extend([values[pdb_numAtoms]])
        table_data.append(case_data)
    
    # define columns names
    columns = ["model", "num Parameters"]
    column_labels = [" ".join(x.split('_')[:2]) + ' (' + x.split('_')[2] + ')' for x in column_labels]                         
    columns.extend(column_labels)
    
    table = tabulate(
        table_data,
        headers=columns,
        tablefmt="pretty",
        showindex=False,
        stralign="center",
        numalign="center",
        colalign=("center",),
    )
    
    # Save table to file csv dataframe format
    with open("benchmark_results.csv", "w") as f:
        f.write(table)

if __name__ == "__main__":
    to_benchmark = {
        "AMARO": {
            "model": "tensornet",
            "embedding_dimension": 128,
            "max_num_neighbors": 64,
            "num_layers": 1,
            "num_rbf": 32,
            },
    }
    pdbs = glob("../data/ff_proteins/noh/*.pdb")[:1]
    print(f"Found {len(pdbs)} pdb files")
    benchmark_all(to_benchmark, pdbs)