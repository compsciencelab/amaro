import os 
os.environ["NUMEXPR_MAX_THREADS"] = "18"
os.environ["OMP_NUM_THREADS"] = "18"

import h5py 
import math 
import shutil
import logging
import tempfile
from glob import glob
from tqdm import tqdm
import concurrent.futures
from os.path import join as opj
from utils import AggForce, ComputationScheduler


logger = logging.getLogger('noh-builder')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('error.log')
fh.setLevel(logging.ERROR)
logger.addHandler(fh)

def readPDBs(pdbList):
    if isinstance(pdbList, list):
        return pdbList
    pdblist = []
    with open(pdbList, "r") as f:
        for line in f:
            pdblist.append(line.strip())
    return sorted(pdblist)

class Payload:
    def __init__(self, scheduler, data_dir, input_dir, output_dir='.'):
        self.scheduler = scheduler
        self.data_dir = data_dir
        self.input_dir = input_dir  
        self.output_dir = output_dir
        
    def runComputation(self, batch_idx):
        logger.info(f"Batch {batch_idx} started")
        run(self.scheduler, batch_idx, self.data_dir, self.input_dir, self.output_dir)
    
  
def run(scheduler, batch_idx, data_dir, input_dir, output_dir='.'):
    """Extract information from the mdCATH dataset and write them to a h5 file per batch"""
    pdb_idxs = scheduler.process(batch_idx)
    desc = pdb_idxs[0] if len(pdb_idxs) == 1 else "reading PDBs"
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        for pdb in tqdm(pdb_idxs, total=len(pdb_idxs), desc=desc):
            with h5py.File(tmp_file, "w") as h5:
                resfile = opj(output_dir, f"mdcath_noh_dataset_{pdb}.h5")    
                if os.path.exists(resfile):
                    logger.error(f"SKIPPING {pdb}: File {resfile} already exists | batch {batch_idx}")
                    continue            
                h5_file = opj(data_dir, f"mdcath_dataset_{pdb}.h5")
                if not os.path.exists(h5_file):
                    logger.error(f"H5 NOT FOUND: File {h5_file} does not exist | batch {batch_idx}")
                    continue
                
                psfpath = glob(opj(input_dir, pdb, "*/structure.psf"))[0]
                pdbpath = glob(opj(input_dir, pdb, "*/structure.pdb"))[0]
                if not os.path.exists(psfpath) or not os.path.exists(pdbpath):
                    logger.error(f"SKIPPING {pdb}: {psfpath} or {pdbpath} does not exist | batch {batch_idx}")
                    continue
                
                group = h5.create_group(pdb)
                h5.attrs["layout"] = "mdcath-dataset-noh"
                with h5py.File(h5_file, "r") as origin:
                    
                    z = origin[pdb]['z'][:]
                    aggregator = AggForce(pdbpath, psfpath, z)
                    group.create_dataset('z', data=aggregator.emb)
                    
                    group.attrs['numResidues'] = origin[pdb].attrs['numResidues']
                    group.attrs['numProteinAtoms'] = origin[pdb].attrs['numProteinAtoms']
                    group.attrs['numChains'] = origin[pdb].attrs['numChains']
                    group.attrs['numNoHAtoms'] = len([el for el in z if el != 1])
                    availample_temps = ['320', '348', '379', '413', '450']
                    
                    for temp in availample_temps:
                        temp_group = group.create_group(temp)
                        for replica in origin[pdb][temp]:
                            repl_group = temp_group.create_group(replica)
                            if 'numFrames' not in origin[pdb][temp][replica].attrs.keys():
                                logger.error(f"numFrames not found in {pdb} {temp} {replica}")
                                del repl_group
                                continue
                                
                            repl_group.attrs['numFrames'] = origin[pdb][temp][replica].attrs['numFrames']
                            try:
                                c = aggregator.process_coords(origin[pdb][temp][replica]['coords'][:])
                                f = aggregator.process_forces(origin[pdb][temp][replica]['forces'][:])
                                assert c.shape== f.shape, f"Forces and Coords hapes do not match {c.shape} vs {f.shape}"
                            except (AssertionError, IndexError) as e:
                                logger.error(f"SKIPPING {pdb} {temp} {replica} {e} | batch {batch_idx}")
                                del repl_group
                                continue
                            repl_group.create_dataset('coords', data=c)
                            repl_group.create_dataset('forces', data=f)
                            repl_group['coords'].attrs['unit'] = 'Angstrom'
                            repl_group['forces'].attrs['unit'] = 'kcal/mol/Angstrom'
                                                        
        shutil.copyfile(tmp_file, resfile)

                
            
def launch():
    data_dir = "/PATH/TO/MDCATH/DATASET/"
    input_dir = "PATH/TO/GPUGRID/INPUTS/DIR"
    output_dir = "/PATH/TO/OUTPUT/DIR"
    pdb_list_file = "/PATH/TO/PDBLIST or LIST OF PDBS"
    pdb_list = readPDBs(pdb_list_file)
    
    batch_size = 1
    toRunBatches = None
    startBatch = None   
    max_workers = 24
    # Get a number of batches
    numBatches = int(math.ceil(len(pdb_list) / batch_size))
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of total batches: {numBatches}")

    if toRunBatches is not None and startBatch is not None:
        numBatches = toRunBatches + startBatch
    elif toRunBatches is not None:
        numBatches = toRunBatches
    elif startBatch is not None:
        pass
    
    # Initialize the parallelization system
    scheduler = ComputationScheduler(batch_size, startBatch, numBatches, pdb_list)
    toRunBatches = scheduler.getBatches()
    logger.info(f"numBatches to run: {len(toRunBatches)}")

    payload = Payload(scheduler, data_dir, input_dir, output_dir)

    error_domains = open("errors_noh_build.txt", "w")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(payload.runComputation, batch): batch for batch in toRunBatches}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(toRunBatches)):
            batch = future_to_batch[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                error_domains.write(f"Batch {batch} failed with exception: {e}\n")
                # Optionally, log the error and continue with the next computation

    return results


if __name__ == "__main__":
    launch()