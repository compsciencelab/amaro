import os
import json
import pandas as pd
import urllib.request
from tqdm import tqdm
from Bio.pairwise2 import align
from os.path import join as opj
from collections import defaultdict


def load_pdb_list(pdb_list):
    """Load PDB list from a file or return list directly."""
    if isinstance(pdb_list, list):
        return pdb_list
    elif isinstance(pdb_list, str) and os.path.isfile(pdb_list):
        print(f"Reading PDB list from {pdb_list}")
        with open(pdb_list, "r") as file:
            return [line.strip() for line in file]
    raise ValueError("Invalid PDB list. Please provide a list or a path to a file.")

def get_json_from_url(domain, info_error_file):
    """ Get JSON data from CATH API for a given domain. """
    
    base_url = 'https://www.cathdb.info/version/v4_2_0/api/rest/domain_summary/'
    try:
        response_bytes = urllib.request.urlopen(opj(base_url, domain)).read()
    except urllib.error.HTTPError as e:
        # if the domain is not found, try with the new version of the API
        try:
            response_bytes = urllib.request.urlopen(opj(base_url.replace('v4_2_0', 'v4_3_0'), domain)).read()
        except urllib.error.HTTPError as e:
            info_error_file.write(f'Error for {domain}: {e}\n')
        return None
    
    # Decode the bytes into a string
    response_str = response_bytes.decode('utf-8')  # Assuming UTF-8 encoding

    # Parse the string into a JSON object or dictionary
    json_data = json.loads(response_str)

    return json_data

def get_seq_from_cath_api(cath_doms: list, error_log: str, output_file: str):    
    """ Get sequence data from CATH API for a list of domains, and return a DataFrame
    of domain sequences and lengths. """
    cath_seq = {}
    for record in tqdm(cath_doms, total=len(cath_doms), desc="mdCATH"):
        dom, length = record['domain'], record['length']
        data = get_json_from_url(dom, error_log)
        seq = data['data']['atom_sequence']
        if len(seq) != length:
            error_log.write(f"Length mismatch for {dom}: {length} vs {len(seq)}\n")
            continue
        cath_seq[dom] = (length, seq)
        
    mdcath_df = pd.DataFrame(cath_seq).T
    mdcath_df.columns = ['length', 'sequence']
    # save the sequence data to a csv file
    mdcath_df.to_csv(output_file)
    return mdcath_df  
        
if __name__ == '__main__':
    mdcath_doms_list = '/PATH/TO/DOMAIN/LIST/FILE.txt' # or a list of mdcath doms
    target_ff = ['cln', 'noPro-trpcage', 'villin', 'a3d']
    output_log = open('output.log', 'w')
    mdcath_doms = load_pdb_list(mdcath_doms_list)
    mdcath_seq_file = 'mdcath_seq.csv'
    
    if os.path.exists(mdcath_seq_file):
        mdcath_seq = pd.read_csv(mdcath_seq_file, index_col=0).to_dict(orient='index')
    else:
        mdcath_seq = get_seq_from_cath_api(mdcath_doms, output_log, mdcath_seq_file)
        
    
    ff_seqs = pd.read_csv('../data/ff_proteins/ff_fasta.csv', index_col=0).to_dict(orient='index')
    
    ############# sequence similarity ############
    seq_sims = defaultdict(dict)
    for ff_name in target_ff:
        for dom in tqdm(mdcath_doms, total=len(mdcath_doms), desc=f"SeqSim - {ff_name}"):
            if dom not in mdcath_seq:
                continue
            reference_seq = mdcath_seq[dom]['sequence']
            target_seq = ff_seqs[ff_name]['sequence']
            target_length = ff_seqs[ff_name]['length']
            
            alignments = align.localxs(target_seq, reference_seq, -1, -1)
            if len(alignments) == 0:
                continue
            best_alignment = alignments[0] # considering lexicographic alignment function
            aligned_T, aligned_R, score, start, end = best_alignment
            matching_residues = sum(1 for a, b in zip(aligned_T, aligned_R) if a == b)

            # Calculate sequence similarity
            seq_sim = (matching_residues / target_length) * 100
            seq_sims[ff_name][dom] = seq_sim

        df = pd.DataFrame(seq_sims[ff_name], index=[0]).T.reset_index()
        df.columns=['mdcath_dom', 'seq_sim']
        output_log.write(f'Max sequence similarity for {ff_name}: {df["seq_sim"].max()} \n')
    
    output_log.close()
    