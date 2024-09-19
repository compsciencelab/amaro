import csv
import json 
import numpy as np
from tqdm import tqdm
from os.path import join as opj
from torch.utils.data import Subset
from collections import defaultdict
from torchmdnet.datasets import MDCATH
from torchmdnet.utils import make_splits
from torch_geometric.loader import DataLoader
from torchmdnet.models.model import load_model


def get_atomtype_errors(errors):
    """ Get the error for atomtypes """
    
    nohcgmap = {"CH0": 1, "CH1": 2, "CH2": 3, "CH3": 4,
                "NH0": 5, "NH1": 6, "NH2": 7, "NH3": 8,
                "OH0": 9, "OH1": 10, "SH0": 11, "SH1": 12,
            } 
    
    atomtype_errors = defaultdict(list)
    for atomtype, idx in nohcgmap.items():
        atomtype_errors[atomtype] = np.abs(errors["f_predicted"] - errors["f_refs"])[errors["z"] == idx].mean()
    
    return atomtype_errors

if __name__ == '__main__':
    
    # define paths and model 
    ckpt = "/PATH/TO/MODEL.ckpt"
    error_log = "error.log"
    data_dir = "/PATH/TO/NOHMDCATH/DIR"
    outdir= '.'
    device = 'cuda'
    
    ######################### DATASET #########################
    dataset = MDCATH(root=data_dir,
                    numAtoms=None, 
                    numNoHAtoms=None, 
                    numResidues=None, 
                    source_file="mdcath_noh_source.h5",
                    file_basename = "mdcath_noh_dataset",
                    pdb_list="test_doms.txt",
                    temperatures=['320', '348', '379', '413', '450'], 
                    skip_frames=1, 
                    solid_ss=50
    )    
    print(f'Loaded {len(dataset)} samples')
    
    ######################### GET THE SPLITS #########################
    
    splits = {}
    splits["idx_train"], splits["idx_val"], splits["idx_test"] = make_splits(len(dataset),
                                                                            train_size=None,
                                                                            val_size=10,
                                                                            test_size=5000,
                                                                            seed=42,
                                                                            splits=None,
                                                                            filename=None,
    )
    print(f'Splits: {len(splits["idx_train"])} train, {len(splits["idx_val"])} val, {len(splits["idx_test"])} test')
    test_dataset = Subset(dataset, splits["idx_test"])
    
    ######################### INFERENCE #########################
    errors = defaultdict(list)
    model = load_model(
            ckpt,
            derivative=True,
            static_shapes=True,
            check_errors=False,
            )
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.to(device)
    model.eval()
    for b in tqdm(DataLoader(test_dataset, batch_size=1, num_workers=2, pin_memory=True, persistent_workers=False), desc="Inference"):
        
        # model forward 
        y, neg_dy = model.forward(
            b.z.to(device),
            b.pos.to(device),
            b.batch.to(device),
        )
    
        y, neg_dy = y.to("cpu").detach(), neg_dy.to("cpu").detach()
        errors["z"].append(b.z.to("cpu").numpy())
        errors["f_refs"].append(b.neg_dy.to("cpu").numpy())
        errors["f_predicted"].append(neg_dy.numpy())
        errors['num_atoms'].append(b.z.shape[0])
        errors['domain_error'].append(np.abs(neg_dy - b.neg_dy).mean().numpy())
        errors['info'].append(list(b.info))
    
    # build array of errors
    for prop in ["z", "num_atoms", "f_refs", "f_predicted",'domain_error', 'info']:
            if prop in ['num_atoms', 'domain_error']:
                errors[prop] = np.array(errors[prop])
            else:
                errors[prop] = np.concatenate(errors[prop], axis=0)

    # Save the errors to a json file
    json_dict = {}
    with open(opj(outdir, error_log), "w") as f:
        for prop in errors.keys():
            if prop == 'info':
                json_dict[prop] = list(errors[prop])
                continue
            json_dict[prop] = errors[prop].flatten().tolist()
        json.dump(json_dict, f, indent=4)
    
    
    f_mae = np.abs(errors["f_predicted"] - errors["f_refs"]).mean()
    print(f'Mean Absolute Error (all): {f_mae:.4f}')
    
    ######################### ATOMTYPE ERRORS #########################
    atomtype_errors = get_atomtype_errors(errors)
    
    with open(opj(outdir, "atomtype_errors.csv"), "w") as f:
        writer = csv.writer(f)
        
        # Writing header
        headers = ["Model", "Mean Absolute Error (all)"] + list(atomtype_errors.keys())
        writer.writerow(headers)
        
        # Writing data rows
        row = ["Model", f'{f_mae:.4f}']
        row += [f'{atomtype_errors[atomtype]:.4f}' for atomtype in atomtype_errors.keys()]
        writer.writerow(row)