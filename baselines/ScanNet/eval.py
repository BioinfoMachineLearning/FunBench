import subprocess
import os
from glob import glob
import pandas as pd

def cast_name(s):
    pdb, _, chain, *_ = s.split('_')
    pdbid = f"{pdb}_{chain}"
    return pdbid


def run_infer_ls(pdb_id_list, no_msa=True,predictions_folder='./predictions'):
    completed_pdbs = [i.split('_')[0]+"_"+i.split('_')[2].split('-')[1][0] for i in os.listdir(predictions_folder) if \
        len(glob(os.path.join(predictions_folder, i,"predictions_*.csv"))) > 0]
    for pdb_id in pdb_id_list:
        if pdb_id in completed_pdbs:
            continue
        #cmd: python predict_bindingsites.py 1brs_A  --noMSA 
        cmd = ['python', 'predict_bindingsites.py', pdb_id, '--predictions_folder',predictions_folder]
        if no_msa:
            cmd += ['--noMSA']
        subprocess.run(cmd)
        
def run_infer_path(pdb_id_path, no_msa=True,predictions_folder='./predictions'):
    with open(pdb_id_path, 'r') as f:
        pdb_id_list = f.readlines()
        pdb_id_list = list(set([pdb_id.strip() for pdb_id in pdb_id_list]))
    run_infer_ls(pdb_id_list, no_msa,predictions_folder)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_id_path', type=str, default='../../construct_testset/data/next_pdb_non_redundant_ids.txt')
    parser.add_argument('--no_msa', action='store_true')
    parser.add_argument('--predictions_folder',dest='predictions_folder',
                        default='./predictions',
                        help='Input name')
    args = parser.parse_args()
    run_infer_path(args.pdb_id_path, args.no_msa, args.predictions_folder)