import subprocess
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import torch


from scoring import bc_score_names, bc_scoring

def cast_name(s):
    pdb, _, chain, *_ = s.split('_')
    chain = chain.split('-')[1][0]
    pdbid = f"{pdb}_{chain}"
    return pdbid


def load_true_labels(label_h5_path):
    """
    return a dataframe with columns ['id','true_binding_sites'], 
    where true_binding_sites is a list of binding sites for each type molecules.
    
    """
    true_df = pd.read_hdf(label_h5_path, key='df')
    # true_df = true_df[['id','binding_labels']]
    true_df['binding_labels'] = true_df['binding_labels'].apply(lambda x: x[:,0].float()) # obtain the protein binding labels
    # rename binding_sites to true_binding_sites
    true_df.rename(columns={'binding_labels':'true_binding_sites'}, inplace=True)
    return true_df

def load_pred_labels(pred_path):
    # if file is empty or not exist, return None
    if not os.path.exists(pred_path) or os.path.getsize(pred_path) == 0:
        print(f"{pred_path} is empty or not exist.")
        return None
    df = pd.read_csv(pred_path,sep='\t',header=None)
    return torch.tensor(df.iloc[:,-1].values)

def compute_metrics(label_h5_path, predictions_folder, output_csv):
    true_df = load_true_labels(label_h5_path)
    res_df = pd.DataFrame(columns=['id',*bc_score_names])
    for p in tqdm(glob(os.path.join(predictions_folder, "*"))):
        # obtain dir name of p
        pname = os.path.basename(p)
        res_csv = os.path.join(p,"prediction.txt")
        
        pred_labels = load_pred_labels(res_csv)
        if pred_labels is None:
            continue
        pdbid = pname
        true_df.isin([pdbid])
        if pdbid not in true_df['id'].values:
            continue
        true_labels = true_df[true_df['id']==pdbid]['true_binding_sites'].iloc[0]
        
        if true_labels.shape[0] != pred_labels.shape[0]:
            print(f"{true_labels.shape[0]} != {pred_labels.shape[0]} for {pdbid}")
            continue
        metrics_tns = bc_scoring(true_labels.unsqueeze(-1),pred_labels.unsqueeze(-1))

        tmp_df = pd.DataFrame({
            'id':pdbid, 
            **dict(zip(bc_score_names, metrics_tns.tolist())),
            })
        res_df = pd.concat([res_df, tmp_df], axis=0)
    print(f"Saving to {output_csv}")
    res_df.to_csv(output_csv, index=False)
    return res_df
    


    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_folder',dest='predictions_folder',
                        default='./predictions',
                        help='Input name')
    parser.add_argument('--label_h5_path',default='/home/zpw97/github-projs/PDBAnnotator/test_df_protein_binding.h5',
                        help='the h5 file containing the true labels.')
    parser.add_argument('--output_csv',default='./BSPred_metrics.csv',)
    args = parser.parse_args()
    compute_metrics(args.label_h5_path, args.predictions_folder, args.output_csv)