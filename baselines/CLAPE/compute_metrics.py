import torch
import pandas as pd
import subprocess
import os
from glob import glob
import pandas as pd
from tqdm import tqdm


from scoring import bc_score_names, bc_scoring

def read_predictions(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    ids = lines[0::3]
    pred_labels_list = lines[2::3]
    # convert a string of int to an torch array
    pred_labels = [torch.tensor([int(i) for i in labels],dtype=float) for labels in pred_labels_list]
    # return a dataframe with columns 'id' and 'pred_label'
    df = pd.DataFrame({'id': ids, 'pred_binding_sites': pred_labels})
    return df

    
def load_true_labels(label_h5_path):
    """
    return a dataframe with columns ['id','true_binding_sites'], 
    where true_binding_sites is a list of binding sites for each type molecules.
    
    """
    true_df = pd.read_hdf(label_h5_path, key='df')
    true_df['target_binding_labels'] = true_df['target_binding_labels'].apply(lambda x: x.float()) # obtain the protein binding labels
    true_df.rename(columns={'target_binding_labels':'true_binding_sites'}, inplace=True)
    return true_df

def compute_metrics(label_h5_path, pred_res_path,output_csv):
    res_df = pd.DataFrame(columns=['id',*bc_score_names])
    pred_df = read_predictions(pred_res_path)
    true_df = load_true_labels(label_h5_path)
    # merge two dataframes
    df = pd.merge(true_df, pred_df, on='id')
    for i in tqdm(range(len(df))):
        pdbid = df['id'][i]
        pred_labels =  df['pred_binding_sites'][i]
        true_labels = df['true_binding_sites'][i]
        if true_labels.shape[0] != pred_labels.shape[0]:
            print(f"true and predicted label length {true_labels.shape[0]} != {pred_labels.shape[0]} for {pdbid}")
            continue
            
        metrics_tns = bc_scoring(true_labels.unsqueeze(-1),pred_labels.unsqueeze(-1))
        tmp_df = pd.DataFrame({
            'id':pdbid, 
            **dict(zip(bc_score_names, metrics_tns.tolist())),
            })
        res_df = pd.concat([res_df, tmp_df], axis=0)
    res_df.to_csv(output_csv, index=False)
    return res_df


    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path',dest='predictions_folder',
                        default='DNA_sites.txt',
                        help='Input name')
    parser.add_argument('--label_h5_path',default='/home/zpw97/github-projs/PDBAnnotator/test_df_dna_binding.h5',
                        help='the h5 file containing the true labels.')
    parser.add_argument('--output_csv',default='DNA_eval_results.csv', help='Output csv file')
    args = parser.parse_args()
    compute_metrics( args.label_h5_path, args.predictions_folder, args.output_csv)
