import glob
import math
import pandas as pd
from scoring import bc_scoring, bc_score_names 
import numpy as np
import torch


def stack_csv(csv_list):
    dfs = [df for df in map(pd.read_csv, csv_list)]
    return pd.concat(dfs, ignore_index=True,axis=0)


def calculate_metrics(merged_df, pred_label, bc_score_names):
    metrics_list = []
    true_label = 'target_binding_labels'
    for i, row in merged_df.iterrows():
        pdbid = row['id']
        if pdbid in true_df['id'].values and row[pred_label] is not None:
            p = row[pred_label]
            y = true_df.loc[true_df['id'] == pdbid, true_label].values[0]
            if isinstance(y, torch.Tensor) and (len(y) == len(p)):
                metrics = bc_scoring(torch.tensor(y).unsqueeze(-1).int(), torch.tensor(p).unsqueeze(-1))
                metrics = metrics.squeeze(-1)
                metrics_list.append([pdbid] + metrics.tolist())

        metrics_df = pd.DataFrame(metrics_list, columns=['pdbid'] + bc_score_names)
        metrics_df.to_csv(f'{pred_label}_metrics.csv', index=False)

    
def main(pred_path, true_df,binding_type):
    """
    pred_dir has ID,Sequence,ZN_prob,ZN_pred,CA_prob,CA_pred,MG_prob,MG_pred,MN_prob,MN_pred
    true df has columns ['id', 'zn_label','ca_label','mg_label',mn_label']
    """
    pred_df = pd.read_csv(pred_path)
    
    pred_df['id'] = pred_df['ID'].apply(lambda x: "_".join(x.split('_')[:2]))
    pred_labels = ['ZN_pred', 'CA_pred', 'MG_pred', 'MN_pred']
    # construct a new column where the entriy is the numpy array stack of the 4 pred_labels of shape (N,4)
    for l in pred_labels:
        pred_df[l] = pred_df[l].apply(lambda x: np.array([float(i) for i in x.split(',')]))
    
    merged_df = pd.merge(pred_df, true_df, on='id')
    calculate_metrics(merged_df, binding_type, bc_score_names)

    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding_type", 
                        type=str,
                        help="on of ['ZN_pred', 'CA_pred', 'MG_pred', 'MN_pred']",
                        default="ZN_pred"
                        )    
    parser.add_argument('--true_df_h5_path', type=str, help='path to the true binding labels csv',
                        default='/home/zpw97/github-projs/PDBAnnotator/test_df_ZN_binding.h5')
    parser.add_argument('--pred_path', type=str, help='path to the directory of prediction csvs',default='predictions/ZN_binding_0_predictions.csv')
    args = parser.parse_args()
    true_df = pd.read_hdf(args.true_df_h5_path, key='df')
    main(args.pred_path, true_df,args.binding_type)

