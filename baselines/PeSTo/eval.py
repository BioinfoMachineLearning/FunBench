import os
import json
import numpy as np
import pandas as pd
import torch as pt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data_encoding import categ_to_resnames
from src.logger import Logger
from src.scoring import bc_scoring, nanmean
from model.config import config_data, config_model, config_runtime
from model.data_handler import Dataset, collate_batch_data
from src.dataset import select_by_sid, select_by_max_ba, select_by_interface_types
from model.model import Model

def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step):
    X, ids_topk, q, M, y = [data.to(device) for data in batch_data]
    z = model.forward(X, ids_topk, q, M)

    pos_ratios += (pt.mean(y, dim=0).detach() - pos_ratios) / (1.0 + np.sqrt(global_step))
    criterion.pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
    dloss = criterion(z, y)

    loss_factors = (pos_ratios / pt.sum(pos_ratios)).reshape(1, -1)
    losses = (loss_factors * dloss) / dloss.shape[0]

    return losses, y.detach(), pt.sigmoid(z).detach()

def compute_scores(y, p, device=pt.device('cpu')):
    scores = [bc_scoring(y, p)]
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()
    index = pd.MultiIndex.from_product([
        ['acc', 'ppv', 'npv', 'tpr', 'tnr', 'mcc', 'auc', 'std'],
    ], names=['Metric'])
    columns = ['Protein', 'NA', 'ion', 'Ligand', 'Lipid']
    df = pd.DataFrame(m_scores, index=index, columns=columns)
    return df

def load_dataset(dataset_path, l_types, r_types, max_ba, max_size, min_num_res):
    dataset = Dataset(dataset_path)
    dataset.set_types(l_types, r_types)
    
    m = dataset.m
    m &= select_by_max_ba(dataset, max_ba)
    m &= (dataset.sizes[:,0] <= max_size)
    m &= (dataset.sizes[:,1] >= min_num_res)
    m &= select_by_interface_types(dataset, l_types, np.concatenate(r_types))
    
    dataset.update_mask(m)
    dataset.set_types(l_types, r_types)
    
    return dataset

def load_model(model_dir, device):
    model_filepath = os.path.join(model_dir, 'model_ckpt.pt')
    model = Model(config_model)
    model.load_state_dict(pt.load(model_filepath, map_location=device))
    model = model.eval().to(device)
    return model

def main(dataset_path, model_dir, result_csv_path, device='cuda'):
    l_types = categ_to_resnames['protein']
    r_types = [
        categ_to_resnames['protein'],
        categ_to_resnames['dna'] + categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ]
    max_ba = 1
    max_size = 1024 * 8
    min_num_res = 48

    dataset = load_dataset(dataset_path, l_types, r_types, max_ba, max_size, min_num_res)
    model = load_model(model_dir, device)

    df_ls = []
    id_idxs = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        pdb_id, X, ids_topk, q, M, y = [i.to(device) if isinstance(i, pt.Tensor) else i for i in data]
        z = model.forward(X, ids_topk, q, M)
        y, p = y.detach(), pt.sigmoid(z).detach()
        score_df = compute_scores(y.cpu(), p.cpu())
        df_ls.append(score_df)
        id_idxs.append(pdb_id)
        if len(df_ls) >= 2:
            new_df = pd.concat(df_ls, keys=id_idxs, names=['PDBID', 'Metric'])
            new_df.to_csv(result_csv_path)

if __name__ == '__main__':
    dataset_path = '../../construct_testset/data/contacts_rr5A_64nn_8192_wat.h5'
    model_dir = "model/save/i_v4_1_2021-09-07_11-21"
    result_csv_path = "./eval_result/metircs.csv"
    main(dataset_path, model_dir, result_csv_path)
