import string, re
import numpy as np
import torch

MAX_INPUT_SEQ = 3000
ID_col = 'ID'
sequence_col = "Sequence"
metal_list = ["ZN", "CA", "MG", "MN"]
LMetalSite_threshold = [0.42, 0.34, 0.5, 0.47]

NN_config = {
    'feature_dim': 1024,
    'hidden_dim': 64,
    'num_encoder_layers': 2,
    'num_heads': 4,
    'augment_eps': 0.05,
    'dropout': 0.2,
}


class MetalDataset:
    def __init__(self, df, protein_features):
        self.df = df
        self.protein_features = protein_features
        self.feat_dim = NN_config['feature_dim']

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        seq_id = self.df.loc[idx, ID_col]
        protein_feat = self.protein_features[seq_id]
        return protein_feat

    def padding(self, batch, maxlen):
        batch_protein_feat = []
        batch_protein_mask = []
        for protein_feat in batch:
            padded_protein_feat = np.zeros((maxlen, self.feat_dim))
            padded_protein_feat[:protein_feat.shape[0]] = protein_feat
            padded_protein_feat = torch.tensor(padded_protein_feat, dtype = torch.float)
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = np.zeros(maxlen)
            protein_mask[:protein_feat.shape[0]] = 1
            protein_mask = torch.tensor(protein_mask, dtype = torch.long)
            batch_protein_mask.append(protein_mask)

        return torch.stack(batch_protein_feat), torch.stack(batch_protein_mask)

    def collate_fn(self, batch):
        maxlen = max([protein_feat.shape[0] for protein_feat in batch])
        batch_protein_feat, batch_protein_mask = self.padding(batch, maxlen)

        return batch_protein_feat, batch_protein_mask, maxlen

import re
from Bio import SeqIO

def process_fasta(fasta_file):
    """
    Process a FASTA file and extract sequence IDs and sequences.

    Parameters:
    fasta_file (str): Path to the input FASTA file.
    MAX_INPUT_SEQ (int): Maximum number of input sequences allowed.

    Returns:
    tuple: A tuple containing a list of IDs and a list of sequences.
           Returns -1 for inconsistent data or 1 if the number of sequences exceeds MAX_INPUT_SEQ.
    """
    ID_list = []
    seq_list = []

    try:
        with open(fasta_file, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                # Extract ID from the description
                description_items = record.description.split('|')
                ID = "_".join(description_items[:2]).replace(" ", "_")
                ID_list.append(ID)
                seq_list.append(str(record.seq).upper())

        if len(ID_list) == len(seq_list):
            return (ID_list, seq_list)
        else:
            return -1
    except FileNotFoundError:
        print(f"Error: File '{fasta_file}' not found.")
        return -1
