import h5py
import numpy as np
import torch as pt


from src.dataset import load_sparse_mask, collate_batch_features
from src.data_encoding import categ_to_resnames


def load_interface_labels(hgrp, t0, t1_l):
    # load stored data
    shape = tuple(hgrp.attrs['Y_shape'])
    ids = pt.from_numpy(np.array(hgrp['Y']).astype(np.int64))

    # matching contacts type for receptor and ligand
    y_ctc_r = pt.any((ids[:,2].view(-1,1) == t0), dim=1).view(-1,1)
    y_ctc_l = pt.stack([pt.any((ids[:,3].view(-1,1) == t1), dim=1) for t1 in t1_l], dim=1)
    y_ctc = (y_ctc_r & y_ctc_l)

    # save contacts of right type
    y = pt.zeros((shape[0], len(t1_l)), dtype=pt.bool)
    y[ids[:,0], pt.where(y_ctc)[1]] = True

    return y


def collate_batch_data(batch_data):
    # collate features
    X, ids_topk, q, M = collate_batch_features(batch_data)

    # collate targets
    y = pt.cat([data[4] for data in batch_data])

    return X, ids_topk, q, M, y


class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath, features_flags=(True, False, False)):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

        # selected features
        self.ftrs = [fn for fn, ff in zip(['qe','qr','qn'], features_flags) if ff]

        # preload data
        with h5py.File(dataset_filepath, 'r') as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/sizes"])
            self.ckeys = np.array(hf["metadata/ckeys"]).astype(np.dtype('U'))
            self.ctypes = np.array(hf["metadata/ctypes"])

            # load parameters to reconstruct data
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(np.dtype('U'))
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(np.dtype('U'))
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype('U'))
            self.mids = np.array(hf["metadata/mids"]).astype(np.dtype('U'))

        # set default selection mask
        self.m = np.ones(len(self.keys), dtype=bool)

        # prepare ckeys mapping
        self.__update_selection()

        # set default runtime selected interface types
        self.t0 = pt.arange(self.mids.shape[0])
        self.t1_l = [pt.arange(self.mids.shape[0])]

    def __update_selection(self):
        # ckeys mapping with keys
        self.ckeys_map = {}
        for key, ckey in zip(self.keys[self.m], self.ckeys[self.m]):
            if key in self.ckeys_map:
                self.ckeys_map[key].append(ckey)
            else:
                self.ckeys_map[key] = [ckey]

        # keep unique keys
        self.ukeys = list(self.ckeys_map)

    def update_mask(self, m):
        # update mask
        self.m &= m

        # update ckeys mapping
        self.__update_selection()

    def set_types(self, l_types, r_types_l):
        self.t0 = pt.from_numpy(np.where(np.isin(self.mids, l_types))[0])
        self.t1_l = [pt.from_numpy(np.where(np.isin(self.mids, r_types))[0]) for r_types in r_types_l]

    def get_largest(self):
        i = np.argmax(self.sizes[:,0] * self.m.astype(int))
        k = np.where(np.isin(self.ukeys, self.keys[i]))[0][0]
        return self[k]

    def __len__(self):
        return len(self.ukeys)

    def __getitem__(self, k):
        # get corresponding interface keys
        key = self.ukeys[k]
        ckeys = self.ckeys_map[key]

        # load data
        with h5py.File(self.dataset_filepath, 'r') as hf:
            # hdf5 group
            hgrp = hf['data/structures/'+key]

            # topology
            X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32)) # [num_atoms, 3]
            M = load_sparse_mask(hgrp, 'M') # [num_atoms, num_residues]
            ids_topk = pt.from_numpy(np.array(hgrp['ids_topk']).astype(np.int64)) # [num_atoms, num_closest_atoms]

            # features
            q_l = []
            for fn in self.ftrs:
                q_l.append(load_sparse_mask(hgrp, fn))
            q = pt.cat(q_l, dim=1)

            # interface labels
            y = pt.zeros((M.shape[1], len(self.t1_l)), dtype=pt.bool)
            for ckey in ckeys:
                y |= load_interface_labels(hf['data/contacts/'+ckey], self.t0, self.t1_l)

        return key, X, ids_topk, q, M, y.float()

import re

def cast_pdbname(input_name):
    _, pdb,__, chain = input_name.split('/')
    return f"{pdb}_{chain.split(':')[0]}"

if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    binder_types = ['protein', 'dna','rna', 'ion', 'ligand', 'lipid']
    metadata = pd.DataFrame(columns=['pdb_id']+binder_types)
    config_data = {
    'l_types': categ_to_resnames['protein'],
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna'],categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ],
    }
    dataset = Dataset('data/contacts_rr5A_64nn_8192_wat.h5') # construct_testset/
    dataset.set_types(config_data['l_types'], config_data['r_types'])
    for i in tqdm(range(len(dataset))):
        try:
            key, X, ids_topk, q, M, y = dataset[i]
            type_mask = pt.any(y,dim=0) # e.g., [True, False, False, False, False] for protein binding sites
            # Create a DataFrame for the current PDB ID and binder types
            pdb_metadata = pd.DataFrame({'pdb_id': [cast_pdbname(key)], **{k:v for k,v in zip(binder_types, type_mask.tolist())}})
            
            # Concatenate the current PDB metadata with the main metadata DataFrame
            metadata = pd.concat([metadata, pdb_metadata], ignore_index=True)
        except:
            pass
    # save metadata
    metadata.to_csv('data/metadata_binding_type.csv', index=False)