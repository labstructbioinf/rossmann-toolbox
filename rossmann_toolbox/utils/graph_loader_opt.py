import copy
import random
from packaging import version

import torch
import dgl
import numpy as np
import pandas as pd
import networkx as nx

from .bio_params import LABEL_DICT, ACIDS_MAP_DEF, SS_MAP_EXT, CM_THRESHOLD


def collate(samples):
    (graphs, labels) = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

class GraphECMLoaderBalanced(torch.utils.data.Dataset):
    needed_columns = {'seq', 'secondary'}
    threshold = CM_THRESHOLD
    FLOAT_DTYPE = np.float32
    counter = 0
    res_info = {}; graphs = {}; edge_features = {}; node_features = {}; labels_encoded = {}
    def __init__(self, frame, contact_maps, edge_features, foldx_info, balance_classes=False, **kw_args):
        '''
        params:
        frame (pd.DataFrame) with columns: seq, alnpositions, simplified_cofactor(can be none)
        ss (np.ndarray) with locations of ss
        adjecency_matrix (np.ndarray) regular adjecency matrix used in defining graph structure
        device (str) cpu default
        cofactor (str) 
        use_ohe_features (bool) tells if add one hot encoding residue to node features
        add_epsilon (False, or float) if float adds add_epsilon value to empty alignment field  
        '''
        columns = frame.columns.tolist()
        assert not (self.needed_columns - set(columns)), f'no column(s) {self.needed_columns - set(columns)}'
        assert isinstance(frame, pd.DataFrame), 'frame should be DataFrame'
        assert isinstance(contact_maps, dict)
        assert isinstance(edge_features, dict)
        assert isinstance(foldx_info, dict)
        assert frame.shape[0] != 0, 'given empty frame'
        available_indices = list(foldx_info.keys())
        if 'simplified_cofactor' not in frame.columns.tolist():
            frame['simplified_cofactor'] = 'NAD'
        self.balance_classes = balance_classes
        self.indices = []
        frame = frame[frame.index.isin(available_indices)]
        for i, (idx, row) in enumerate(frame.iterrows()):            
            seq_emb = np.asarray([ACIDS_MAP_DEF[s] for s in row['seq']], dtype=np.int64)
            sec_emb = np.asarray([SS_MAP_EXT[s] for s in row['secondary']], dtype=np.int64)
            self.res_info[idx] = (seq_emb, sec_emb)
            self.graphs[idx] =  contact_maps[idx][0] < self.threshold         
            edge_dist_based = np.nan_to_num(edge_features[idx])
            edge_foldx_based = foldx_info[idx]['edge_data']

            self.edge_features[idx] = np.concatenate([edge_dist_based, edge_foldx_based], axis=1).astype(self.FLOAT_DTYPE)
            self.node_features[idx] = foldx_info[idx]['node_data'].astype(self.FLOAT_DTYPE)
            #print(self.node_features[idx], foldx_info)
            self.labels_encoded[idx] = LABEL_DICT[row['simplified_cofactor']]
            self.indices.append(idx)
        self.NAD_indices, self.non_NAD_indices = [], []
        for idx, cof in self.labels_encoded.items():
            if cof == 0:
                self.NAD_indices.append(idx)
            else:
                self.non_NAD_indices.append(idx)
        #self._validate_dicts()
        self._map_new_()
        #self._fix_samples_balancing_()
        self.num_samples_per_epoch = len(self.indices)
        if self.num_samples_per_epoch == 0:
            print(len(self.labels_encoded))
            print(len(self.NAD_indices), len(self.non_NAD_indices))
            raise ValueError('zero length loader')
        
    def __len__(self):
        return self.num_samples_per_epoch
    
    def __getitem__(self, idx):

        idx_m = self.index_map[idx]
        features_edge = self.edge_features[idx_m]
        features_node = self.node_features[idx_m]
        g = dgl.from_networkx(nx.Graph(self.graphs[idx_m]))
        seq_res_nb, sec_res_nb = self.res_info[idx_m]
        if self.balance_classes:
            if self.counter == self.num_samples_per_epoch:
                self._fix_samples_balancing_()
            else:
                self.counter += 1
        g.ndata['residues'] = torch.from_numpy(seq_res_nb)
        g.ndata['secondary'] = torch.from_numpy(sec_res_nb)
        g.edata['features'] = torch.from_numpy(features_edge)
        g.ndata['features'] = torch.from_numpy(features_node)
        return g, self.labels_encoded[idx_m]
    
    def _fix_samples_balancing_(self):
        if self.balance_classes:
            NAD_subsample = len(self.NAD_indices)
            NAD_subsample = int(self.SUBSAMPLE*len(self.NAD_indices))
            random.shuffle(self.NAD_indices)

            NAD_subsamples = self.NAD_indices[:NAD_subsample]
            self.indices = NAD_subsamples + self.non_NAD_indices
            random.shuffle(self.indices)
            self.counter = 0
        else:
            self.indices = list(self.res_info.keys())
        self._map_new_()
        
    def _map_new_(self):
        self.index_map = {num : idx for num, idx in enumerate(self.indices)}
        
    def _validate_dicts(self):
        pass
        '''
        for idx in self.adjecency_matrix.keys():
            if self.embeddings[idx].shape[0] != self.res_info[idx][0].size:
                raise ValueError(f'shape mismatch for idx {idx} between emb and res_info')
        '''