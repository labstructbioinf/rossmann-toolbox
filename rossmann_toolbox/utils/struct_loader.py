import os
import sys


import torch
import dgl
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..models import GatLit
from .graph_loader_opt import GraphECMLoaderBalanced


def collate(samples):
    '''
    dgl batch to gpu
    https://discuss.dgl.ai/t/best-way-to-send-batched-graphs-to-gpu/171/9
    '''
    (graphs, labels) = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def sigmoid(x):
    return 1/(1 + np.exp(-x))
       
class Deepligand3D:
    NUM_WORKERS = 4
    LABEL_DICT = {'NAD' : 0, 'NADP' : 1, 'SAM' : 2,  'FAD' : 3}
    LABEL_DICT_R = {0 :'NAD', 1 :'NADP', 2 :'SAM',  3:'FAD' }
    COFACTORS = list(LABEL_DICT.keys())
    CM_THRESHOLD = None # assigned when data loader is initialized
    def __init__(self, weights_dir, device='cpu', **kw_args):
        """
        Monte carlo Deepligand version
        params:
            path_model (str) path to model weights
            device (str) 'cpu' or 'cuda'
        """
        assert device  in ['cuda', 'cpu']
        self.device_type = device
        self.COFACTORS_STD = [f'{cof}_std' for cof in self.COFACTORS]
        self.CLASSES = len(self.COFACTORS)
        self.list_of_model_paths = [f'{weights_dir}/struct_ensemble/model{i}.ckpt' for i in range(1, 5)]
        self.list_of_hp_paths = [f'{weights_dir}/struct_ensemble//hp{i}.ckpt' for i in range(1, 5)]
        self.device = torch.device(device)

        self.model_list = torch.nn.ModuleList()

        for path, hp_path in zip(self.list_of_model_paths, self.list_of_hp_paths):
            conf = torch.load(path)
            hps = torch.load(hp_path)
            model = GatLit(hps)
            model.load_state_dict(conf)
            self.model_list.append(model.eval().to(self.device))

    def predict(self, dataframe, contact_maps, edge_feats, foldx_info, raw_scores = False, verbose=False):
        """
        params:
            dataframe (pd.DataFrame) with 'seq' and 'secondary' columns
            contact_maps (dict) similar as above
            edge_feats (dict) similar as above
            raw_scores (bool) if True probabilities are replaced with raw scores
            verbose (bool) default False
        returns:
            pd.DataFrame
        """
        available_sequences = foldx_info.keys() & contact_maps.keys() & edge_feats.keys()
        self.sequences = dataframe['seq'].tolist()
        indices = dataframe.index.tolist()
        num_sequences = len(self.sequences)
        available_sequences = available_sequences & set(indices)
        if verbose and (len(available_sequences) == 0):
            raise ValueError('mismatched keys')
        else:
            pass
            #print('seq to process:', len(available_sequences), num_sequences)
        indices_with_embeddings = [i for i, seq in enumerate(indices) if seq in available_sequences]
        mask_with_embeddings = [True if idx in indices_with_embeddings else False for idx in indices]
        sequences_without_embeddings = set(indices) - available_sequences
        if verbose and (len(sequences_without_embeddings) > 0):
            print(f'found {len(sequences_without_embeddings)} sequences without embeddings/contact maps')
            dataframe_no_missings = dataframe[~dataframe.index.isin(sequences_without_embeddings)].copy()
        else:
            dataframe_no_missings = dataframe.copy()

        loader = self._prepare_samples(dataframe_no_missings, contact_maps, edge_feats, foldx_info)

        single_preds = []
        for x, _ in loader:
            if self.device_type == 'cuda':
                x = x.to(self.device)
            storage = [model(x, scores=raw_scores).detach().unsqueeze(0) for model in self.model_list]
            storage = torch.cat(storage, axis=0)
            single_preds.append(storage)
        # (num_models, loader_len, 4)

        del loader
        # (num_rounds, num_models, loader_len, 4)
        results = torch.cat(single_preds, axis=1)
        cofactors_means = []; cofactors_std = []
        means_with_nans = np.empty((num_sequences, 4))
        means_with_nans[:] = np.nan
        stds_with_nans = means_with_nans.copy()
        mean = results.view(-1, num_sequences, 4).mean(0)
        std = results.view(-1, num_sequences, 4).std(0)
        mean, std = mean.cpu().numpy(), std.cpu().numpy()
        means_with_nans[indices_with_embeddings, :] = mean
        stds_with_nans[indices_with_embeddings, :] = std

        df_results = pd.DataFrame(means_with_nans, columns=self.COFACTORS)
        df_results_std = pd.DataFrame(stds_with_nans, columns=self.COFACTORS_STD)
        df_results['seq'] = self.sequences
        df_output = pd.concat([df_results, df_results_std], axis=1).round(8)
        del df_results, df_results_std

        return df_output


    def _prepare_samples(self, dataframe, contact_maps, edge_feats, foldx_info, BATCH_SIZE=64):
        
        self.CM_THRESHOLD = GraphECMLoaderBalanced.threshold
        dataset = GraphECMLoaderBalanced(frame=dataframe,
                        contact_maps=contact_maps,
                        edge_features=edge_feats,
                        foldx_info=foldx_info)
        if len(dataset) == 0:
            raise ValueError('dataset is empty')

        kw_args = {'batch_size' : BATCH_SIZE,
                    'shuffle' : False,
                    'num_workers' : self.NUM_WORKERS,
                    'drop_last' : False
                    }
        dataset_loaded = []
        #pin dataset to class instance for futher use
        self.dataset = dataset
        loader = DataLoader(dataset, collate_fn=collate, **kw_args)
        return loader

    def generate_embeddings(self, dataframe, contact_maps, edge_feats, foldx_info, verbose=False,
                            as_array=False, **kw_args):
        """
        params:
            dataframe (pd.DataFrame) with 'seq' and 'secondary' columns
            contact_maps (dict) similar as above
            edge_feats (dict) similar as above
            raw_scores (bool) if True probabilities are replaced with raw scores
            verbose (bool) default False
            as_array (bool) if True returns numpy array
        returns:
            np.ndarray/dict
        """

        available_sequences = foldx_info.keys() & contact_maps.keys() & edge_feats.keys()
        self.sequences = dataframe['seq'].tolist()
        indices = dataframe.index.tolist()
        num_sequences = len(self.sequences)
        available_sequences = available_sequences & set(indices)
        
        if verbose and (len(available_sequences) == 0):
            raise ValueError('mismatched keys')
        else:
            pass
            #print('seq to process:', len(available_sequences), num_sequences)
        indices_with_embeddings = [i for i, seq in enumerate(indices) if seq in available_sequences]
        mask_with_embeddings = [True if idx in indices_with_embeddings else False for idx in indices]
        sequences_without_embeddings = set(indices) - available_sequences
        
        if verbose and (len(sequences_without_embeddings) > 0):
            print(f'found {len(sequences_without_embeddings)} sequences without embeddings/contact maps')
            dataframe_no_missings = dataframe[~dataframe.index.isin(sequences_without_embeddings)].copy()
        else:
            dataframe_no_missings = dataframe.copy()

        loader = self._prepare_samples(dataframe_no_missings, contact_maps, edge_feats, foldx_info, BATCH_SIZE=1)

        single_preds = []
        with torch.no_grad():
            for x, _ in loader:
                if self.device_type == 'cuda':
                    x = x.to(self.device)
                storage = [self.forward_pass(x, model).unsqueeze(0) for model in self.model_list]
                storage = torch.cat(storage, axis=0).cpu().numpy()
                if not as_array:
                    storage = {cof_name : storage[:, :, i].mean(0)[:, np.newaxis] for i, cof_name in enumerate(self.COFACTORS)}
                single_preds.append(storage)
        del loader
        return single_preds


    def forward_pass(self, g, model):
        """
        execute custom DL forward pass to extract node scores
        """
        features = model(g, nodes=True)
        g.ndata['sum'] = features
        h = dgl.sum_nodes(g, 'sum')
        ######### attention block ###############
        attn = h.clone()
        attn = torch.nn.functional.relu(attn)
        attn = model.affine(attn)
        attn = torch.sigmoid(attn)
        features = features*attn

        feats = features.cpu().detach()
        feats = feats.reshape(-1, 4)
        return feats


class ShowNxGraph:
    #default cmap
    '''
    base class for graph plots
    '''
    def __init__(self, cmap=plt.cm.GnBu, figsize=(10, 10)):
        '''
        optional params:
            cmap - plt.cm object
            figsize - tuple(int, int) size of a plot
        '''
        self.cmap = cmap
        self.figsize = figsize

    def color_nodes(self, nodes):

        colored = []
        for node in nodes:
            colored.append(self.cmap(node))
        return colored

    def draw(self, g, residue_scores, node_labels, node_positions = 'default', node_size=150, ax = None):
        '''
        draw
        params:
            g (nx.graph)
            residue_scores (np.ndarray, dict) - if array color nodes with default cmap if dict
                ( in form of {nude_nb : (R,G,B)}) used given values
            node_labels (list) node names
            node_positions (str or nx.layout) position of residues
            node_size (int) node size
            ax (None or plt.figure) if None creates plt.subplots instance if figure fits to ax
        return:
            fig, ax
        '''

        assert isinstance(residue_scores, (dict, np.ndarray)), 'wrong type of residue_scores'
        assert node_labels is None or isinstance(node_labels, (list, str)), 'wrong arg type'
        assert isinstance(node_positions, (str, dict)), 'wrong arg type'
        assert isinstance(node_size, int), 'wrong arg type'




        if node_labels is not None:
            sec_labels = {i: s for i,s in enumerate(node_labels)}
            #define topology
            #if len(set(secondary) - {'C1','C2', 'C3', 'C4', 'E1', 'E2', 'H1', 'H2'}) > 0:
            #    secondary = reduce_ss_alphabet(secondary, True)
            #else:
            #    secondary = {i : s for i,s in enumerate(secondary)}

        if isinstance(node_positions, str):
            p = Positions(list(secondary.values()))
            p._find_changes()
            positions = p.get_positions(p._new_sec_as_dict())
        else:
            positions = node_positions
        #define colors
        if isinstance(residue_scores, np.ndarray):
            node_colors = self.color_nodes(residue_scores)
        elif isinstance(residue_scores, dict):
            node_colors = list(residue_scores.values())

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=self.figsize)
        else:
            fig = None
        nx.draw_networkx_nodes(g, positions, node_color=node_colors, ax=ax, alpha=0.9, node_size=node_size)
        nx.draw_networkx_edges(g, positions, ax=ax, alpha=0.4)
        if node_labels is not None:
            nx.draw_networkx_labels(g, positions, labels = sec_labels,ax=ax)

        return fig, ax

















