import sys

import dgl
import torch
from torch import nn
from torch import cat
from torch.nn import functional as F

from .edge_gat_layer import MultiHeadEGATLayer as GATConv

class LayerBlock(nn.Module):
    def __init__(self, in_features_n, in_features_e,
                 out_features_n, out_features_e, dropout_rate,
                 heads, attn_drop = 0.0, residual=True, attention_scaler = 'softmax', **kw_args):
        super().__init__()
        self.num_heads = heads
        self.out_features_n = out_features_n
        self.out_features_e = out_features_e
        self.in_features_n = in_features_n
        self.in_features_e = in_features_e
        self.dropout_rate = dropout_rate
        self.attn_drop = attn_drop
        self.bn = nn.BatchNorm1d(self.out_features_n*self.num_heads)
        self.bn_edge = nn.BatchNorm1d(self.out_features_e*self.num_heads)
        self.dp = nn.Dropout(self.dropout_rate)
        self.gat = GATConv(in_dim_n=self.in_features_n, 
                           in_dim_e = self.in_features_e, 
                           out_dim_n=self.out_features_n,
                           out_dim_e=self.out_features_e,
                            num_heads=self.num_heads, attn_drop=self.attn_drop,
                            residual=residual, activation=None,
                           attention_scaler = attention_scaler,
                            feat_drop=0)
    
    def forward(self, g, h, e):
        h, e = self.gat(g, h, e)
        h = h.view(-1, self.out_features_n*self.num_heads)
        e = e.view(-1, self.out_features_e*self.num_heads)
        h = self.bn(h)
        e = self.bn_edge(e)
        h = F.leaky_relu(h)
        h = self.dp(h)
        
        return h, e

class GAT(nn.Module):
    
    def __init__(self,
                 in_dim_n, in_dim_e, n_classes, p_feat = 0.33, \
                 p_attn = 0.0, residual = True, num_blocks=5,
                 blocks_heads = 2, block_in = 64, block_out = 32, hidden_dim_e=10,
                 include_seqvec = False, attention_scaler='softmax',
                  **kw_args):
        super(GAT, self).__init__()

        self.in_dim_n = int(in_dim_n)
        self.in_dim_e = int(in_dim_e)
        self.hidden_edge = hidden_dim_e
        self.n_classes = int(n_classes)
        self.p_attn = p_attn
        self.num_blocks = num_blocks
        self.blocks_heads = blocks_heads
        self.dropout = p_feat
        self.residual = bool(residual)
        self.num_bfi = int(block_in)
        self.num_bfo = int(block_out)
        self.include_seqvec = include_seqvec
        self.attention_scaler = attention_scaler
        
        # hidden sizes
        self.emb_size = 21#15
        self.emb2_size = 12#7
        self.dict_size = 21
        self.dict_size2 = 11
        self.in_dim =  self.emb_size + self.emb2_size + self.in_dim_n
        
        
        ### normalization layers
        self.bn_input = nn.BatchNorm1d(self.in_dim)
        self.bn_first = nn.BatchNorm1d(self.num_bfi)
        self.bn_edge = nn.BatchNorm1d(self.hidden_edge)
        self.dp_first = nn.Dropout(self.dropout)
        self.bn_last = nn.BatchNorm1d(self.n_classes)
        self.GATBlocks = nn.ModuleList()
        #### embedding layers
        self.emb = nn.Embedding(self.dict_size, self.emb_size)
        self.emb2 = nn.Embedding(self.dict_size2, self.emb2_size)

        
        self.gat_first = GATConv(in_dim_n=self.in_dim, in_dim_e=self.in_dim_e,
                                 out_dim_n=self.num_bfi, out_dim_e=self.hidden_edge,
                            num_heads=1, attention_scaler=attention_scaler)
        for n in range(self.num_blocks):
            self.GATBlocks.append(LayerBlock(in_features_n=self.num_bfi,
                                             in_features_e=self.hidden_edge,
                                            out_features_n=self.num_bfo,
                                             out_features_e=self.hidden_edge//2,
                                            dropout_rate=self.dropout,
                                            heads=self.blocks_heads,
                                            attention_scaler=self.attention_scaler))      
        self.gat_last = GATConv(in_dim_n=self.num_bfo*self.blocks_heads,
                                in_dim_e=self.hidden_edge*self.blocks_heads//2,
                                out_dim_n=self.n_classes, out_dim_e=1, num_heads=1, attn_drop=0,
                                attention_scaler = self.attention_scaler,
                                omit_edges=True)

        self.affine = nn.Linear(self.n_classes,self.n_classes, bias=True)
  
    def forward(self, g, embeddings=False, nodes=False, scores=False):
        #### extract feature vector
        emb = self.emb(g.ndata['residues']).view(-1, self.emb_size)
        emb2 = self.emb2(g.ndata['secondary']).view(-1, self.emb2_size)

        h = g.ndata['features']
        h = cat([emb, emb2, h], dim=1)
        e = g.edata['features']
        #print(e.shape, h.shape)
        
        #print(h.shape)
        ####### first layer  block #############
        h, e = self.gat_first(g, h, e)
        h = h.view(-1, self.num_bfi)
        e = e.view(-1, self.hidden_edge)
        h = self.bn_first(h)
        e = self.bn_edge(e)
        h = F.leaky_relu(h)
        h = self.dp_first(h)
        ###### loop layer block #############
        for block in self.GATBlocks:
            h, e = block(g, h, e)
        ###### last layer block #############
        h, e = self.gat_last(g, h, e)
        h = h.view(-1, self.n_classes)
        h = self.bn_last(h)
        ##### classify layer block #############
        if nodes:
            return h
        g.ndata['sum'] = h
        h = dgl.sum_nodes(g, 'sum')
        ######### attention block ###############
        h_affine = h.clone()
        h_affine = F.leaky_relu(h_affine)
        h_affine = self.affine(h_affine) 
        h_affine = torch.sigmoid(h_affine)
        if  embeddings:
            return h, h_affine
        h = h*h_affine
        if not scores:
            h = torch.sigmoid(h)
            return h
        else:
            return h
