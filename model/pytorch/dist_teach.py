import numpy as np
import torch
import torch.nn as nn
import math
from model.pytorch.graph_transformer_edge_layer import GraphTransformerLayer
import torch.nn.functional as F
import dgl

from model.pytorch.cal_graph import _calculate_supports

from iTransformer import iTransformer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.input_dim = int(model_kwargs.get('input_dim'))
        self.output_dim = int(model_kwargs.get('output_dim'))
        self.num_node = int(model_kwargs.get('num_nodes'))
        self.model_dim = int(model_kwargs.get('model_dim'))
        self.dec_dim = int(model_kwargs.get('dec_dim'))
        self.num_heads = int(model_kwargs.get('num_heads'))
        self.num_encoder_layers = int(model_kwargs.get('num_encoder_layers'))
        self.batch_size = int(model_kwargs.get('batch_size'))
        self.num_decoder_layers = int(model_kwargs.get('num_decoder_layers'))  # 解码器层数
        self.dropout = float(model_kwargs.get('dropout', 0.1))
        self.l1_decay = float(model_kwargs.get('l1_decay', 1e-5))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.horizon = int(model_kwargs.get('horizon'))  # for the decoder

        # Add additional parameters required by GTNModel

        self.g_heads = int(model_kwargs.get('g_heads'))
        self.g_dim = int(model_kwargs.get('g_dim'))
        self.num_g_layers = int(model_kwargs.get('num_g_layers'))
        self.layer_norm = model_kwargs.get('layer_norm', True)
        self.use_bias = model_kwargs.get('use_bias', True)
        self.batch_norm = model_kwargs.get('batch_norm', False) #已测试，False好
        self.residual = model_kwargs.get('residual', True)
        self.edge_feat = model_kwargs.get('edge_feat', True)
        self.g_threshold = model_kwargs.get('g_threshold')
        self.pos_att = model_kwargs.get('pos_att')
        self.gck = model_kwargs.get('gck')
        self.se = model_kwargs.get('static_edge', False)
        #self.num_atom_type = int(model_kwargs.get('num_atom_type', 10)) #节点类型
        #self.num_bond_type = int(model_kwargs.get('num_bond_type', 4)) #边类型
        #self.readout = model_kwargs.get('readout', 'max')
        #self.dua_pos_enc = model_kwargs.get('lap_pos_enc', False)
        #self.sig_pos_enc = model_kwargs.get('wl_pos_enc', False)


"""
    Graph Transformer with edge features

"""

class GTModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger,graph, cuda,**model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.device = cuda
        self._logger = logger

        g, e, Kr = _calculate_supports(graph,self.g_threshold)

        self.e = e.to(self.device)
        self.kr = Kr.to(self.device)
        g = g.to(self.device)
        self.batch_g = dgl.batch([g for _ in range(self.batch_size)])

        self.encoder = iTransformer(
                num_variates = self.num_node,
                lookback_len = self.seq_len,                  # or the lookback length in the paper
                dim = self.model_dim,                          # model dimensions
                depth = self.num_encoder_layers,                          # depth
                heads = self.num_heads,                          # attention heads
                dim_head = self.model_dim//2,                      # head dimension
                pred_length = self.seq_len,     # can be one prediction, or many
                num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
                use_reversible_instance_norm = False
            )
        
        self.decoder =  iTransformer(
                num_variates = self.num_node,
                lookback_len = self.seq_len*3,                  # or the lookback length in the paper
                dim = self.dec_dim,                          # model dimensions
                depth = self.num_decoder_layers,                          # depth
                heads = self.num_heads,                          # attention heads
                dim_head = self.dec_dim//2,                      # head dimension
                pred_length = self.horizon,     # can be one prediction, or many
                num_tokens_per_variate = 1,         # experimental setting that projects each variate to more than one token. the idea is that the network can learn to divide up into time tokens for more granular attention across time. thanks to flash attention, you should be able to accommodate long sequence lengths just fine
                use_reversible_instance_norm = False
            )
        self.GTlayers = nn.ModuleList([ GraphTransformerLayer(self.g_dim, self.g_dim, self.g_heads, self.pos_att, self.dropout,
                                                    self.layer_norm, self.batch_norm, self.residual, self.use_bias, self.gck) for _ in range(self.seq_len) ])

        self.g_conv=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.g_dim, kernel_size = 3, stride=1, padding=1) ,
            nn.BatchNorm1d(self.g_dim)
            
  
            
        )
        self.e_conv=nn.Sequential(
            nn.Conv1d(in_channels=self.num_node, out_channels=self.g_dim, kernel_size = 3, stride=1, padding=1) ,
            nn.BatchNorm1d(self.g_dim)
       
        )
        self.kr_conv=nn.Sequential(
            nn.Conv1d(in_channels=self.num_node, out_channels=self.g_dim, kernel_size = 3, stride=1, padding=1) ,
            nn.BatchNorm1d(self.g_dim)
     
        )
 

        self.g_upemb_lin =  nn.Sequential(
            nn.Linear(self.g_dim,64),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(64,self.num_node)
        )
        self.h_upemb_lin =  nn.Sequential(
            nn.Linear(self.g_dim,8),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(8,1)
        )
        
        
        self._reset_parameters()
        self._logger.info(
            "Total trainable parameters {}".format(count_parameters(self))
        )
        self._logger.info(
            "Model parameters: model_dim: {}, g_dim: {}, dec_dim: {},num_encoder: {}, num_g: {}"
            .format(self.model_dim,self.g_dim, self.dec_dim,self.num_encoder_layers, self.num_g_layers)
        )

    def _reset_parameters(self):
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, graph, batches_seen=None, tgt=None, h_lap_pos_enc=None, h_wl_pos_enc=None, src_mask=None, tgt_mask=None):
        src_g = src
        src = self.encoder(src)
        src = src[self.seq_len] #batch,len,num_node
        h_g=[]
        h_l=[]
        for i in range(self.seq_len):
            h = self.g_conv(src_g[:,i,:].view(self.batch_size,1,self.num_node)).view(-1,self.g_dim)
       
            batch_e_ = src[:,i,:].reshape(1,-1)
            batch_e = self.e * batch_e_
            batch_kr = self.kr * batch_e_
            
            batch_e = self.e_conv(batch_e.view(self.batch_size, self.num_node,-1)).view(-1,self.g_dim)
            batch_kr =  self.kr_conv(batch_kr.view(self.batch_size, self.num_node,-1)).view(-1,self.g_dim)

            h, batch_e = self.GTlayers[i](self.batch_g, h, batch_e, batch_kr)
            
            self.batch_g.ndata['h'] = h
            self.batch_g.edata['e'] = batch_e

            h = self.h_upemb_lin(h.view(self.batch_size,self.num_node,self.g_dim)).permute(0,2,1)
            h_l.append(h)
            
            g = dgl.mean_nodes(self.batch_g, 'h')
            g = self.g_upemb_lin(g)
            h_g.append(g)

        h_g = torch.stack(h_g, dim = 1)
        h_l = torch.stack(h_l, dim = 1).squeeze(2)
       
        memory = torch.cat((src, h_g, h_l),dim=1) 
        memory = self.decoder(memory)[self.horizon]

        return memory
