import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from iTransformer import iTransformer
from model.pytorch.graph_transformer_edge_layer import GraphTransformerLayer
from model.pytorch.cal_graph import _calculate_supports
from model.pytorch.divi_graph import split_graph_pcc, split_graph_nor, split_graph_pcc_plus
from model.pytorch.divi_graph import split_time_series, merge_time_series
from lib.loss_function import PaENLoss

class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.input_dim = int(model_kwargs.get('input_dim'))
        self.output_dim = int(model_kwargs.get('output_dim'))
        self.num_node = int(model_kwargs.get('num_nodes'))
        self.model_dim = int(model_kwargs.get('model_dim'))
        self.dec_dim = int(model_kwargs.get('dec_dim'))
        self.num_en_heads = int(model_kwargs.get('num_en_heads'))
        self.num_de_heads = int(model_kwargs.get('num_de_heads'))
        self.num_encoder_layers = int(model_kwargs.get('num_encoder_layers'))
        self.batch_size = int(model_kwargs.get('batch_size'))
        self.num_decoder_layers = int(model_kwargs.get('num_decoder_layers'))
        self.dropout = float(model_kwargs.get('dropout', 0.1))
        self.l1_decay = float(model_kwargs.get('l1_decay', 1e-5))
        self.seq_len = int(model_kwargs.get('seq_len'))
        self.horizon = int(model_kwargs.get('horizon'))
        self.g_heads = int(model_kwargs.get('g_heads'))
        self.g_dim = int(model_kwargs.get('g_dim'))
        self.num_g_layers = int(model_kwargs.get('num_g_layers'))
        self.layer_norm = model_kwargs.get('layer_norm', True)
        self.use_bias = model_kwargs.get('use_bias', True)
        self.batch_norm = model_kwargs.get('batch_norm', False)
        self.residual = model_kwargs.get('residual', True)
        self.edge_feat = model_kwargs.get('edge_feat', True)
        self.g_threshold = model_kwargs.get('g_threshold')
        self.pos_att = model_kwargs.get('pos_att')
        self.gck = model_kwargs.get('gck')
        self.info_type = model_kwargs.get('info_type')
        self.add_edge = model_kwargs.get('add_edge')
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CrossEdgeExtractor(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CrossEdgeExtractor, self).__init__()
        # 定义一个线性层，用于融合拼接后的特征
        self.fc = nn.Linear(input_dim * 2, hidden_dim)  # input_dim*2，因为拼接了两个节点的特征
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, ts, cross_edges):

        batchsize, seq_len, num_nodes = ts.shape
        num_cross_edges = cross_edges.shape[0]

        node1_indices = cross_edges[:, 0].long()  # 第一个节点的索引
        node2_indices = cross_edges[:, 1].long()  # 第二个节点的索引


        ts_node1 = ts[:, :, node1_indices]  # [batchsize, seq_len, num_cross_edge]
        ts_node2 = ts[:, :, node2_indices]  # [batchsize, seq_len, num_cross_edge]

        # 拼接两个节点的时间序列，沿最后一个维度拼接
        ts_concat = torch.cat((ts_node1, ts_node2), dim=-1)  # [batchsize, seq_len, num_cross_edge * 2]

        # 对每条边的时间序列进行融合
        ts_concat = ts_concat.view(batchsize, seq_len, num_cross_edges * 2)  # 扁平化拼接后的张量

        ts_fused = self.fc(ts_concat)  # [batchsize, seq_len, hidden_dim]

        # 对每个时间步进行处理，通过最终的输出层
        output = self.output_fc(ts_fused)  # [batchsize, seq_len, output_dim]

        return output


class Stud_M(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger, cuda, G, e_all, kr_all, subG, num_subgraph, split_indices, num_sub_node, cross_edges, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.device = cuda
        self._logger = logger
        self.num_subgraph = num_subgraph
        self.split_indices = split_indices
        # 预先批处理子图

        self.g_batched = dgl.batch([subG for _ in range(self.batch_size)]).to(self.device)
        self.batch_G = dgl.batch([G for _ in range(self.batch_size)]).to(self.device)  # 批处理 G
        
        self.cross_edges = cross_edges.to(self.device)
        self.cross_u_emb = nn.Linear(self.cross_edges.shape[0],self.num_node)
     
        self.num_sub_node = num_sub_node
 
        self.tol_emb = nn.Linear(self.num_node,self.num_sub_node)
        self.encoder = iTransformer(
            num_variates=self.num_sub_node,
            lookback_len=self.seq_len,
            dim=self.dec_dim,
            depth=self.num_decoder_layers,
            heads=self.num_de_heads,
            dim_head=self.dec_dim//2,
            pred_length=self.seq_len,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=False
        ) 
        
        self.decoder = iTransformer(
            num_variates=self.num_sub_node,
            lookback_len=self.seq_len*3,
            dim=self.dec_dim,
            depth=self.num_decoder_layers,
            heads=self.num_de_heads,
            dim_head=self.dec_dim//2,
            pred_length=self.horizon,
            num_tokens_per_variate=1,
            use_reversible_instance_norm=False
        ) 
        

        self.share_GTlayers = GraphTransformerLayer(
            self.g_dim,  self.g_dim, self.g_heads, self.pos_att, self.dropout,
            self.layer_norm, self.batch_norm, self.residual, self.use_bias, self.gck) 
        
   
        self.GTlayers = nn.ModuleList([GraphTransformerLayer(
             self.g_dim,  self.g_dim, self.g_heads, self.pos_att, self.dropout,
            self.layer_norm, self.batch_norm, self.residual, self.use_bias, self.gck
        ) for _ in range(self.num_g_layers)])
        
        self.e_all = e_all.unsqueeze(1).expand(-1, 64, 1).to(self.device)
        self.kr_all = kr_all.unsqueeze(1).expand(-1, 64, 1).to(self.device)

        self.extract_cross_edge = CrossEdgeExtractor(self.g_dim, 64, self.num_node)
      
        
        self.g_emb  = nn.Linear(self.seq_len, self.g_dim) 
        self.e_emb  = nn.Linear(self.seq_len*self.num_sub_node, self.g_dim)
        self.dec_emb = nn.Linear(self.g_dim, self.seq_len) 
        if  self.add_edge:
            self.generator = GraphGenerator(
                input_dim=self.g_dim, 
                output_dim=self.g_dim,
                hidden_dim = 16
            )
        self._reset_parameters()


        self._logger.info( "Student_Model parameters: model_dim: {}, g_dim: {}, dec_dim: {},num_encoder: {}, num_g: {},  sub_graph: {}, sub_node: {}"
            .format(self.model_dim,self.g_dim, self.dec_dim,self.num_encoder_layers, self.num_g_layers, self.num_subgraph, self.num_sub_node))
        # 计算并打印模型的参数量
        total_params = count_parameters(self)
        self._logger.info("Total trainable parameters: {}".format(total_params))
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            
    

    def forward(self, x):
        
        ts_s = self.encoder(x)[self.seq_len] 
        g = self.g_batched
        e = g.edata['e'].view(self.batch_size, -1, 1).to(self.device)
        kr = g.edata['Kr'].view(self.batch_size, -1, 1).to(self.device)
   
        ts_h = self.g_emb(ts_s.permute(0,2,1))
        ts_e = self.e_emb(ts_s.reshape(self.batch_size,-1)).unsqueeze(1)
    
        e_emb = (e * ts_e).view(-1, self.g_dim) 
        kr_emb = (kr * ts_e).view(-1, self.g_dim) 

        h = ts_h.view(-1, self.g_dim)
   
             
        total_h_1 = torch.full((self.batch_size, self.num_node, self.g_dim), 0.0, device=self.device, dtype=torch.float)
        total_h_1[:, self.split_indices, :] = h.view(self.batch_size, self.num_sub_node, -1)
        
        if self.add_edge:
            generated_features = self.generator(total_h_1, self.cross_edges)
            pa = PaENLoss(total_h_1, generated_features)
            total_h_1 =   generated_features + total_h_1
            
        e_all = self.e_all * ts_e.permute(1,0,2)
        kr_all = self.kr_all * ts_e.permute(1,0,2)
        
        total_h_1, e_all = self.share_GTlayers(self.batch_G, total_h_1.view(-1,self.g_dim), 
                                                e_all.view(-1,self.g_dim), kr_all.view(-1,self.g_dim))
        
        for GTlayers in self.GTlayers:  
            h, e_emb = GTlayers(g, h, e_emb, kr_emb)
            
        total_h_1 =self.tol_emb(total_h_1.view(-1,self.num_node)).view(self.batch_size, -1, self.g_dim)
        h = h.view(self.batch_size, -1, self.g_dim)

        h = self.dec_emb(h).permute(0,2,1)
        total_h_1 = self.dec_emb(total_h_1).permute(0,2,1)

        h = ts_s + h
        subcomb = torch.cat((ts_s,total_h_1, h),dim = 1)
        y = self.decoder(subcomb)[self.horizon]
        return y, h, pa


class Stud_CombM(nn.Module):
    def __init__(self, logger, cuda, G, e_all, kr_all, cross_edges, **model_kwargs):
        super(Stud_CombM, self).__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.device = cuda
        self.logger = logger
        # Graph structure initialization
        self.G = G.to(self.device)
        self.batch_G = dgl.batch([self.G for _ in range(self.batch_size)]).to(self.device)
        self.e_all = e_all.to(self.device)
        self.kr_all = kr_all.to(self.device)
        self.cross_edges = cross_edges.to(self.device)

        # Generator and discriminator initialization
        self.generator = GraphGenerator(
            input_dim=self.cross_edges.shape[0], 
            output_dim=self.num_node,
            hidden_dim = 128)
        # Embedding for merging the generated edge features into the graph features

        # Decoder initialization using iTransformer
        if self.add_edge:
            self.decoder = iTransformer(
                num_variates=self.num_node,
                lookback_len=self.seq_len*2,
                dim=self.model_dim,
                depth=self.num_encoder_layers,
                heads=self.num_en_heads,
                dim_head=self.model_dim // 2,
                pred_length=self.horizon,
                num_tokens_per_variate=1,
                use_reversible_instance_norm=False
                )                           
        else:
            self.decoder = iTransformer(
                num_variates=self.num_node,
                lookback_len=self.seq_len,
                dim=self.model_dim,
                depth=self.num_encoder_layers,
                heads=self.num_en_heads,
                dim_head=self.model_dim // 2,
                pred_length=self.horizon,
                num_tokens_per_variate=1,
                use_reversible_instance_norm=False
            )
        self.extract_cross_edge = CrossEdgeExtractor(self.cross_edges.shape[0], 64, self.cross_edges.shape[0])
        self._reset_parameters()

        # Log model parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info("Total Student_Comb_Model parameters: {}".format(total_params))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, student_outputs):
        # Step 1: Encode time series features
       
        if self.add_edge:
            att = self.extract_cross_edge(student_outputs, self.cross_edges)
            generated_features = self.generator(att, self.cross_edges)
            pa = PaENLoss(att, generated_features)
            #merged = student_outputs + generated_features
            merged = torch.cat((student_outputs, generated_features),dim = 1)
          
        else: 
            merged = student_outputs
            pa = torch.tensor([0.0]).to(student_outputs.device)
            
        merged = self.decoder(merged)[self.horizon]

        return merged, pa



class GraphGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128, dp = 0.1):
        super(GraphGenerator, self).__init__()
        self.input_dim = input_dim
        self.ln = nn.LayerNorm(input_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第一个全连接层
        self.ou = nn.Linear(hidden_dim, output_dim)  # 第一个全连接层
        self.dropout1 = nn.Dropout(p=dp)
        self.dropout2 = nn.Dropout(p=dp)
        self.dropout3 = nn.Dropout(p=dp)

        

    def forward(self, x, cross_edges):
        device = x.device
        
        generated_features = F.gelu(self.fc1(x + torch.randn_like(x)))  
        generated_features = self.dropout1(generated_features)
        generated_features = F.gelu(self.fc2(generated_features))
        generated_features = self.dropout2(generated_features)
        generated_features = self.ou(generated_features) 
        
        return generated_features

