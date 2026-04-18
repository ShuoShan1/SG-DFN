import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model_utils.gcn_layer import Local_Global_Network
from src.model_utils.sem_layer import Deep_Residual_Network
from src.model_utils.diff_gat import DIFF_GraphAttention
from src.loss_utils.mulit_align_loss import MultiLevelAlignmentLoss


class Encoder_Model(nn.Module):
    def __init__(self, node_hidden, rel_hidden,att_hidden,triple_size, node_size, rel_size,att_size,ent_semantic_emb,rel_semantic_emb,att_semantic_emb,device,
                 adj_matrix, r_index, r_val, rel_matrix,att_matrix,ent_matrix,ill_ent,dropout_rate=0.0,
                gamma=3, lr=0.005, depth=2,high_adj = []):# high_adj
        super(Encoder_Model, self).__init__()
        self.node_hidden = node_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.att_size = att_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.att_adj = att_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)
        self.ill_ent = ill_ent
        self.high_adj = high_adj.to(device)

        self.loss_fn = MultiLevelAlignmentLoss(
            gamma=3.0,
            alpha=1,    # 结构损失权重
            beta=1,     # 语义损失权重
            theta=1,    # 融合损失权重
            lambda_struct=30,
            tau_struct=10,
            lambda_sem=20,
            tau_sem=5,
            temperature=0.1
        )

        # ----------图谱级信息----------
        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)
        self.att_embedding = nn.Embedding(att_size, att_hidden)
        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
        torch.nn.init.xavier_uniform_(self.att_embedding.weight)
        
        self.e_encoder = Local_Global_Network(node_size=self.node_size,rel_size=self.rel_size,triple_size=self.triple_size,node_dim=self.node_hidden,depth=self.depth,use_bias=True)
        self.r_encoder = Local_Global_Network(node_size=self.node_size,rel_size=self.rel_size,triple_size=self.triple_size,node_dim=self.node_hidden,depth=self.depth,use_bias=True)
        self.a_encoder = Local_Global_Network(node_size=self.node_size,rel_size=self.rel_size,triple_size=self.triple_size,node_dim=self.node_hidden,depth=self.depth,use_bias=True)
        
        
        # ----------语义级信息----------
        self.ent_semantic_emb = ent_semantic_emb.to(device)
        self.rel_semantic_emb = rel_semantic_emb.to(device)
        self.att_semantic_emb = att_semantic_emb.to(device)
        self.ent_semantic_encoder = Deep_Residual_Network(in_features=4096,out_features=500)
        self.rel_semantic_encoder = Deep_Residual_Network(in_features=4096,out_features=500)
        self.att_semantic_encoder = Deep_Residual_Network(in_features=4096,out_features=500)
        

        # ----------融合编码器----------
        self.fusion_encoder = DIFF_GraphAttention(node_size = self.node_size,
                                                  node_dim = 2000, 
                                                  depth = 1)
   
    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)


    def gcn_forward(self):
        # ----------图结构嵌入----------
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight,self.node_size)
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight,self.rel_size)
        att_teature = self.avg(self.att_adj, self.att_embedding.weight,self.att_size)
        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val,self.high_adj]
        e_feature = self.e_encoder([ent_feature] + opt)
        r_feature = self.r_encoder([rel_feature] + opt)
        a_feature = self.a_encoder([att_teature] + opt)
        fusion_kg_feature = torch.cat([e_feature,r_feature,a_feature], dim=-1) 
        fusion_kg_feature = self.dropout(fusion_kg_feature)
        
        # # # ----------语义嵌入----------
        ent_sem_feature = self.ent_semantic_encoder(self.ent_semantic_emb)
        rel_sem_feature = self.rel_semantic_encoder(self.rel_semantic_emb)
        att_sem_feature = self.att_semantic_encoder(self.att_semantic_emb)
        fusion_sem_feature = ent_sem_feature + rel_sem_feature + att_sem_feature

        
        # # ----------融合嵌入----------
        fusion_out_feature = torch.cat([fusion_kg_feature,fusion_sem_feature], dim=-1)
        fusion_out_feature = self.fusion_encoder(fusion_out_feature,self.ent_adj)
        fusion_out_feature = torch.cat([fusion_kg_feature,fusion_sem_feature,fusion_out_feature], dim=-1)

        return fusion_out_feature,fusion_kg_feature,fusion_sem_feature
        # return fusion_kg_feature,fusion_kg_feature,fusion_kg_feature # 纯粹结构
        # return fusion_sem_feature,fusion_sem_feature,fusion_sem_feature # 纯粹语义
   

    def forward(self, train_paris:torch.Tensor, flag):
        if flag:

            out_feature,str_feature,sem_feature = self.gcn_forward()
            
            loss_dict = self.loss_fn(
                pairs=train_paris,
                struct_emb=str_feature,
                sem_emb=sem_feature, 
                fusion_emb=out_feature,
                node_size=self.node_size
            )
            return loss_dict['total_loss']
            # return loss_dict['struct_loss'] # 纯粹结构
            # return loss_dict['sem_loss'] # 纯粹语义

       
        else:
            # print("嵌入重置")
            torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
            torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
            torch.nn.init.xavier_uniform_(self.att_embedding.weight)
            self.ent_semantic_encoder.reset_parameters()
            self.rel_semantic_encoder.reset_parameters()
            self.att_semantic_encoder.reset_parameters()

            out_feature,str_feature,sem_feature = self.gcn_forward()
            loss_dict = self.loss_fn(
                pairs=train_paris,
                struct_emb=str_feature,
                sem_emb=sem_feature,
                fusion_emb=out_feature, 
                node_size=self.node_size
            )
            
            return loss_dict['total_loss']
            # return loss_dict['struct_loss']



    def get_embeddings(self, index_a, index_b):
        out_feature,str_feature,sem_feature = self.gcn_forward()
    
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)

        out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
        str_feature = str_feature / (torch.linalg.norm(str_feature, dim=-1, keepdim=True) + 1e-5)
        sem_feature = sem_feature / (torch.linalg.norm(sem_feature, dim=-1, keepdim=True) + 1e-5)
        print(out_feature.device)
        return Lvec, Rvec,out_feature,str_feature,sem_feature


