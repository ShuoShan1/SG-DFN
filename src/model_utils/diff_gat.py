from torch_scatter import scatter_sum
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 差分融合注意力部分 ==========
class DIFF_GraphAttention(nn.Module):
    def __init__(self,node_size,node_dim,depth=1,alpha=0.5):
        super(DIFF_GraphAttention, self).__init__()

        self.node_size = node_size
        self.node_dim = node_dim
        self.activation = torch.nn.Tanh()
        self.depth = depth
        
        self.high_atts = nn.ParameterList()       # 主注意力参数
        self.diff_atts = nn.ParameterList()        # 差分注意力参数
        self.alpha = alpha                         # 差分权重参数
        
        
        for l in range(self.depth):
            # 主注意力
            high_att = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(high_att)
            self.high_atts.append(high_att)
            
            # 差分注意力
            diff_att = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(diff_att)
            self.diff_atts.append(diff_att)


    def forward(self, features, adj_nei):
        outputs = []
        features = self.activation(features)
     
        for l in range(self.depth):
            high_att = self.high_atts[l]
            diff_att = self.diff_atts[l]

            neighs = features[adj_nei[1, :].long()]
            main_att = torch.squeeze(torch.mm(neighs, high_att), dim=-1)
            diff_att_score = torch.squeeze(torch.mm(neighs, diff_att), dim=-1)
  
            combined_att = main_att - self.alpha * diff_att_score
            att_sparse = torch.sparse_coo_tensor(indices=adj_nei, values=combined_att, size=[self.node_size, self.node_size])
            att_softmax = torch.sparse.softmax(att_sparse, dim=1)
            weighted_neighs = neighs * torch.unsqueeze(att_softmax.coalesce().values(), dim=-1)
            new_features = scatter_sum(src=weighted_neighs,dim=0,index=adj_nei[0,:].long())
            
            features = self.activation(new_features)
            outputs.append(features)
        
        return features


