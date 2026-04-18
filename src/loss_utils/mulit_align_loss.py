import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLevelAlignmentLoss(nn.Module):
    """
    多层次实体对齐损失函数
    包含：图结构损失、语义对齐损失、融合一致性损失
    """
    def __init__(self, gamma=3.0, alpha=1, beta=1, theta=1, 
                 lambda_struct=30, tau_struct=10,
                 lambda_sem=20, tau_sem=5,
                 temperature=0.1):
        super(MultiLevelAlignmentLoss, self).__init__()
        self.gamma = gamma  # margin for contrastive loss
        self.alpha = alpha  # weight for structural loss
        self.beta = beta    # weight for semantic loss  
        self.theta = theta  # weight for fusion consistency loss
        
        # 图结构损失参数
        self.lambda_struct = lambda_struct
        self.tau_struct = tau_struct
        
        # 语义损失参数
        self.lambda_sem = lambda_sem
        self.tau_sem = tau_sem
        self.temperature = temperature
        
    def structural_contrastive_loss(self, pairs, struct_emb, node_size):
        """
        图结构对比损失 - 适用于图结构特征
        保持原有的强对比学习机制
        """
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = struct_emb[l], struct_emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, struct_emb)
        r_neg_dis = squared_dist(r_emb, struct_emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=node_size) - F.one_hot(r, num_classes=node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=node_size) - F.one_hot(r, num_classes=node_size))

        del r_neg_dis, l_neg_dis

        # 标准化处理
        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()

        l_loss = torch.logsumexp(self.lambda_struct * l_loss + self.tau_struct, dim=-1)
        r_loss = torch.logsumexp(self.lambda_struct * r_loss + self.tau_struct, dim=-1)
        
        return torch.mean(l_loss + r_loss)
    


    def forward(self, pairs, struct_emb, sem_emb, fusion_emb, node_size):
        """
        总损失函数
        """
        # 图结构损失
        struct_loss = self.structural_contrastive_loss(pairs, struct_emb, node_size)
        
        # 语义对齐损失
        sem_loss = self.structural_contrastive_loss(pairs, sem_emb, node_size)
        
        # 融合一致性损失
        fusion_loss = self.structural_contrastive_loss(pairs, fusion_emb, node_size)
        # 加权总损失
        total_loss = (self.alpha * struct_loss + 
                     self.beta * sem_loss 
                     + self.theta * fusion_loss
                     )
        
        return {
            'total_loss': total_loss,
            'struct_loss': struct_loss,
            'sem_loss': sem_loss,
            'fusion_loss': fusion_loss
        }

   