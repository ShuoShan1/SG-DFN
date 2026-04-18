import torch
import numpy as np
from src.base.base_utils import *


class Evaluate:
    def __init__(self, test_pair):
        self.test_pair = test_pair

        def dot(tensor):
            A, B = [matrix.squeeze(0) for matrix in tensor]
            A_sim = torch.matmul(A, B.t())
            return A_sim.unsqueeze(0)

        self.dot = dot

        k = 10
        def avg_top_k(x):
            x = x.squeeze(0)
            top_k_values, _ = torch.topk(x, k=k, dim=-1)
            return (top_k_values.sum(dim=-1) / k).unsqueeze(0)

        self.avg_top_k = avg_top_k

        def CSLS(tensor):
            # 确保所有输入张量都在同一个设备
            device = tensor[0].device 
            sim, LR, RL, ans_rank = [m.squeeze(0).to(device) for m in tensor]  # 全部移到同一设备
            
            epoch_num, all_num = sim.shape[0], sim.shape[1]
            LR, RL, ans_rank = [m.unsqueeze(1).to(device) for m in [LR, RL, ans_rank]]  # 确保维度调整后仍在 GPU
            
            LRS, RLS = LR.expand(-1, epoch_num), RL.expand(-1, all_num)
            
            sim = 2 * sim - LRS.T
            sim = sim - RLS
            rank = torch.argsort(-sim, dim=-1)
            
            # 确保 ans_rank 也在 GPU 上，并转换为 int32
            ans_rank_expanded = ans_rank.expand_as(rank).to(torch.int32).to(device)
            results = torch.where(rank == ans_rank_expanded)
            
            return results[1].unsqueeze(0)
        

        self.CSLS = CSLS

    def CSLS_cal(self, Lvec, Rvec, evaluate=True, batch_size=1024):
        L_sim, R_sim = [], []
        for epoch in range(len(Lvec) // batch_size + 1):
            L_batch = Lvec[epoch * batch_size:(epoch + 1) * batch_size]
            R_batch = Rvec[epoch * batch_size:(epoch + 1) * batch_size]
            L_sim.append(self.dot([L_batch.clone().detach().unsqueeze(0), Rvec.clone().detach().unsqueeze(0)]))
            R_sim.append(self.dot([R_batch.clone().detach().unsqueeze(0), Lvec.clone().detach().unsqueeze(0)]))

        LR, RL = [], []
        for epoch in range(len(Lvec) // batch_size + 1):
            LR.append(self.avg_top_k(L_sim[epoch]))
            RL.append(self.avg_top_k(R_sim[epoch]))

        if evaluate:
            results = []
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(Lvec)))])
                result = self.CSLS([R_sim[epoch], torch.cat(LR, dim=1), RL[epoch], torch.tensor(ans_rank).unsqueeze(0)])
                results.append(result)
            return torch.cat(results, dim=1)[0].tolist()
        else:
            l_rank, r_rank = [], []
            for epoch in range(len(Lvec) // batch_size + 1):
                ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch + 1) * batch_size, len(Lvec)))])
                r_rank.append(self.CSLS([R_sim[epoch], torch.cat(LR, dim=1), RL[epoch], torch.tensor(ans_rank).unsqueeze(0)]))
                l_rank.append(self.CSLS([L_sim[epoch], torch.cat(RL, dim=1), LR[epoch], torch.tensor(ans_rank).unsqueeze(0)]))

            return torch.cat(r_rank, dim=1)[0], torch.cat(l_rank, dim=1)[0]


def CSLS_evaluate(test_pair,Lvec, Rvec,out_feature):
    evals = Evaluate(test_pair=test_pair)
    results = evals.CSLS_cal(Lvec, Rvec)
    def cal(results):
        hits1, hits10, mrr = 0, 0, 0
        for x in results:
            if x < 1:
                hits1 += 1
            if x < 10:
                hits10 += 1
            mrr += 1 / (x + 1)
        return hits1, hits10, mrr

    hits1, hits10, mrr = cal(results)
    print(f"| Hits@1: {((hits1 / len(Lvec))*100):.3f} | Hits@10: {((hits10 / len(Lvec))*100):.3f} | MRR: {((mrr / len(Lvec))*100):.3f} |\n")
    result_dict = {"Hits@1":hits1 / len(Lvec),"Hits@10":hits10 / len(Lvec),"MRR":mrr / len(Lvec)}
    return out_feature,result_dict