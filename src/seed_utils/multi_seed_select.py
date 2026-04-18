from tqdm import tqdm
import time
import numpy as np
import torch
from src.seed_utils.sinkhorn import matrix_sinkhorn


class MultiViewCooperativeSeedSelection:
    """
    多视角协同种子筛选类
    实现基于语义、结构、融合三个视角的协同种子筛选策略
    """
    def __init__(self, semantic_emb, structural_emb,fusion_emb, alpha=0.5, beta=0.5):
        """
        初始化多视角种子筛选器
        
        Args:
            semantic_emb: 语义嵌入 (语义特征)
            structural_emb: 结构嵌入 (结构特征)
            alpha: 语义嵌入权重
            beta: 结构嵌入权重
        """
        self.semantic_emb = semantic_emb
        self.structural_emb = structural_emb
        self.alpha = alpha
        self.beta = beta
        
        self.fusion_emb = fusion_emb
    
   
    
    def bidirectional_selection(self, similarity_matrix, entity1, entity2, train_pair):
        # 转换为列表
        entity1_list = entity1.tolist() if isinstance(entity1, torch.Tensor) else entity1
        entity2_list = entity2.tolist() if isinstance(entity2, torch.Tensor) else entity2
        
        # 构建训练集实体集合
        train_entities_src = set([pair[0] for pair in train_pair])
        train_entities_tgt = set([pair[1] for pair in train_pair])
        
        # 正向选择：对于每个源实体，找到最相似的目标实体
        forward_pairs = {}  # {src: tgt}
        max_values_forward, max_indices_forward = torch.max(similarity_matrix, dim=1)
        
        for i, src_entity in enumerate(entity1_list):
            if src_entity not in train_entities_src:
                tgt_idx = max_indices_forward[i].item()
                tgt_entity = entity2_list[tgt_idx]
                if tgt_entity not in train_entities_tgt:
                    forward_pairs[src_entity] = tgt_entity
        
        # 反向选择：对于每个目标实体，找到最相似的源实体
        backward_pairs = {}  # {tgt: src}
        similarity_matrix_t = similarity_matrix.t()
        max_values_backward, max_indices_backward = torch.max(similarity_matrix_t, dim=1)
        
        for j, tgt_entity in enumerate(entity2_list):
            if tgt_entity not in train_entities_tgt:
                src_idx = max_indices_backward[j].item()
                src_entity = entity1_list[src_idx]
                if src_entity not in train_entities_src:
                    backward_pairs[tgt_entity] = src_entity
        
        # 双向验证：只保留双向都选中的实体对
        candidate_seeds = []
        for src, tgt in forward_pairs.items():
            if tgt in backward_pairs and backward_pairs[tgt] == src:
                candidate_seeds.append([src, tgt])
        
        return candidate_seeds
    
    def sim_similarity(self, entity1, entity2, emb_type='hybrid', k=10):
        if emb_type == 'semantic':
            emb = self.semantic_emb
        elif emb_type == 'structural':
            emb = self.structural_emb
        elif emb_type == 'fusion':
            emb = self.fusion_emb
        elif emb_type == 'hybrid':
            # 混合嵌入
            emb = torch.cat([self.alpha * self.semantic_emb,self.beta * self.structural_emb], dim=-1)
        else:
            raise ValueError("emb_type must be 'semantic', 'structural', 'fusion' or 'hybrid'")
            
        lemb = emb[entity1]
        remb = emb[entity2]
        
        # 计算基础相似度矩阵
        A_sim = torch.mm(lemb, remb.t())
        
        return A_sim
    
    def sinkhorn_similarity(self, entity1, entity2, emb_type='hybrid', k=10):

        similarity_matrix = self.sim_similarity(entity1, entity2, emb_type, k)
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        distance_matrix = 1 - similarity_matrix

        # 调用matrix_sinkhorn
        sinkhorn_result = matrix_sinkhorn(
            pred_or_m=distance_matrix, 
            expected=None,  
            device=distance_matrix.device
        )
        
        return sinkhorn_result


    def get_candidate_pools(self, entity1, entity2, train_pair):
        """
        从三个视角分别获取候选种子集合
        Spool = Ssem + Sstr + Sfinal
        
        Args:
            entity1: 源实体索引
            entity2: 目标实体索引
            train_pair: 训练种子对
            
        Returns:
            三个视角的候选种子集合和合并后的候选池
        """
        print("正在计算三个视角的候选种子...")
        

        # 1. 语义视角
        sem_sim = self.sinkhorn_similarity(entity1, entity2, emb_type='semantic',k=10)
        S_sem = self.bidirectional_selection(sem_sim, entity1, entity2, train_pair)

        
        # 2. 结构视角
        str_sim = self.sinkhorn_similarity(entity1, entity2, emb_type='structural',k=10)
        S_str = self.bidirectional_selection(str_sim, entity1, entity2, train_pair)
        
        # 3. 融合视角
        final_sim = self.sinkhorn_similarity(entity1, entity2, emb_type='fusion',k=10)
        S_final = self.bidirectional_selection(final_sim, entity1, entity2, train_pair)
        
        # 合并候选池
        S_pool = []
        seen = set()
        for seed_list in [S_sem, S_str, S_final]:
            for seed in seed_list:
                seed_tuple = tuple(seed)
                if seed_tuple not in seen:
                    S_pool.append(seed)
                    seen.add(seed_tuple)

        return S_sem, S_str, S_final, S_pool
    
    def compute_view_support(self, S_sem, S_str, S_final, S_pool):
        """
        计算每个候选种子对的视角支持度
        vsupport(ei^s, ej^t) = [I_sem, I_str, I_final]
        ssupport(ei^s, ej^t) = I_sem + I_str + I_final
        
        Args:
            S_sem: 语义视角候选种子
            S_str: 结构视角候选种子
            S_final: 融合视角候选种子
            S_pool: 候选池
            
        Returns:
            视角支持度字典 {(src, tgt): (support_vector, support_sum)}
        """

        # 构建视角集合以快速查找
        sem_set = set([tuple(seed) for seed in S_sem])
        str_set = set([tuple(seed) for seed in S_str])
        final_set = set([tuple(seed) for seed in S_final])
        
        view_support = {}
        
        for seed in S_pool:
            seed_tuple = tuple(seed)
            
            # 计算视角支持度向量
            I_sem = 1 if seed_tuple in sem_set else 0
            I_str = 1 if seed_tuple in str_set else 0
            I_final = 1 if seed_tuple in final_set else 0
            
            support_vector = [I_sem, I_str, I_final]
            support_sum = I_sem + I_str + I_final
            
            view_support[seed_tuple] = (support_vector, support_sum)
        
        # 统计支持度分布
        support_dist = {}
        for _, (_, support_sum) in view_support.items():
            support_dist[support_sum] = support_dist.get(support_sum, 0) + 1
        return view_support
    
    def conflict_resolution(self, S_pool, view_support):
        """
        基于视角支持度的冲突消解
        
        策略：
        1. 构建实体映射关系
        2. 对于有冲突的映射，保留视角支持度严格最大的候选对
        3. 如果存在多个候选对具有相同的最大支持度，全部过滤
        
        Args:
            S_pool: 候选池
            view_support: 视角支持度字典
            
        Returns:
            消解后的种子集合
        """
        print("开始冲突消解...")
        
        # 构建映射关系字典
        M_s2t = {}  # 源实体 -> 目标实体集合
        M_t2s = {}  # 目标实体 -> 源实体集合
        
        for seed in S_pool:
            src, tgt = seed[0], seed[1]
            
            if src not in M_s2t:
                M_s2t[src] = []
            M_s2t[src].append(tgt)
            
            if tgt not in M_t2s:
                M_t2s[tgt] = []
            M_t2s[tgt].append(src)
        
        # 冲突消解 - 源到目标方向
        S_resolved_s2t = set()
        
        for src, tgt_list in M_s2t.items():
            if len(tgt_list) == 1:
                # 无冲突，直接保留
                S_resolved_s2t.add((src, tgt_list[0]))
            else:
                # 有冲突，按视角支持度消解
                candidates = [(tgt, view_support[(src, tgt)][1]) for tgt in tgt_list]
                max_support = max([support for _, support in candidates])
                
                # 找到所有具有最大支持度的候选
                max_candidates = [tgt for tgt, support in candidates if support == max_support]
                
                # 只有当最大支持度唯一且等于3时才保留
                if len(max_candidates) == 1 and max_support == 3:
                    S_resolved_s2t.add((src, max_candidates[0]))
                # 否则全部过滤（保守策略）
        
        # 冲突消解 - 目标到源方向
        S_resolved_t2s = set()
        
        for tgt, src_list in M_t2s.items():
            if len(src_list) == 1:
                # 无冲突，直接保留
                S_resolved_t2s.add((src_list[0], tgt))
            else:
                # 有冲突，按视角支持度消解
                candidates = [(src, view_support[(src, tgt)][1]) for src in src_list]
                max_support = max([support for _, support in candidates])
                
                # 找到所有具有最大支持度的候选
                max_candidates = [src for src, support in candidates if support == max_support]
                
                # 只有当最大支持度唯一且等于3时才保留
                if len(max_candidates) == 1 and max_support == 3:
                    S_resolved_t2s.add((max_candidates[0], tgt))
                # 否则全部过滤（保守策略）
        
        # 取交集得到最终的有效种子集
        S_valid = S_resolved_s2t & S_resolved_t2s
  
        # 转换回列表格式
        valid_seeds = [[src, tgt] for src, tgt in S_valid]
        
        return valid_seeds
    
    def multi_view_cooperative_mining(self, entity1, entity2, train_pair,test_pair):
        """
        多视角协同种子挖掘主函数
        
        Args:
            entity1: 源实体索引
            entity2: 目标实体索引
            train_pair: 训练种子对
            test_pair:测试种子对（用于案例分析）
            
        Returns:
            最终筛选的种子对列表
        """
        print("=" * 60)
        print("开始多视角协同种子挖掘")
        start_time = time.time()
        
        # 步骤1: 获取三个视角的候选池
        S_sem, S_str, S_final, S_pool = self.get_candidate_pools(
            entity1, entity2, train_pair
        )
        print(f"  语义视角候选池: {len(S_sem)} 个候选")
        print(f"  结构视角候选池: {len(S_str)} 个候选")
        print(f"  最终候选池: {len(S_final)} 个候选")
        print(f"  总候选池: {len(S_pool)} 个候选")
        # 步骤2: 计算视角支持度f
        view_support = self.compute_view_support(S_sem, S_str, S_final, S_pool)
        
        # 步骤3: 冲突消解
        final_seeds = self.conflict_resolution(S_pool, view_support)
        print(f"  冲突消解后种子对: {len(final_seeds)} 个")
    
        elapsed_time = time.time() - start_time
        print(f"多视角协同种子挖掘完成")
        print(f"用时: {elapsed_time:.2f}s")
        print(f"最终获得 {len(final_seeds)} 个高质量种子对")
        print("=" * 60)
        
        return final_seeds


def multi_view_bnns(entity1, entity2, semantic_emb,structural_emb, fusion_emb, train_pair, test_pair, alpha=0.5, beta=0.5):
               
    """
    多视角协同种子筛选的对外接口

    Args:
        entity1: 源实体索引
        entity2: 目标实体索引
        structural_emb: 结构嵌入
        semantic_emb: 语义嵌入
        fusion_emb: 融合嵌入
        train_pair: 训练种子对
        test_pair:测试种子对（用于案例分析）
        alpha: 语义权重
        beta: 结构权重
        
    Returns:
        筛选后的种子对列表
    """
    # 创建多视角协同筛选器
    selector = MultiViewCooperativeSeedSelection(
        semantic_emb=semantic_emb,
        structural_emb=structural_emb,
        fusion_emb = fusion_emb,
        alpha=alpha,
        beta=beta
    )
    
    # 执行多视角协同挖掘
    new_seedpairs = selector.multi_view_cooperative_mining(
        entity1, entity2, train_pair,test_pair
    )
    
    return new_seedpairs