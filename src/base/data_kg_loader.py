import numpy as np
import scipy.sparse as sp
import os
import time
from src.base.data_att_loader import get_all_atts
from collections import Counter
import random
class KGs:
    # data_path ：最外围数据集路径
    def __init__(self, args):

        self.data_path = args.dataset_folder 
        # self.att_threshold = args.att_threshold
        self.entity1, self.rel1, self.triples1 = None, None, None
        self.entity2, self.rel2, self.triples2 = None, None, None
        self.train_pairs, self.valid_pairs, self.test_pairs,self.ill_ent = None, None, None, None
        
        
        # ----------下面的暂时还未填充相关信息----------
        self.att_triples, self.att = None, None
        # 新实体集合，锚点实体集合，
        self.new_ent = set()
        self.anchors = None
        # 测试对列表
        self.new_test_pairs = []
        # 总实体数、总关系数和总属性数
        self.total_ent_num, self.total_rel_num, self.total_att_num = 0, 0, 0
        # 初始化三元组总数
        self.triple_num = 0
        # 初始化可信对
        self.credible_pairs = None
        # 初始化新实体的邻居数组
        self.new_ent_nei = np.array([])
        
        
    # 连接数据集文件夹路径以及具体的数据集类型信息
    def _get_file_path(self, filename):
        return os.path.join(self.data_path, filename)


    # 加载三元组数据
    def load_triples(self, filename):
   
        triples = []
        entity = set()
        rel = {0}  # 初始化关系集合，包含自连接关系
        file_path = self._get_file_path(filename)
        with open(file_path, 'r') as f:
            for line in f:
                head, r, tail = [int(item) for item in line.split()]
                entity.add(head)
                entity.add(tail)
                rel.add(r + 1)  # 关系编号加1
                triples.append((head, r + 1, tail))
        return entity, rel, triples


    # 加载全部实体对
    def load_alignment_pair(self, filename):
        alignment_pair = []
        file_path = self._get_file_path(filename)
        with open(file_path, 'r') as f:
            for line in f:
                e1, e2 = line.split()
                alignment_pair.append((int(e1), int(e2)))
        return alignment_pair


    # K折验证
    def sliding_window(self, ill_ent, start, rates):
        # 通过滑动窗口机制，实现对数据集的k折交叉验证
        # 从总数据长度的 start 开始进行数据集的切分
        
        total_length = len(ill_ent)
        total = sum(rates)
        rates = np.array(rates) / total  # 将比例转换为小数形式
        window_size = int(round(start * total_length))
        new_ill_ent = ill_ent[window_size:] + ill_ent[:window_size]  # 将数据集尾部和头部连接起来

        train_data_size = int(round(rates[0] * total_length))
        train_data = new_ill_ent[0 : train_data_size]
        valid_data_size = int(round(rates[1] * total_length))
        valid_data = new_ill_ent[train_data_size : train_data_size + valid_data_size]
        test_data_size = int(round(rates[2] * total_length))
        test_data = new_ill_ent[train_data_size + valid_data_size:train_data_size + valid_data_size + test_data_size]
        
        return train_data, valid_data, test_data


    # 构建矩阵
    def get_matrix(self, triples, entity, rel,att_triples,att):
        """
        构建邻接矩阵，关系矩阵，属性矩阵。

        :param triples: 知识图谱中的三元组列表。
        :param entity: 实体集合。
        :param rel: 关系集合。
        :return: 邻接矩阵，关系索引矩阵，关系值矩阵，邻接特征矩阵，关系特征矩阵等
        """
        ent_size = max(entity) + 1
        rel_size = max(rel) + 1
        att_size = max(att) + 1
        print(f"实体：{ent_size}, 关系：{rel_size}, 属性：{att_size}")
        
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))
        rel_in = np.zeros((ent_size, rel_size))
        rel_out = np.zeros((ent_size, rel_size))
        att_features = np.zeros((ent_size, att_size))
        radj = []
        
        # 添加自连接
        for i in range(ent_size):
            adj_features[i, i] = 1
  

        # 构建邻接矩阵和关系矩阵
        for h, r, t in triples:
            adj_matrix[h, t] = 1
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1
            adj_features[t, h] = 1
            rel_out[h][r] += 1
            rel_in[t][r] += 1
            radj.append([h, t, r])
            radj.append([t, h, r + rel_size])
        
        # ---------------构建属性---------------
     
        for entity,att_key,att_value in att_triples:
            att_features[entity][att_key] += 1
        
        # 构建关系索引和值
        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
            if ' '.join([str(h), str(t)]) in s:
                r_index.append([count, r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add(' '.join([str(h), str(t)]))
                r_index.append([count, r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]

        rel_features = np.concatenate([rel_in, rel_out], axis=1)
        rel_features = sp.lil_matrix(rel_features)
        return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features,att_features


    # 加载基础数据，包括实体、关系、三元组、切分后的数据集
    def load_base_data(self, args):
        path = self._get_file_path(f'{args.dataset}/{args.language}/')
        self.entity1, self.rel1, self.triples1 = self.load_triples(path + 'triples_1')
        self.entity2, self.rel2, self.triples2 = self.load_triples(path + 'triples_2')
        self.ill_ent = self.load_alignment_pair(path + 'ref_ent_ids')    
        # random.shuffle(self.ill_ent)
        self.total_ent_num = len(list(self.entity1)) + len(list(self.entity2))
        alignment_pairs = self.ill_ent
        self.train_pairs, self.valid_pairs, self.test_pairs = self.sliding_window(alignment_pairs, args.start_windows, args.rates)
        
        # 添加属性信息
        self.att_features = []
        self.att_triplrs_zh,self.att_triplrs_en,self.att_value_zh,self.att_value_en,self.att= get_all_atts(args)
        
        
    # 加载邻接矩阵，关系矩阵,属性矩阵等数据
    def load_matrix_data(self):
        adj_matrix, r_index, r_val, adj_features, rel_features,att_features = self.get_matrix(self.triples1 + self.triples2,self.entity1.union(self.entity2),
                                                                                 self.rel1.union(self.rel2),self.att_triplrs_zh+self.att_triplrs_en,self.att)
        
        ent_adj = np.stack(adj_matrix.nonzero(), axis=1)
        ent_adj_with_loop = np.stack(adj_features.nonzero(), axis=1)
        ent_rel_adj = np.stack(rel_features.nonzero(), axis=1)
        ent_att_adj = np.stack(att_features.nonzero(), axis=1)
        
        self.total_ent_num = adj_features.shape[0]
        self.total_rel_num = rel_features.shape[1] 
        self.total_att_num = att_features.shape[1]
        self.triple_num = ent_adj.shape[0]

        return ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,ent_att_adj
    
    def get_att_info(self, args):
        self.att_triplrs_zh,self.att_triplrs_en,self.att_value_zh,self.att_value_en,self.att= get_all_atts(args)
        return self.att_triplrs_zh,self.att_triplrs_en,self.att_value_zh,self.att_value_en,self.att
    
    def load_kg_data(self,args):
        self.load_base_data(args)
        ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,ent_att_adj = self.load_matrix_data()
        
        
        return [list(self.entity1),list(self.entity2),self.triples1,self.triples2],\
                [np.array(self.train_pairs), np.array(self.valid_pairs), np.array(self.test_pairs), np.array(self.ill_ent)],\
               [ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,ent_att_adj]
