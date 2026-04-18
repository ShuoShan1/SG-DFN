import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
import argparse
import time

from src.base.base_utils import set_device
from src.base.data_kg_loader import KGs
from src.base.data_llm_emb_loader import LLMSemEmbeddingLoader
from src.base.hign_neighbor import HighOrderNeighborsLoader
from src.model_utils.gnn_model import Encoder_Model
from src.seed_utils.multi_seed_select import multi_view_bnns
from src.eval_utils.evals import CSLS_evaluate


def train_base(args,train_pairs,valid_pair,test_pair,model:Encoder_Model):
    emb_reset_flag = 1
    total_train_time = 0.0

    for epoch in range(args.epoch):
        time1 = time.time()
        total_loss = 0
        np.random.shuffle(train_pairs)
        batch_num = len(train_pairs) // args.batch_size + 1
        model.train()
        for b in range(batch_num):
            pairs = train_pairs[b * args.batch_size:(b + 1) * args.batch_size]
            if len(pairs) == 0:
                continue
            pairs = torch.from_numpy(pairs).to(device)
            optimizer.zero_grad()
            loss = model(pairs, emb_reset_flag)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
        time2 = time.time()
        total_train_time += time2 - time1
        print(f'[epoch {epoch + 1}/{args.epoch}]  epoch loss: {(total_loss):.3f}, time cost: {(time2 - time1):.3f}s')
        emb_reset_flag = 1
    

        # -----------test-----------
        if (epoch + 1) % args.round in [0,args.round-1,args.round-2,args.round-3] or (epoch + 1) % 5 == 0:
            print("-----------开始测试-----------")
            model.eval()
            with torch.no_grad():
                Lvec, Rvec,out_feature,str_feature,sem_feature = model.get_embeddings(test_pair[:, 0], test_pair[:, 1])
                out_feature, result_dict = CSLS_evaluate(test_pair,Lvec, Rvec,out_feature)

        # -----------Iteration-----------
        if (epoch + 1) in [args.round, args.round*2, args.round*3,args.round*4,args.round*5]:
            print("-----------开始种子优化筛选-----------")
            opt_seedpairs_fusion = multi_view_bnns(entity1, entity2, sem_feature,str_feature,out_feature,train_pair,test_pair) # 多视角协同的种子筛选
            train_pairs = np.concatenate((train_pair, opt_seedpairs_fusion))
            emb_reset_flag = 0
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='alignment model')

    # ----------数据----------
    parser.add_argument('--log_path', default=str(PROJECT_ROOT / 'logs'), type=str)
    parser.add_argument('--project_path', default=str(PROJECT_ROOT), type=str)
    parser.add_argument('--dataset_folder', default=str(PROJECT_ROOT / 'datasets'), type=str)
    parser.add_argument('--emb_data_folder', default=str(PROJECT_ROOT / 'entity_emb'), type=str)
    parser.add_argument('--preload_cache_dir', default=str(PROJECT_ROOT / 'cache_data'), type=str)
    parser.add_argument('--dataset', default='DBP15K', type=str)
    parser.add_argument('--language', default='zh_en', type=str)  # zh_en en_fr en_de
    # ----------模型----------
    parser.add_argument('--ent_hidden', default=100, type=int)
    parser.add_argument('--rel_hidden', default=100, type=int)
    parser.add_argument('--att_hidden', default=100, type=int)
    parser.add_argument('--topk', default=15, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)  
    parser.add_argument('--depth', default=2, type=int)

    # ----------训练----------
    parser.add_argument('--semi_supervise', default=True, type=bool)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--rates', default=[3, 0, 7], type=list)
    parser.add_argument('--start_windows', default=0.0, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--round', default=30, type=int)# 种子优化轮次间隙
    parser.add_argument('--gpu', default=7, type=int)


    args = parser.parse_args()
    device = set_device(args.gpu)

    def seed_everything(seed_value):
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    seed_everything(args.seed)

    print("-----------加载图谱结构信息-----------")
    start_time = time.time()
    kgs = KGs(args)
    base_data,dataset_data,matrix_data = kgs.load_kg_data(args)
    train_pair, valid_pair, test_pair,ill_ent = dataset_data[0],dataset_data[1],dataset_data[2],dataset_data[3]
    entity1,entity2,triples1,triples2 = base_data[0],base_data[1],base_data[2],base_data[3],
    ent_adj, r_index, r_val, ent_adj_with_loop, ent_rel_adj,ent_att_adj = matrix_data[0],matrix_data[1],matrix_data[2],matrix_data[3],matrix_data[4],matrix_data[5]
    
    ent_adj = torch.from_numpy(np.transpose(ent_adj))
    ent_rel_adj = torch.from_numpy(np.transpose(ent_rel_adj))
    ent_att_adj = torch.from_numpy(np.transpose(ent_att_adj))
    ent_adj_with_loop = torch.from_numpy(np.transpose(ent_adj_with_loop))
    r_index = torch.from_numpy(np.transpose(r_index))
    r_val = torch.from_numpy(r_val)
    end_time = time.time()
    print(f"加载图谱结构信息用时：{(end_time - start_time):.3f}s")


    print("-----------加载语义嵌入信息-----------")
    start_time = time.time()
    emb_loader = LLMSemEmbeddingLoader(args.dataset_folder,args.emb_data_folder,args.dataset,args.language,args.preload_cache_dir)
    ent_semantic_emb = emb_loader.load_entity_embeddings()
    rel_semantic_emb = emb_loader.load_relation_embeddings()
    att_semantic_emb = emb_loader.load_attribute_embeddings()
    end_time = time.time()
    print(f"加载语义嵌入信息用时：{(end_time - start_time):.3f}s")
   

    print("-----------构建高阶邻居-----------")
    start_time = time.time()
    high_order_loader = HighOrderNeighborsLoader(args.dataset_folder,args.emb_data_folder,args.dataset,args.language,args.preload_cache_dir)
    high_adj = high_order_loader.compute_high_order_neighbors(topk=args.topk,use_cache=True)
    end_time = time.time()
    print(f"构建高阶邻居用时：{(end_time - start_time):.3f}s")

    print("-----------加载模型-----------")
    model = Encoder_Model(node_hidden=args.ent_hidden,rel_hidden=args.rel_hidden,att_hidden=args.att_hidden,triple_size=kgs.triple_num,
                          node_size=kgs.total_ent_num,rel_size=kgs.total_rel_num,att_size = kgs.total_att_num,
                          ent_semantic_emb = ent_semantic_emb,rel_semantic_emb = rel_semantic_emb,att_semantic_emb = att_semantic_emb,
                          device=device,adj_matrix=ent_adj,r_index=r_index,r_val=r_val,
                          rel_matrix=ent_rel_adj,att_matrix=ent_att_adj,ent_matrix=ent_adj_with_loop,
                          ill_ent=ill_ent,dropout_rate=args.dropout_rate,lr=args.lr,
                          depth=args.depth,high_adj = high_adj  
                          ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    print("-----------开始训练-----------")
    if args.semi_supervise:
        train_base(args, train_pair, valid_pair,test_pair, model)
      
