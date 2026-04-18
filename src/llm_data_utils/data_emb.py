import sys
import openai
import time
from tqdm import tqdm
import asyncio
import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any
from llm_service.emb_serve import TextEmbedService

class EntityEmbeddingProcessor:
    """
    实体嵌入处理器
    """
    
    def __init__(self, embed_service: TextEmbedService):
        self.embed_service = embed_service
    
    def load_entities_from_json(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        从JSON文件加载实体数据
        """
        print(f"加载JSON文件: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        print(f"成功加载 {len(entities)} 个实体")
        return entities
    
    def extract_entity_texts(self, entities: List[Dict[str, Any]]) -> List[List[str]]:
        """
        提取每个实体的名称和别名文本
        """
        entity_texts = []
        # for entity in entities:
        #     # 获取实体名称和别名
        #     entity_name = entity.get('entity_name', '')
        #     aliases = entity.get('aliases', [])
            
        #     # 合并所有文本（实体名称 + 别名）
        #     texts = [entity_name] + aliases
        #     entity_texts.append(texts)

        for entity in entities:
            # 获取实体名称和别名
            entity_name = entity.get('entity_name', '')
            # aliases = entity.get('aliases', [])
            
            # 合并所有文本（实体名称）
            texts = [entity_name]
            entity_texts.append(texts)
        
        
        return entity_texts
    
    def extract_relation_descriptions(self, entities: List[Dict[str, Any]]) -> List[str]:
        """
        提取每个实体的关系描述文本
        """
        relation_descriptions = []
        for entity in entities:
            # 获取关系描述
            description = entity.get('relation_description', '')
            relation_descriptions.append(description)
        
        return relation_descriptions
    
    def extract_attribute_descriptions(self, entities: List[Dict[str, Any]]) -> List[str]:
        """
        提取每个实体的属性描述文本
        """
        attribute_descriptions = []
        for entity in entities:
            # 获取关系描述
            description = entity.get('attribute_description', '')
            attribute_descriptions.append(description)
        
        return attribute_descriptions
    
    async def generate_entity_embeddings(self, entity_texts: List[List[str]], batch_size: int = 100) -> List[np.ndarray]:
        """
        为每个实体生成嵌入向量
        """
        print("开始生成实体嵌入向量...")
        
        # 将所有文本展平用于批量处理
        all_texts = []
        text_to_entity_idx = []  # 记录每个文本属于哪个实体
        
        for entity_idx, texts in enumerate(entity_texts):
            for text in texts:
                if text.strip():  # 只处理非空文本
                    all_texts.append(text)
                    text_to_entity_idx.append(entity_idx)
        
        print(f"总共需要处理 {len(all_texts)} 个文本")
        
        # 批量处理所有文本的嵌入
        all_embeddings = []
        for i in tqdm(range(0, len(all_texts), batch_size), desc="生成嵌入向量"):
            batch_texts = all_texts[i:i+batch_size]
            batch_embeddings = await self.embed_service.text_embedding(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # 将嵌入向量按实体分组
        entity_embeddings = [[] for _ in range(len(entity_texts))]
        for text_idx, entity_idx in enumerate(text_to_entity_idx):
            if text_idx < len(all_embeddings):
                entity_embeddings[entity_idx].append(all_embeddings[text_idx])
        
        # 对每个实体的所有嵌入向量进行平均池化和归一化
        final_entity_embeddings = []
        for i, embeddings in tqdm(enumerate(entity_embeddings), desc="处理实体嵌入", total=len(entity_embeddings)):
            if embeddings:
                # 平均池化
                avg_embedding = np.mean(embeddings, axis=0)
                # L2归一化
                normalized_embedding = avg_embedding / np.linalg.norm(avg_embedding)
                final_entity_embeddings.append(normalized_embedding)
            else:
                # 如果没有有效的文本，使用零向量
                print(f"警告: 实体 {i} 没有有效的文本，使用零向量")
                final_entity_embeddings.append(np.zeros(1024))  # 假设维度为1024
        
        return final_entity_embeddings
    
    async def generate_relation_description_embeddings(self, relation_descriptions: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        为每个实体的关系描述生成嵌入向量
        注意：这里每个实体只有一个描述，直接嵌入并归一化，不需要平均
        """
        print("开始生成关系描述嵌入向量...")
        
        # 过滤空描述
        valid_descriptions = []
        valid_indices = []
        
        for idx, description in enumerate(relation_descriptions):
            if description and description.strip():
                valid_descriptions.append(description)
                valid_indices.append(idx)
        
        print(f"总共需要处理 {len(valid_descriptions)} 个有效关系描述")
        
        # 批量处理所有描述的嵌入
        all_embeddings = []
        for i in tqdm(range(0, len(valid_descriptions), batch_size), desc="生成关系描述嵌入"):
            batch_descriptions = valid_descriptions[i:i+batch_size]
            batch_embeddings = await self.embed_service.text_embedding(batch_descriptions)
            all_embeddings.extend(batch_embeddings)
        
        # 创建结果列表，对没有有效描述的实体使用零向量
        final_description_embeddings = []
        embedding_idx = 0
        
        for i in tqdm(range(len(relation_descriptions)), desc="处理关系描述嵌入"):
            if i in valid_indices and embedding_idx < len(all_embeddings):
                # 获取嵌入并归一化
                embedding = all_embeddings[embedding_idx]
                normalized_embedding = embedding / np.linalg.norm(embedding)
                final_description_embeddings.append(normalized_embedding)
                embedding_idx += 1
            else:
                # 如果没有有效的描述，使用零向量
                print(f"警告: 实体 {i} 没有有效的关系描述，使用零向量")
                final_description_embeddings.append(np.zeros(1024))  # 假设维度为1024
        
        return final_description_embeddings
    
    async def generate_attribute_description_embeddings(self, attribute_descriptions: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        为每个实体的属性描述生成嵌入向量
        注意：这里每个实体只有一个描述，直接嵌入并归一化，不需要平均
        """
        print("开始生成属性描述嵌入向量...")
        
        # 过滤空描述
        valid_descriptions = []
        valid_indices = []
        
        for idx, description in enumerate(attribute_descriptions):
            if description and description.strip():
                valid_descriptions.append(description)
                valid_indices.append(idx)
        
        print(f"总共需要处理 {len(valid_descriptions)} 个有效属性描述")
        
        # 批量处理所有描述的嵌入
        all_embeddings = []
        for i in tqdm(range(0, len(valid_descriptions), batch_size), desc="生成属性描述嵌入"):
            batch_descriptions = valid_descriptions[i:i+batch_size]
            batch_embeddings = await self.embed_service.text_embedding(batch_descriptions)
            all_embeddings.extend(batch_embeddings)
        
        # 创建结果列表，对没有有效描述的实体使用零向量
        final_description_embeddings = []
        embedding_idx = 0
        
        for i in tqdm(range(len(attribute_descriptions)), desc="处理属性描述嵌入"):
            if i in valid_indices and embedding_idx < len(all_embeddings):
                # 获取嵌入并归一化
                embedding = all_embeddings[embedding_idx]
                normalized_embedding = embedding / np.linalg.norm(embedding)
                final_description_embeddings.append(normalized_embedding)
                embedding_idx += 1
            else:
                # 如果没有有效的描述，使用零向量
                print(f"警告: 实体 {i} 没有有效的属性描述，使用零向量")
                final_description_embeddings.append(np.zeros(1024))  # 假设维度为1024
        
        return final_description_embeddings
    
    def save_embeddings_to_pkl(self, embeddings: List[np.ndarray], file_path: str):
        """
        将嵌入向量保存为pkl文件
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 保存为pkl文件
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        print(f"嵌入向量已保存到: {file_path}")
        print(f"总共保存了 {len(embeddings)} 个实体的嵌入向量")
        print(f"嵌入向量维度: {embeddings[0].shape if embeddings else '无'}")

async def main():
    """
    主函数
    """
    # 初始化服务
    embed_service = TextEmbedService()
    processor = EntityEmbeddingProcessor(embed_service)
    
    # 配置参数
    json_file_path = "XX.json"
    output_folder = "XX"
    batch_size = 200
    
    # 加载实体数据
    entities = processor.load_entities_from_json(json_file_path)
    
    # 选项1: 处理实体名称
    # print("\n" + "="*50)
    # print("处理实体名称...")
    # entity_texts = processor.extract_entity_texts(entities)
    # entity_embeddings = await processor.generate_entity_embeddings(entity_texts, batch_size)
    
    # # 保存实体名称嵌入
    # output_path = os.path.join(output_folder, "wd_llm_ent_emb_1024.pkl")
    # processor.save_embeddings_to_pkl(entity_embeddings, output_path)
    # # 打印一些统计信息
    # print("\n处理完成!")
    # print(f"处理了 {len(entities)} 个实体")
    # print(f"平均每个实体有 {np.mean([len(texts) for texts in entity_texts]):.2f} 个文本")



    # # # 选项2: 处理关系描述嵌入
    # print("\n" + "="*50)
    # print("处理关系描述嵌入...")
    # relation_descriptions = processor.extract_relation_descriptions(entities)
    # relation_embeddings = await processor.generate_relation_description_embeddings(relation_descriptions, batch_size)
    
    # # 保存关系描述嵌入
    # output_path = os.path.join(output_folder, "wd_llm_rel_emb_1024.pkl")
    # processor.save_embeddings_to_pkl(relation_embeddings, output_path)
    
    # # 打印一些统计信息
    # print("\n处理完成!")
    # print(f"处理了 {len(entities)} 个实体")

    # # 统计关系描述的有效性
    # valid_descriptions = sum(1 for desc in relation_descriptions if desc and desc.strip())
    # print(f"有效关系描述数量: {valid_descriptions}/{len(entities)}")



    # 选项3: 处理属性描述嵌入
    # print("\n" + "="*50)
    # print("处理属性描述嵌入...")
    # attribute_descriptions = processor.extract_attribute_descriptions(entities)
    # attribute_embeddings = await processor.generate_attribute_description_embeddings(attribute_descriptions, batch_size)
    
    # # 保存属性描述嵌入
    # output_path = os.path.join(output_folder, "wd_llm_att_emb_1024.pkl")
    # processor.save_embeddings_to_pkl(attribute_embeddings, output_path)
    
    # # 打印一些统计信息
    # print("\n处理完成!")
    # # 统计属性描述的有效性
    # valid_descriptions = sum(1 for desc in attribute_descriptions if desc and desc.strip())
    # print(f"有效属性描述数量: {valid_descriptions}/{len(entities)}")

if __name__ == "__main__":
    # 运行主函数
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f'\n总耗时: {(end_time - start_time):.2f} 秒')