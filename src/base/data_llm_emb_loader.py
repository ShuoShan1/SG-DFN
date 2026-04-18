import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import os
import time


@dataclass
class DatasetConfig:
    """数据集配置"""
    dataset_name: str  # DBP15K, SRPRS, DWY100K
    language: str      # zh_en, ja_en, fr_en, en_de, en_fr, ...
    
    @property
    def lang_pair(self) -> Tuple[str, str]:
        """返回语言对 (lang1, lang2)"""
        langs = self.language.split('_')
        return (langs[0], langs[1])


class EmbeddingPathManager:
    """嵌入文件路径管理器"""
    
    def __init__(self, dataset_folder: str, emb_data_folder: str, config: DatasetConfig):
        self.dataset_folder = Path(dataset_folder)
        self.emb_data_folder = Path(emb_data_folder)
        self.config = config
        
        # =============== 确定基础路径 ===============
        if config.dataset_name == "DBP15K":
            self.base_emb_path = self.emb_data_folder / "DBP15K" / config.language
            self.base_dataset_path = self.dataset_folder / "DBP15K" / config.language
        elif config.dataset_name == "SRPRS":
            self.base_emb_path = self.emb_data_folder / "SRPRS" / config.language
            self.base_dataset_path = self.dataset_folder / "SRPRS" / config.language
        elif config.dataset_name == "DWY100K":
            self.base_emb_path = self.emb_data_folder / "DWY100K" / config.language
            self.base_dataset_path = self.dataset_folder / "DWY100K" / config.language
        else:
            raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    
    # =============== 获取处理后的实体嵌入文件路径 ===============
    def get_entity_emb_paths(self):
        lang1, lang2 = self.config.lang_pair
        return {
            'kg1': self.base_emb_path / f"{lang1}_llm_ent_emb_4096.pkl",
            'kg2': self.base_emb_path / f"{lang2}_llm_ent_emb_4096.pkl"
        }
    
    def get_relation_emb_paths(self):
        lang1, lang2 = self.config.lang_pair
        return {
            'kg1': self.base_emb_path / f"{lang1}_llm_rel_emb_4096.pkl",
            'kg2': self.base_emb_path / f"{lang2}_llm_rel_emb_4096.pkl"
        }
    
    def get_attribute_emb_paths(self):
        lang1, lang2 = self.config.lang_pair
        return {'kg1': self.base_emb_path / f"{lang1}_llm_att_emb_4096.pkl",
            'kg2': self.base_emb_path / f"{lang2}_llm_att_emb_4096.pkl"
        }


    # =============== 获取实体ID映射文件路径 ===============
    def get_entity_id_paths(self):
        return {
            'kg1': self.base_dataset_path / "ent_ids_1",
            'kg2': self.base_dataset_path / "ent_ids_2"
        }
    
    # =============== 获取缓存目录 ===============
    def get_cache_dir(self, cache_root):
        if cache_root:
            cache_dir = Path(cache_root) / self.config.dataset_name / self.config.language
        else:
            cache_dir = self.base_dataset_path / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir




class EmbeddingLoader:
    """嵌入加载基础类"""
    
    @staticmethod
    def load_pickle(file_path: Path) -> List:
        """加载pickle文件"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def load_entity_ids(file_path: Path) -> List[int]:
        """加载实体ID映射"""
        ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ids.append(int(parts[0]))
        return ids
    
    @staticmethod
    def list_to_tensor(data: List) -> torch.Tensor:
        if not data:
            return torch.empty(0)

        if isinstance(data[0], torch.Tensor):
            return torch.stack([t.cpu() if t.is_cuda else t for t in data])

        if isinstance(data[0], list):
            return torch.tensor(data, dtype=torch.float32)

        if isinstance(data[0], np.ndarray):
            return torch.from_numpy(np.array(data)).float()
        
        return torch.tensor(data, dtype=torch.float32)


class CacheManager:

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.metadata_file = cache_dir / "cache_metadata.pkl"
    
    def get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pt"
    
    def is_cache_valid(self, cache_key: str, source_files: List[Path]) -> bool:
        cache_file = self.get_cache_path(cache_key)
        
        if not cache_file.exists():
            return False
        
        if not self.metadata_file.exists():
            return False
        
        try:
            metadata = self.load_metadata()
            if cache_key not in metadata:
                return False
            
            cache_info = metadata[cache_key]
            
            for src_file in source_files:
                if not src_file.exists():
                    return False
                
                src_mtime = src_file.stat().st_mtime
                cached_mtime = cache_info['source_mtimes'].get(str(src_file), 0)
                
                if src_mtime > cached_mtime:
                    return False
            
            return True
        except Exception as e:
            print(f"缓存验证失败: {e}")
            return False
    
    def save_cache(self, cache_key: str, data: torch.Tensor, source_files: List[Path]):
        cache_file = self.get_cache_path(cache_key)
        torch.save(data, cache_file)
        
        metadata = self.load_metadata()
        metadata[cache_key] = {
            'created_time': time.time(),
            'source_mtimes': {str(f): f.stat().st_mtime for f in source_files if f.exists()},
            'shape': list(data.shape)
        }
        self.save_metadata(metadata)
    
    def load_cache(self, cache_key: str) -> torch.Tensor:
        cache_file = self.get_cache_path(cache_key)
        return torch.load(cache_file, map_location='cpu')
    
    def load_metadata(self) -> Dict:
        if not self.metadata_file.exists():
            return {}
        
        with open(self.metadata_file, 'rb') as f:
            return pickle.load(f)
    
    def save_metadata(self, metadata: Dict):
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(metadata, f)


class LLMSemEmbeddingLoader:

    def __init__(
        self,dataset_folder: str,emb_data_folder: str,dataset_name: str,language: str,cache_dir: Optional[str] = None):
        self.config = DatasetConfig(dataset_name, language)
        self.path_manager = EmbeddingPathManager(dataset_folder, emb_data_folder, self.config)
        self.loader = EmbeddingLoader()
        cache_path = self.path_manager.get_cache_dir(cache_dir)
        self.cache_manager = CacheManager(cache_path)
        print(f"初始化嵌入加载:")
        print(f"  数据集: {dataset_name}\n  语言对: {language}\n  缓存目录: {cache_path}")


    def _merge_kg_embeddings(
        self,kg1_ids: List[int],kg2_ids: List[int],kg1_emb: torch.Tensor,kg2_emb: torch.Tensor,normalize: bool = True) -> torch.Tensor:
    
        # 确定总实体数（最大ID+1）
        max_id = max(max(kg1_ids) if kg1_ids else 0, max(kg2_ids) if kg2_ids else 0)
        total_entities = max_id + 1
        
        emb_dim = kg1_emb.shape[1]
        merged_emb = torch.zeros(total_entities, emb_dim, dtype=torch.float32)
        
        # 填充KG1的嵌入到对应的ID位置
        if kg1_ids:
            for i, entity_id in enumerate(kg1_ids):
                if i < len(kg1_emb):  # 确保不越界
                    merged_emb[entity_id] = kg1_emb[i]
        
        # 填充KG2的嵌入到对应的ID位置
        if kg2_ids:
            for i, entity_id in enumerate(kg2_ids):
                if i < len(kg2_emb):  # 确保不越界
                    merged_emb[entity_id] = kg2_emb[i]
        
        # 归一化（如果需要）
        if normalize:
            norms = torch.linalg.norm(merged_emb, dim=-1, keepdim=True)
            norms = torch.clamp(norms, min=1e-5)
            merged_emb = merged_emb / norms
        
        return merged_emb
    
    def load_entity_embeddings(self, use_cache: bool = True) -> torch.Tensor:
        """加载实体嵌入"""
        cache_key = "entity_embeddings"
        
        # 获取文件路径
        emb_paths = self.path_manager.get_entity_emb_paths()
        id_paths = self.path_manager.get_entity_id_paths()
        source_files = list(emb_paths.values()) + list(id_paths.values())
        
        # 检查缓存
        if use_cache and self.cache_manager.is_cache_valid(cache_key, source_files):
            print("从缓存加载实体嵌入...")
            return self.cache_manager.load_cache(cache_key)
        
        print("生成实体嵌入...")
        start_time = time.time()
        
        # 加载数据
        kg1_ids = self.loader.load_entity_ids(id_paths['kg1'])
        kg2_ids = self.loader.load_entity_ids(id_paths['kg2'])
        
        kg1_emb_list = self.loader.load_pickle(emb_paths['kg1'])
        kg2_emb_list = self.loader.load_pickle(emb_paths['kg2'])
        
        kg1_emb = self.loader.list_to_tensor(kg1_emb_list)
        kg2_emb = self.loader.list_to_tensor(kg2_emb_list)
        
        print(f"  KG1: {len(kg1_ids)} 个实体, 嵌入维度: {kg1_emb.shape}")
        print(f"  KG2: {len(kg2_ids)} 个实体, 嵌入维度: {kg2_emb.shape}")
        
        # 合并嵌入（根据实际ID位置填充）
        merged_emb = self._merge_kg_embeddings(kg1_ids, kg2_ids, kg1_emb, kg2_emb)
        
        # 保存缓存
        if use_cache:
            self.cache_manager.save_cache(cache_key, merged_emb, source_files)
        
        elapsed = time.time() - start_time
        print(f"实体嵌入加载完成: shape={merged_emb.shape}, 耗时={elapsed:.3f}s")
        
        return merged_emb
    
    def load_relation_embeddings(self, use_cache: bool = True) -> torch.Tensor:
        """加载关系嵌入"""
        cache_key = "relation_embeddings"
        
        # 获取文件路径
        emb_paths = self.path_manager.get_relation_emb_paths()
        source_files = list(emb_paths.values())
        
        # 检查缓存
        if use_cache and self.cache_manager.is_cache_valid(cache_key, source_files):
            print("从缓存加载关系嵌入...")
            return self.cache_manager.load_cache(cache_key)
        
        print("生成关系嵌入...")
        start_time = time.time()
        
        # 加载数据
        kg1_emb_list = self.loader.load_pickle(emb_paths['kg1'])
        kg2_emb_list = self.loader.load_pickle(emb_paths['kg2'])
        
        kg1_emb = self.loader.list_to_tensor(kg1_emb_list)
        kg2_emb = self.loader.list_to_tensor(kg2_emb_list)
        
        print(f"  KG1: {kg1_emb.shape[0]} 个关系, 嵌入维度: {kg1_emb.shape[1]}")
        print(f"  KG2: {kg2_emb.shape[0]} 个关系, 嵌入维度: {kg2_emb.shape[1]}")
       

        # 需要加载关系ID映射来正确排序
        id_paths = self.path_manager.get_entity_id_paths()
        kg1_ids = self.loader.load_entity_ids(id_paths['kg1'])
        kg2_ids = self.loader.load_entity_ids(id_paths['kg2'])
        
        # 确保嵌入数量与ID数量匹配
        if len(kg1_emb) != len(kg1_ids):
            print(f"  警告: KG1关系嵌入数量({len(kg1_emb)})与实体数量({len(kg1_ids)})不匹配")
            # 如果不匹配，假设属性嵌入是按顺序的，需要调整
            if len(kg1_emb) < len(kg1_ids):
                # 补零
                padding = torch.zeros(len(kg1_ids) - len(kg1_emb), kg1_emb.shape[1])
                kg1_emb = torch.cat([kg1_emb, padding], dim=0)
            else:
                # 截断
                kg1_emb = kg1_emb[:len(kg1_ids)]
        
        if len(kg2_emb) != len(kg2_ids):
            print(f"  警告: KG2关系嵌入数量({len(kg2_emb)})与实体数量({len(kg2_ids)})不匹配")
            if len(kg2_emb) < len(kg2_ids):
                padding = torch.zeros(len(kg2_ids) - len(kg2_emb), kg2_emb.shape[1])
                kg2_emb = torch.cat([kg2_emb, padding], dim=0)
            else:
                kg2_emb = kg2_emb[:len(kg2_ids)]
        
        # 按实体ID位置合并
        combined_emb = self._merge_kg_embeddings(kg1_ids, kg2_ids, kg1_emb, kg2_emb, normalize=False)


        # 保存缓存
        if use_cache:
            self.cache_manager.save_cache(cache_key, combined_emb, source_files)
        
        elapsed = time.time() - start_time
        print(f"关系嵌入加载完成: shape={combined_emb.shape}, 耗时={elapsed:.3f}s")
        
        return combined_emb
    
    def _load_relation_ids(self, file_path: Path) -> List[int]:
        """加载关系ID映射"""
        ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ids.append(int(parts[0]))
        return ids
    
    def load_attribute_embeddings(self, use_cache: bool = True) -> torch.Tensor:
        """加载属性嵌入"""
        cache_key = "attribute_embeddings"
        
        # 获取文件路径
        emb_paths = self.path_manager.get_attribute_emb_paths()
        source_files = list(emb_paths.values())
        
        # 检查缓存
        if use_cache and self.cache_manager.is_cache_valid(cache_key, source_files):
            print("从缓存加载属性嵌入...")
            return self.cache_manager.load_cache(cache_key)
        
        print("生成属性嵌入...")
        start_time = time.time()
        
        # 加载数据
        kg1_emb_list = self.loader.load_pickle(emb_paths['kg1'])
        kg2_emb_list = self.loader.load_pickle(emb_paths['kg2'])
        
        kg1_emb = self.loader.list_to_tensor(kg1_emb_list)
        kg2_emb = self.loader.list_to_tensor(kg2_emb_list)
        
        print(f"  KG1: {kg1_emb.shape[0]} 个属性, 嵌入维度: {kg1_emb.shape[1]}")
        print(f"  KG2: {kg2_emb.shape[0]} 个属性, 嵌入维度: {kg2_emb.shape[1]}")
        
        id_paths = self.path_manager.get_entity_id_paths()
        kg1_ids = self.loader.load_entity_ids(id_paths['kg1'])
        kg2_ids = self.loader.load_entity_ids(id_paths['kg2'])
        
        # 确保嵌入数量与ID数量匹配
        if len(kg1_emb) != len(kg1_ids):
            print(f"  警告: KG1属性嵌入数量({len(kg1_emb)})与实体数量({len(kg1_ids)})不匹配")
            # 如果不匹配，假设属性嵌入是按顺序的，需要调整
            if len(kg1_emb) < len(kg1_ids):
                # 补零
                padding = torch.zeros(len(kg1_ids) - len(kg1_emb), kg1_emb.shape[1])
                kg1_emb = torch.cat([kg1_emb, padding], dim=0)
            else:
                # 截断
                kg1_emb = kg1_emb[:len(kg1_ids)]
        
        if len(kg2_emb) != len(kg2_ids):
            print(f"  警告: KG2属性嵌入数量({len(kg2_emb)})与实体数量({len(kg2_ids)})不匹配")
            if len(kg2_emb) < len(kg2_ids):
                padding = torch.zeros(len(kg2_ids) - len(kg2_emb), kg2_emb.shape[1])
                kg2_emb = torch.cat([kg2_emb, padding], dim=0)
            else:
                kg2_emb = kg2_emb[:len(kg2_ids)]
        
        # 按实体ID位置合并
        combined_emb = self._merge_kg_embeddings(kg1_ids, kg2_ids, kg1_emb, kg2_emb, normalize=False)
        
        # 保存缓存
        if use_cache:
            self.cache_manager.save_cache(cache_key, combined_emb, source_files)
        
        elapsed = time.time() - start_time
        print(f"属性嵌入加载完成: shape={combined_emb.shape}, 耗时={elapsed:.3f}s")
        
        return combined_emb
    
    def preload_all(self, force_rebuild: bool = False):
        """预加载所有嵌入到缓存"""
        print("=" * 60)
        print("开始预加载所有嵌入数据")
        print("=" * 60)
        
        results = {}
        use_cache = not force_rebuild
        
        results['entity'] = self.load_entity_embeddings(use_cache=use_cache)
        results['relation'] = self.load_relation_embeddings(use_cache=use_cache)
        results['attribute'] = self.load_attribute_embeddings(use_cache=use_cache)
        
        print("=" * 60)
        print("所有嵌入数据预加载完成")
        print("=" * 60)
        
        return results
