import torch
import numpy as np
import pickle
import time
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class HighOrderConfig:
    """高阶邻居配置"""
    dataset_name: str  # DBP15K, SRPRS
    language: str      # zh_en, ja_en, fr_en, en_de, en_fr
    
    @property
    def lang_pair(self) -> Tuple[str, str]:
        """返回语言对 (lang1, lang2)"""
        langs = self.language.split('_')
        return (langs[0], langs[1])


class HighOrderPathManager:
    """高阶邻居路径管理器"""
    
    def __init__(self, dataset_folder: str, emb_data_folder: str, config: HighOrderConfig):
        self.dataset_folder = Path(dataset_folder)
        self.emb_data_folder = Path(emb_data_folder)
        self.config = config
        
        # 确定基础路径
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
    
    def get_entity_id_paths(self) -> dict:
        return {
            'kg1': self.base_dataset_path / "ent_ids_1",
            'kg2': self.base_dataset_path / "ent_ids_2"
        }
    
    def get_entity_emb_paths(self) -> dict:
        lang1, lang2 = self.config.lang_pair
        return {
            'kg1': self.base_emb_path / f"{lang1}_llm_ent_emb_4096.pkl",#_llm_ent_emb_4096  _tra_ent_emb  
            'kg2': self.base_emb_path / f"{lang2}_llm_ent_emb_4096.pkl"
        }
    
    def get_cache_dir(self, cache_root: Optional[str] = None) -> Path:
        """获取缓存目录"""
        if cache_root:
            cache_dir = Path(cache_root) / self.config.dataset_name / self.config.language
        else:
            cache_dir = self.base_dataset_path / "high_order_cache"
        
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


class HighOrderCacheManager:
    """高阶邻居缓存管理器"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
    
    def get_cache_path(self, topk: int) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"high_order_neighbors_topk{topk}.pt"
    
    def get_metadata_path(self, topk: int) -> Path:
        """获取元数据文件路径"""
        return self.cache_dir / f"high_order_metadata_topk{topk}.pkl"
    
    def compute_source_hash(self, source_files: list, topk: int) -> str:
        """计算源文件哈希值"""
        hash_data = []
        
        for path in source_files:
            if path.exists():
                stat = path.stat()
                hash_data.append(f"{path}_{stat.st_mtime}_{stat.st_size}")
            else:
                hash_data.append(f"{path}_missing")
        
        hash_data.append(f"topk_{topk}")
        hash_string = "".join(sorted(hash_data))
        return hashlib.md5(hash_string.encode()).hexdigest()
    
    def is_cache_valid(self, topk: int, source_files: list) -> bool:
        """检查缓存是否有效"""
        cache_path = self.get_cache_path(topk)
        metadata_path = self.get_metadata_path(topk)
        
        if not cache_path.exists() or not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            current_hash = self.compute_source_hash(source_files, topk)
            return metadata.get('source_hash') == current_hash
        except Exception as e:
            print(f"缓存验证失败: {e}")
            return False
    
    def save_cache(self, high_adj: torch.Tensor, topk: int, source_files: list, metadata_extra: dict = None):
        """保存缓存"""
        cache_path = self.get_cache_path(topk)
        metadata_path = self.get_metadata_path(topk)
        
        # 保存邻接矩阵
        torch.save(high_adj, cache_path)
        
        # 保存元数据
        metadata = {
            'source_hash': self.compute_source_hash(source_files, topk),
            'created_time': time.time(),
            'topk': topk,
            'shape': list(high_adj.shape)
        }
        
        if metadata_extra:
            metadata.update(metadata_extra)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"高阶邻居结果已保存到: {cache_path}")
        print(f"邻接矩阵形状: {high_adj.shape}")
    
    def load_cache(self, topk: int) -> Optional[torch.Tensor]:
        """加载缓存"""
        cache_path = self.get_cache_path(topk)
        
        if cache_path.exists():
            return torch.load(cache_path, map_location='cpu')
        return None


class HighOrderNeighborComputer:
    """高阶邻居计算器"""
    
    @staticmethod
    def load_entity_ids(file_path: Path) -> list:
        """加载实体ID映射"""
        ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    ids.append(int(parts[0]))
        return ids
    
    @staticmethod
    def load_embeddings(file_path: Path) -> torch.Tensor:
        """加载嵌入文件（支持多种格式）"""
        with open(file_path, 'rb') as f:
            emb_data = pickle.load(f)
        
        # 如果已经是 torch.Tensor
        if isinstance(emb_data, torch.Tensor):
            return emb_data.cpu().float()
        
        # 如果是 numpy array
        if isinstance(emb_data, np.ndarray):
            return torch.from_numpy(emb_data).float()
        
        # 如果是 list
        if isinstance(emb_data, list):
            # 情况1: 纯数值 list，如 [[1.0, 2.0], [3.0, 4.0]]
            if len(emb_data) == 0:
                return torch.empty(0, 0, dtype=torch.float32)
            
            first_elem = emb_data[0]
            
            # 情况2: 元素是 torch.Tensor（可能是错误保存的）
            if isinstance(first_elem, torch.Tensor):
                # 确保所有张量形状一致
                try:
                    return torch.stack(emb_data, dim=0).float()
                except RuntimeError as e:
                    raise ValueError(f"无法 stack 张量列表，可能形状不一致: {e}")
            
            # 情况3: 元素是 numpy array
            if isinstance(first_elem, np.ndarray):
                try:
                    return torch.from_numpy(np.stack(emb_data, axis=0)).float()
                except ValueError as e:
                    raise ValueError(f"无法 stack numpy 数组列表: {e}")
            
            # 情况4: 元素是纯数值（int/float）或 list[float]
            # 尝试直接构造
            try:
                return torch.tensor(emb_data, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"无法从列表构造张量，数据格式可能不规则: {type(first_elem)}, error: {e}")
        
        # 其他类型不支持
        raise ValueError(f"Unsupported embedding format: {type(emb_data)}")
    
    @staticmethod
    def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """计算余弦相似度矩阵"""
        # 归一化
        emb1_norm = emb1 / (torch.linalg.norm(emb1, dim=-1, keepdim=True) + 1e-5)
        emb2_norm = emb2 / (torch.linalg.norm(emb2, dim=-1, keepdim=True) + 1e-5)
        
        # 计算相似度
        similarity = torch.mm(emb1_norm, emb2_norm.t())
        return similarity
    
    @staticmethod
    def build_high_order_adjacency(
        kg1_ids: list,
        kg2_ids: list,
        kg1_emb: torch.Tensor,
        kg2_emb: torch.Tensor,
        topk: int
    ) -> torch.Tensor:
        """构建高阶邻接矩阵"""
        # 计算实体总数
        total_entities = max(max(kg1_ids), max(kg2_ids)) + 1
        
        # 计算每个KG内部的相似度
        print(f"计算KG1内部相似度 ({len(kg1_ids)} 个实体)...")
        similarity_kg1 = HighOrderNeighborComputer.compute_cosine_similarity(kg1_emb, kg1_emb)
        
        print(f"计算KG2内部相似度 ({len(kg2_ids)} 个实体)...")
        similarity_kg2 = HighOrderNeighborComputer.compute_cosine_similarity(kg2_emb, kg2_emb)
        
        # 获取Top-K邻居
        print(f"选取Top-{topk}邻居...")
        _, top_indices_kg1 = torch.topk(similarity_kg1, k=topk, dim=1)
        _, top_indices_kg2 = torch.topk(similarity_kg2, k=topk, dim=1)
        
        # 初始化邻接矩阵（稀疏表示更高效）
        edges = []
        
        # 添加KG1的高阶边
        for i, entity_id in enumerate(kg1_ids):
            for j in range(topk):
                neighbor_idx = top_indices_kg1[i][j].item()
                neighbor_id = kg1_ids[neighbor_idx]
                edges.append([entity_id, neighbor_id])
        
        # 添加KG2的高阶边
        for i, entity_id in enumerate(kg2_ids):
            for j in range(topk):
                neighbor_idx = top_indices_kg2[i][j].item()
                neighbor_id = kg2_ids[neighbor_idx]
                edges.append([entity_id, neighbor_id])
        
        # 转换为张量
        if edges:
            high_adj = torch.tensor(edges, dtype=torch.long).t()
        else:
            high_adj = torch.empty((2, 0), dtype=torch.long)
        
        print(f"高阶邻接矩阵构建完成: {high_adj.shape[1]} 条边")
        return high_adj


class HighOrderNeighborsLoader:

    def __init__(
        self,dataset_folder: str,emb_data_folder: str,dataset_name: str,language: str,cache_dir: Optional[str] = None):
        self.config = HighOrderConfig(dataset_name, language)
        self.path_manager = HighOrderPathManager(dataset_folder, emb_data_folder, self.config)
        self.computer = HighOrderNeighborComputer()
        
        cache_path = self.path_manager.get_cache_dir(cache_dir)
        self.cache_manager = HighOrderCacheManager(cache_path)
        
        print(f"初始化高阶邻居加载器:")
        print(f"  数据集: {dataset_name}")
        print(f"  语言对: {language}")
        print(f"  缓存目录: {cache_path}")
    
    def compute_high_order_neighbors(self, topk: int, use_cache: bool = True) -> torch.Tensor:
        """
        计算高阶邻居
        
        Args:
            topk: Top-K参数
            use_cache: 是否使用缓存
            
        Returns:
            high_adj: 高阶邻接矩阵，形状 (2, num_edges)
        """
        # 获取所有相关文件路径
        id_paths = self.path_manager.get_entity_id_paths()
        emb_paths = self.path_manager.get_entity_emb_paths()
        source_files = list(id_paths.values()) + list(emb_paths.values())
        
        # 检查缓存
        if use_cache and self.cache_manager.is_cache_valid(topk, source_files):
            print("从缓存加载高阶邻居...")
            result = self.cache_manager.load_cache(topk)
            if result is not None:
                print(f"成功从缓存加载，形状: {result.shape}")
                return result
        
        print("缓存无效或不存在，开始计算高阶邻居...")
        start_time = time.time()
        
        # 加载实体ID
        print("加载实体ID映射...")
        kg1_ids = self.computer.load_entity_ids(id_paths['kg1'])
        kg2_ids = self.computer.load_entity_ids(id_paths['kg2'])
        print(f"  KG1: {len(kg1_ids)} 个实体")
        print(f"  KG2: {len(kg2_ids)} 个实体")
        
        # 加载嵌入
        print("加载实体嵌入...")
        kg1_emb = self.computer.load_embeddings(emb_paths['kg1'])
        kg2_emb = self.computer.load_embeddings(emb_paths['kg2'])
        print(f"  KG1 嵌入: {kg1_emb.shape}")
        print(f"  KG2 嵌入: {kg2_emb.shape}")
        
        # 确保嵌入数量与ID数量匹配
        if len(kg1_emb) < len(kg1_ids):
            raise ValueError(f"KG1嵌入数量({len(kg1_emb)})小于ID数量({len(kg1_ids)})")
        if len(kg2_emb) < len(kg2_ids):
            raise ValueError(f"KG2嵌入数量({len(kg2_emb)})小于ID数量({len(kg2_ids)})")
        
        # 只取需要的嵌入
        kg1_emb = kg1_emb[:len(kg1_ids)]
        kg2_emb = kg2_emb[:len(kg2_ids)]
        
        # 构建高阶邻接矩阵
        high_adj = self.computer.build_high_order_adjacency(
            kg1_ids, kg2_ids, kg1_emb, kg2_emb, topk
        )
        
        # 保存缓存
        if use_cache:
            metadata_extra = {
                'dataset': self.config.dataset_name,
                'language': self.config.language,
                'num_kg1_entities': len(kg1_ids),
                'num_kg2_entities': len(kg2_ids),
                'embedding_dim': kg1_emb.shape[1]
            }
            self.cache_manager.save_cache(high_adj, topk, source_files, metadata_extra)
        
        elapsed = time.time() - start_time
        print(f"高阶邻居计算完成，耗时: {elapsed:.3f}s")
        
        return high_adj
    
    def precompute_high_order(self, topk: int, force_rebuild: bool = False):
        """预计算高阶邻居并保存到缓存"""
        print("=" * 60)
        print(f"预计算高阶邻居 (Top-K={topk})")
        print("=" * 60)
        
        use_cache = not force_rebuild
        result = self.compute_high_order_neighbors(topk, use_cache=use_cache)
        
        print("=" * 60)
        print("预计算完成")
        print("=" * 60)
        
        return result

