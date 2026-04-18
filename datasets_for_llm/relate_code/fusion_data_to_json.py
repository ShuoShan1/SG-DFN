# import pandas as pd
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import argparse
from collections import defaultdict
import pandas as pd

# 加载数据集数据，将每个实体转化为格式化的json文件，包括实体名称，实体id，关系，属性，关系描述，属性描述  这几个键

class EntityInfoProcessor:
    """实体信息处理器，用于加载和处理知识图谱数据"""

    def __init__(self, data_dir: str, output_dir: str, kg_name: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.kg_name = kg_name

        # 解析知识图谱名称
        self.kg_pair = kg_name.split('_')
        if len(self.kg_pair) != 2:
            raise ValueError(f"知识图谱名称格式错误: {kg_name}，应为 'kg1_kg2' 格式，如 'zh_en'")

        self.kg1, self.kg2 = self.kg_pair
        self.kg1_suffix = self.get_kg_suffix(self.kg1)
        self.kg2_suffix = self.get_kg_suffix(self.kg2)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def get_kg_suffix(self, kg_code: str) -> str:
        """根据知识图谱代码获取对应的文件后缀"""
        # 根据数据集的经验模式进行映射
        # DBP15K: zh_en, ja_en, fr_en (第一个语言通常是非英语，第二个是英语)
        # SRPRS: en_de, en_fr (第一个语言是英语，第二个是非英语)
        # DWY100K: wd_dbp, yg_dbp (特定的命名方式)

        # 特殊处理DWY100K数据集
        if self.kg_name == "wd_dbp":
            if kg_code == "wd":
                return "1"
            elif kg_code == "dbp":
                return "2"
        elif self.kg_name == "yg_dbp":
            if kg_code == "yg":
                return "1"
            elif kg_code == "dbp":
                return "2"
        else:
            # 一般规则：在成对的语言代码中，第一个语言对应'1'，第二个语言对应'2'
            if kg_code == self.kg1:
                return '1'
            elif kg_code == self.kg2:
                return '2'

        # 默认返回'1'
        print(f"警告: 未找到语言代码 '{kg_code}' 的映射，使用默认映射'1'")
        return '1'
        
    def extract_name_from_uri(self, uri: str) -> str:
        """从URI中提取实体名称"""
        return uri.split("/")[-1]

    def load_entities(self, kg_suffix: str) -> Dict[int, str]:
        """加载指定知识图谱的实体ID和实体名称的映射"""
        entities_path = os.path.join(self.data_dir, f"ent_ids_{kg_suffix}")
        if not os.path.exists(entities_path):
            raise FileNotFoundError(f"实体文件不存在: {entities_path}")
        
        entities = pd.read_csv(entities_path, sep="\t", header=None, names=["entity_id", "entity_uri"])
        entities["entity_name"] = entities["entity_uri"].apply(self.extract_name_from_uri)
        return dict(zip(entities["entity_id"], entities["entity_name"]))

    def load_relations(self, kg_suffix: str) -> Dict[int, str]:
        """加载指定知识图谱的关系ID和关系名称的映射"""
        relations_path = os.path.join(self.data_dir, f"rel_ids_{kg_suffix}")
        if not os.path.exists(relations_path):
            raise FileNotFoundError(f"关系文件不存在: {relations_path}")
        
        relations = pd.read_csv(relations_path, sep="\t", header=None, names=["rel_id", "rel_uri"])
        relations["rel_name"] = relations["rel_uri"].apply(self.extract_name_from_uri)
        return dict(zip(relations["rel_id"], relations["rel_name"]))

    def load_triples(self, kg_suffix: str) -> List[Tuple[int, int, int]]:
        """加载指定知识图谱的三元组数据"""
        triples_path = os.path.join(self.data_dir, f"triples_{kg_suffix}")
        if not os.path.exists(triples_path):
            raise FileNotFoundError(f"三元组文件不存在: {triples_path}")
        
        triples = pd.read_csv(triples_path, sep="\t", header=None, names=["head", "rel", "tail"])
        return list(triples.itertuples(index=False, name=None))

    def load_attributes(self, kg_code: str) -> List[Tuple[str, str, str]]:
        """加载指定知识图谱的属性数据（不包含属性类型）"""

        # 根据数据集类型确定属性文件名格式
        if self.kg_name == "wd_dbp":
            # wd_dbp 使用特殊的属性文件命名方式
            if kg_code == "wd":
                attributes_path = os.path.join(self.data_dir, "atts_properties_wd.txt")
            elif kg_code == "dbp":
                attributes_path = os.path.join(self.data_dir, "atts_properties_dbp.txt")
            else:
                # 如果不是标准的wd/dbp，尝试常规命名
                attributes_path = os.path.join(self.data_dir, f"atts_properties_{kg_code}.txt")
        elif self.kg_name == "yg_dbp":
            # yg_dbp 使用atts_properties_yg.txt和atts_properties_dbp.txt
            if kg_code == "yg":
                attributes_path = os.path.join(self.data_dir, "atts_properties_yg.txt")
            elif kg_code == "dbp":
                attributes_path = os.path.join(self.data_dir, "atts_properties_dbp.txt")
            else:
                # 如果不是标准的yg/dbp，尝试常规命名
                attributes_path = os.path.join(self.data_dir, f"atts_properties_{kg_code}.txt")
        else:
            # 对于DBP15K和SRPRS数据集，使用原有逻辑
            attributes_path = os.path.join(self.data_dir, f"atts_properties_{kg_code}.txt")

        if not os.path.exists(attributes_path):
            print(f"警告: 属性文件不存在: {attributes_path}")
            return []

        try:
            # 读取四列：实体URI、属性名、属性值、属性类型
            attributes = pd.read_csv(attributes_path,sep="\t",header=None,names=["entity_uri", "attr_name", "attr_value", "attr_type"])
            # 只保留前三列：实体URI、属性名、属性值
            attributes = attributes[["entity_uri", "attr_name", "attr_value"]]
            attributes["entity_name"] = attributes["entity_uri"].apply(self.extract_name_from_uri)
            return list(attributes[["entity_name", "attr_name", "attr_value"]].itertuples(index=False, name=None))
        except Exception as e:
            print(f"警告: 读取属性文件失败: {e}")
            return []

    def build_entity_relations_neighbors(self, triples: List[Tuple[int, int, int]]) -> Dict[int, List[Tuple[int, int]]]:
        """构建实体-关系-邻居的映射"""
        entity_relations_neighbors = defaultdict(list)
        
        for head, rel, tail in triples:
            # 头实体的关系和尾实体
            entity_relations_neighbors[head].append((rel, tail))
            # 尾实体的关系和头实体（反向关系）
            entity_relations_neighbors[tail].append((rel, head))
            
        return dict(entity_relations_neighbors)

    def build_entity_attributes(self, attributes: List[Tuple[str, str, str]]) -> Dict[str, List[Tuple[str, str]]]:
        """构建实体-属性的映射（不包含属性类型）"""
        entity_attributes = defaultdict(list)
        
        for entity_name, attr_name, attr_value in attributes:
            entity_attributes[entity_name].append((attr_name, attr_value))
            
        return dict(entity_attributes)

    def generate_entity_data(self, entities: Dict[int, str], relations: Dict[int, str],
                           entity_relations_neighbors: Dict[int, List[Tuple[int, int]]],
                           entity_attributes: Dict[str, List[Tuple[str, str]]]) -> Tuple[List[Dict], Dict[str, int]]:
        """生成实体综合数据（包含关系、属性、别名、关系描述和属性描述）"""

        combined_data = []

        # 统计信息
        stats = {
            "total_entities": len(entities),
            "entities_with_attributes": 0,
            "entities_without_attributes": 0,
            "entities_with_relations": 0,
            "entities_without_relations": 0
        }

        for entity_id in tqdm(entities, desc="处理实体"):
            entity_name = entities[entity_id]

            # 初始化实体数据结构
            entity_data = {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "relations": [],
                "attributes": [],
                "aliases": [],  # 空的别名信息，供后续处理
                "relation_description": "",  # 空的关系描述，供后续处理
                "attribute_description": ""  # 空的属性描述，供后续处理
            }

            # 处理关系数据
            relations_dict = defaultdict(list)
            has_relations = entity_id in entity_relations_neighbors

            if has_relations:
                for rel_id, neighbor_id in entity_relations_neighbors[entity_id]:
                    rel_name = relations.get(rel_id, f"unknown_relation_{rel_id}")
                    neighbor_name = entities.get(neighbor_id, f"unknown_entity_{neighbor_id}")
                    relations_dict[rel_name].append(neighbor_name)

                # 转换为列表格式
                entity_data["relations"] = [
                    {"relation_name": rel_name, "neighbors": neighbors}
                    for rel_name, neighbors in relations_dict.items()
                ]

            # 处理属性数据
            attributes_dict = defaultdict(list)
            has_attributes = entity_name in entity_attributes

            if has_attributes:
                for attr_name, attr_value in entity_attributes[entity_name]:
                    attributes_dict[attr_name].append(attr_value)

                # 转换为列表格式
                entity_data["attributes"] = [
                    {"attribute_name": attr_name, "values": values}
                    for attr_name, values in attributes_dict.items()
                ]

            combined_data.append(entity_data)
            stats["entities_with_relations" if has_relations else "entities_without_relations"] += 1
            stats["entities_with_attributes" if has_attributes else "entities_without_attributes"] += 1

        return combined_data, stats

    def save_json_data(self, data: List[Dict], filename: str) -> None:
        """保存数据为JSON格式"""
        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"数据已保存到: {output_path}")
        except Exception as e:
            print(f"保存文件失败: {output_path}, 错误: {e}")

    def save_stats_data(self, stats: Dict[str, int], kg_code: str) -> None:
        """保存统计信息为JSON格式"""
        stats_filename = f"{kg_code}_stats.json"
        output_path = os.path.join(self.output_dir, stats_filename)
        
        try:
            # 添加计算字段
            enhanced_stats = stats.copy()
            total = stats['total_entities']
            enhanced_stats.update({
                "attribute_coverage_rate": round(stats['entities_with_attributes'] / total * 100, 2),
                "relation_coverage_rate": round(stats['entities_with_relations'] / total * 100, 2),
                "kg_code": kg_code
            })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_stats, f, ensure_ascii=False, indent=2)
            print(f"统计信息已保存到: {output_path}")
        except Exception as e:
            print(f"保存统计文件失败: {output_path}, 错误: {e}")

    def process_kg(self, kg_code: str, kg_suffix: str) -> None:
        """处理指定的知识图谱"""
        print(f"\n开始处理知识图谱: {kg_code} (后缀: {kg_suffix})")

        try:
            # 加载数据
            entities = self.load_entities(kg_suffix)
            relations = self.load_relations(kg_suffix)
            triples = self.load_triples(kg_suffix)
            attributes = self.load_attributes(kg_code)

            print(f"加载完成: {len(entities)}个实体, {len(relations)}个关系, {len(triples)}个三元组, {len(attributes)}个属性")

            # 构建映射
            entity_relations_neighbors = self.build_entity_relations_neighbors(triples)
            entity_attributes = self.build_entity_attributes(attributes)

            # 生成数据
            combined_data, stats = self.generate_entity_data(
                entities, relations, entity_relations_neighbors, entity_attributes
            )

            # 显示统计信息
            self.print_stats(kg_code, stats)

            # 保存综合数据
            self.save_json_data(combined_data, f"{kg_code}_kg_info_with_alias_with_rel_desc_with_att_desc.json")

            print(f"{kg_code}知识图谱处理完成！综合数据: {len(combined_data)}个实体")

        except Exception as e:
            print(f"处理知识图谱 {kg_code} 时出现错误: {e}")
            raise

    def print_stats(self, kg_code: str, stats: Dict[str, int]) -> None:
        """打印统计信息"""
        total = stats['total_entities']
        print(f"\n{kg_code} 知识图谱统计信息:")
        print(f"  实体总数: {total}")
        print(f"  有属性实体: {stats['entities_with_attributes']} ({stats['entities_with_attributes']/total*100:.2f}%)")
        print(f"  无属性实体: {stats['entities_without_attributes']} ({stats['entities_without_attributes']/total*100:.2f}%)")
        print(f"  有关系实体: {stats['entities_with_relations']} ({stats['entities_with_relations']/total*100:.2f}%)")
        print(f"  无关系实体: {stats['entities_without_relations']} ({stats['entities_without_relations']/total*100:.2f}%)")

    def info_fusion(self) -> None:
        """主处理函数"""
        print(f"========== 开始处理知识图谱对: {self.kg_name} ==========")

        try:
            # 处理第一个知识图谱
            self.process_kg(self.kg1, self.kg1_suffix)
            
            # 处理第二个知识图谱
            self.process_kg(self.kg2, self.kg2_suffix)
            
            print(f"\n知识图谱对 {self.kg_name} 处理完成！")
            
        except Exception as e:
            print(f"处理知识图谱对时出现错误: {e}")
            raise

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='实体信息处理器')
    parser.add_argument('--kg_name', type=str, default='wd_dbp', help='知识图谱对名称，如 zh_en, en_de, en_fr, wd_dbp, yg_dbp')
    parser.add_argument('--dataset_type', type=str, default='DWY100K', choices=['DBP15K', 'SRPRS', 'DWY100K'],
                        help='数据集类型，DBP15K, SRPRS 或 DWY100K')
    parser.add_argument('--data_dir', type=str, default='XX/datasets',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='XX/datasets_for_llm',
                        help='输出目录')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 构建完整路径
    data_dir = os.path.join(args.data_dir, args.dataset_type, args.kg_name)
    output_dir = os.path.join(args.output_dir, args.dataset_type, args.kg_name)

    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录不存在: {data_dir}")
        return

    try:
        # 创建处理器实例
        processor = EntityInfoProcessor(data_dir, output_dir, args.kg_name)

        # 执行信息融合
        processor.info_fusion()

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()