import pandas as pd
import os
from tqdm import tqdm

# 加载指定路径下的数据集以实现构成实体-关系-属性汇总的txt 作为llm的输入

# 数据集路径
DATA_DIR = "XX"

# 提取URI的最后一部分作为名称
def extract_name_from_uri(uri):
    return uri.split("/")[-1]

# 加载实体ID和实体名称
def load_entities():
    entities_path = os.path.join(DATA_DIR, "ent_ids_2")
    entities = pd.read_csv(entities_path, sep="\t", header=None, names=["entity_id", "entity_uri"])
    entities["entity_name"] = entities["entity_uri"].apply(extract_name_from_uri)
    return dict(zip(entities["entity_id"], entities["entity_name"]))

# 加载关系ID和关系名称
def load_relations():
    relations_path = os.path.join(DATA_DIR, "rel_ids_2")
    relations = pd.read_csv(relations_path, sep="\t", header=None, names=["rel_id", "rel_uri"])
    relations["rel_name"] = relations["rel_uri"].apply(extract_name_from_uri)
    return dict(zip(relations["rel_id"], relations["rel_name"]))

# 加载三元组
def load_triples():
    triples_path = os.path.join(DATA_DIR, "triples_2")
    triples = pd.read_csv(triples_path, sep="\t", header=None, names=["head", "rel", "tail"])
    return triples[["head", "rel", "tail"]].values.tolist()

# 加载属性
def load_attributes():
    attributes_path = os.path.join(DATA_DIR, "atts_properties_en.txt")
    attributes = pd.read_csv(attributes_path, sep="\t", header=None, names=["entity_uri", "attr_name", "attr_value", "attr_type"])
    attributes["entity_name"] = attributes["entity_uri"].apply(extract_name_from_uri)
    return attributes[["entity_name", "attr_name", "attr_value", "attr_type"]].values.tolist()

# 构建实体-关系-邻居的映射
def build_entity_relations_neighbors(triples):
    entity_relations_neighbors = {}
    for head, rel, tail in triples:
        if head not in entity_relations_neighbors:
            entity_relations_neighbors[head] = []
        entity_relations_neighbors[head].append((rel, tail))

        if tail not in entity_relations_neighbors:
            entity_relations_neighbors[tail] = []
        entity_relations_neighbors[tail].append((rel, head))
    return entity_relations_neighbors

# 构建实体-属性的映射
def build_entity_attributes(attributes):
    entity_attributes = {}
    for entity_name, attr_name, attr_value, attr_type in attributes:
        if entity_name not in entity_attributes:
            entity_attributes[entity_name] = []
        entity_attributes[entity_name].append((attr_name, attr_value, attr_type))
    return entity_attributes

# 输出实体的关系、邻居和属性
def print_entity_info(entity_id, entities, relations, entity_relations_neighbors, entity_attributes, output_file):
    entity_name = entities[entity_id]
    output_file.write(f"ID: {entity_id}, 名称: {entity_name}\n")

    # 输出关系和邻居
    if entity_id in entity_relations_neighbors:
        output_file.write("关系和邻居:\n")
        for rel_id, neighbor_id in entity_relations_neighbors[entity_id]:
            rel_name = relations[rel_id]
            neighbor_name = entities[neighbor_id]
            output_file.write(f"  {rel_name}:{neighbor_name}\n")
   
    # 输出属性
    if entity_name in entity_attributes:
        output_file.write("属性:\n")
        for attr_name, attr_value, attr_type in entity_attributes[entity_name]:
            output_file.write(f"  {attr_name}:{attr_value}\n")
    

    output_file.write("-" * 50 + "\n")

# 主函数
def info_fusion():
    # 加载数据
    entities = load_entities()
    relations = load_relations()
    triples = load_triples()
    attributes = load_attributes()

    # 构建实体-关系-邻居的映射
    entity_relations_neighbors = build_entity_relations_neighbors(triples)

    # 构建实体-属性的映射
    entity_attributes = build_entity_attributes(attributes)

    # 输出结果到文件
    output_path = os.path.join(DATA_DIR, "en_entity_info_output.txt")
    with open(output_path, "w", encoding="utf-8") as output_file:
        for entity_id in tqdm(entities, desc="处理实体"):
            print_entity_info(entity_id, entities, relations, entity_relations_neighbors, entity_attributes, output_file)

    print(f"实体信息已保存到: {output_path}")



def process_text_file(file_path: str) -> list:
    """
    处理文本文件：
    1. 通过 "--------------------------------------------------" 切分文本块。
    2. 将每个文本块中的换行符替换为句号。
    3. 返回处理后的文本块列表。

    :param file_path: 文本文件路径
    :return: 处理后的文本块列表
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 通过 "--------------------------------------------------" 切分文本块
    text_blocks = content.split('--------------------------------------------------')

    # 去除空白文本块（如果有）
    text_blocks = [block.strip() for block in text_blocks if block.strip()]

    # 将每个文本块中的换行符替换为句号
    processed_blocks = [block.replace('\n', '。') for block in text_blocks]

    return processed_blocks


# 示例使用
if __name__ == "__main__":
    # 输入文件路径
    file_path = "XX"  # 替换为你的文件路径

    # 处理文件
    result = process_text_file(file_path)

    # 打印结果
    for i, block in enumerate(result):
        print(f"文本块 {i + 1}:\n{block}\n")
    
