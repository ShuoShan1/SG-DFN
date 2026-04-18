import asyncio
import time
import json
from datetime import datetime
import os
import argparse

from llm_service.llm_serve import Llm_Service
from src.llm_data_utils.data_pro_prompt import EaPrompt

class LLMKGProcessor:

    def __init__(self):
        # 初始化服务
        self.llm_service = Llm_Service()
        self.prompt_builder = EaPrompt()

    def process_kg_relation_description(self, json_file_path: str, dataset_name: str, concurrency: int, max_entities: int = None):
        """
        对每个实体进行关系描述处理，调用LLM生成标准化的关系描述并保存到原JSON文件
        """
        # 读取JSON文件
        print(f"读取文件: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)

        # 限制处理的实体数量
        if max_entities:
            entities = entities[:max_entities]
            print(f"限制处理前 {max_entities} 个实体")

        print(f"成功读取 {len(entities)} 个实体")

        # 用于跟踪进度
        self.processed_count = 0
        self.total_count = len(entities)

        async def process_single_entity(entity, semaphore, index):
            """处理单个实体的关系描述"""
            async with semaphore:
                try:
                    # 构建关系三元组信息
                    formatted_relations = []
                    for relation in entity.get('relations', []):
                        relation_name = relation.get('relation_name', '')
                        for neighbor in relation.get('neighbors', []):
                            triple = f"({entity['entity_name']},{relation_name},{neighbor})"
                            formatted_relations.append(triple)

                    # 构建实体信息字符串（不包括属性信息）
                    entity_info = "Entity name: " + entity['entity_name'] + "\nSource KG: " + dataset_name + "\nRelation triple information:\n" + '\n'.join(formatted_relations)

                    # 构建关系描述提示词
                    prompt =  self.prompt_builder.relation_description_prompt(entity_info)
                    # 调用LLM服务
                    message = self.llm_service.build_message_for_llm(prompt)
                    result = await self.llm_service.async_chat(message, "Qwen2.5_72b_GPTQ")

                    # 清理结果（移除可能的引号或特殊字符）
                    description = result.strip()
                    if description.startswith('"') and description.endswith('"'):
                        description = description[1:-1]

                    # 添加到实体数据中
                    entity['relation_description'] = description

                    # 更新进度并显示
                    self.processed_count += 1
                    progress_info = f"({self.processed_count}/{self.total_count})"
                    print(f"关系描述处理完成: {entity['entity_name']} -> {description} {progress_info}")

                    return entity

                except Exception as e:
                    print(f"处理实体 {entity.get('entity_name', 'unknown')} 的关系描述时出错: {e}")
                    entity['relation_description'] = ""  # 出错时设置为空字符串

                    # 更新进度并显示（即使出错也计数）
                    self.processed_count += 1
                    progress_info = f"({self.processed_count}/{self.total_count})"
                    print(f"关系描述处理完成（出错）: {entity.get('entity_name', 'unknown')} -> '' {progress_info}")

                    return entity

        async def process_all_entities():
            """处理所有实体"""
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [process_single_entity(entity, semaphore, i) for i, entity in enumerate(entities)]
            results = await asyncio.gather(*tasks)
            return results

        # 执行异步处理
        print(f"开始处理 {len(entities)} 个实体的关系描述，并发数: {concurrency}")
        start_time = time.time()

        processed_entities = asyncio.run(process_all_entities())

        end_time = time.time()
        print(f"所有实体关系描述处理完成，总用时: {(end_time - start_time):.3f}s")

        # 直接保存到原文件
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_entities, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到原文件: {json_file_path}")
        except Exception as e:
            print(f"保存文件失败: {e}")

        return processed_entities


    def process_kg_attribute_description(self, json_file_path: str, dataset_name: str, concurrency: int, max_entities: int = None):
            """
            对每个实体进行属性描述处理，调用LLM生成标准化的属性描述并保存到原JSON文件
            """
            # 读取JSON文件
            print(f"读取文件: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                entities = json.load(f)

            # 限制处理的实体数量
            if max_entities:
                entities = entities[:max_entities]
                print(f"限制处理前 {max_entities} 个实体")

            print(f"成功读取 {len(entities)} 个实体")

            # 用于跟踪进度
            self.processed_count = 0
            self.total_count = len(entities)

            async def process_single_entity(entity, semaphore, index):
                """处理单个实体的属性描述"""
                async with semaphore:
                    try:
                        # 构建属性三元组信息
                        formatted_attributes = []
                        for attributes in entity.get('attributes', []):
                            attribute_name = attributes.get('attribute_name', '')
                            for values in attributes.get('values', []):
                                triple = f"({entity['entity_name']},{attribute_name},{values})"
                                formatted_attributes.append(triple)

                        # 构建实体信息字符串（不包括属性信息）
                        entity_info = "Entity name: " + entity['entity_name'] + "\nSource KG: " + dataset_name + "\nAttribute triple information:\n" + '\n'.join(formatted_attributes)

                        # 构建关系描述提示词
                        prompt =  self.prompt_builder.attribute_description_prompt(entity_info)
                        # 调用LLM服务
                        message = self.llm_service.build_message_for_llm(prompt)
                        result = await self.llm_service.async_chat(message, "Qwen2.5_72b_GPTQ")

                        # 清理结果（移除可能的引号或特殊字符）
                        description = result.strip()
                        if description.startswith('"') and description.endswith('"'):
                            description = description[1:-1]

                        # 添加到实体数据中
                        entity['attribute_description'] = description

                        # 更新进度并显示
                        self.processed_count += 1
                        progress_info = f"({self.processed_count}/{self.total_count})"
                        print(f"属性描述处理完成: {entity['entity_name']} -> {description} {progress_info}")

                        return entity

                    except Exception as e:
                        print(f"处理实体 {entity.get('entity_name', 'unknown')} 的属性描述时出错: {e}")
                        entity['attribute_description'] = ""  # 出错时设置为空字符串

                        # 更新进度并显示（即使出错也计数）
                        self.processed_count += 1
                        progress_info = f"({self.processed_count}/{self.total_count})"
                        print(f"属性描述处理完成（出错）: {entity.get('entity_name', 'unknown')} -> '' {progress_info}")

                        return entity

            async def process_all_entities():
                """处理所有实体"""
                semaphore = asyncio.Semaphore(concurrency)
                tasks = [process_single_entity(entity, semaphore, i) for i, entity in enumerate(entities)]
                results = await asyncio.gather(*tasks)
                return results

            # 执行异步处理
            print(f"开始处理 {len(entities)} 个实体的属性描述，并发数: {concurrency}")
            start_time = time.time()

            processed_entities = asyncio.run(process_all_entities())

            end_time = time.time()
            print(f"所有实体属性描述处理完成，总用时: {(end_time - start_time):.3f}s")

            # 直接保存到原文件
            try:
                with open(json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_entities, f, ensure_ascii=False, indent=2)
                print(f"结果已保存到原文件: {json_file_path}")
            except Exception as e:
                print(f"保存文件失败: {e}")

            return processed_entities

    def process_all_features(self, json_file_path: str, dataset_name: str, concurrency: int, max_entities: int = None,
                             process_aliases: bool = True, process_relations: bool = True, process_attributes: bool = True):
        """
        统一处理所有特征（关系描述、属性描述），并将结果保存到原文件
        """
        print(f"开始统一处理文件: {json_file_path}")

        # 读取JSON文件
        print(f"读取文件: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            entities = json.load(f)

        # 限制处理的实体数量
        if max_entities:
            entities = entities[:max_entities]
            print(f"限制处理前 {max_entities} 个实体")

        print(f"成功读取 {len(entities)} 个实体")

        # 根据参数选择性处理 
         

        if process_relations:
            print("\n开始处理关系描述...")
            entities = self._process_relation_descriptions_batch(entities, dataset_name, concurrency)

        if process_attributes:
            print("\n开始处理属性描述...")
            entities = self._process_attribute_descriptions_batch(entities, dataset_name, concurrency)

        # 保存最终结果到原文件
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(entities, f, ensure_ascii=False, indent=2)
            print(f"所有处理结果已保存到原文件: {json_file_path}")
        except Exception as e:
            print(f"保存文件失败: {e}")

        return entities



    def _process_relation_descriptions_batch(self, entities, dataset_name, concurrency):
        """批量处理关系描述"""
        self.processed_count = 0
        self.total_count = len(entities)

        async def process_single_entity(entity, semaphore, index):
            async with semaphore:
                try:
                    # 构建关系三元组信息
                    formatted_relations = []
                    for relation in entity.get('relations', []):
                        relation_name = relation.get('relation_name', '')
                        for neighbor in relation.get('neighbors', []):
                            triple = f"({entity['entity_name']},{relation_name},{neighbor})"
                            formatted_relations.append(triple)

                    # 构建实体信息字符串（不包括属性信息）
                    entity_info = "Entity name: " + entity['entity_name'] + "\nSource KG: " + dataset_name + "\nRelation triple information:\n" + '\n'.join(formatted_relations)

                    # 构建关系描述提示词
                    prompt =  self.prompt_builder.relation_description_prompt(entity_info)
                    # 调用LLM服务
                    message = self.llm_service.build_message_for_llm(prompt)
                    result = await self.llm_service.async_chat(message, "Qwen2.5_72b_GPTQ")

                    # 清理结果（移除可能的引号或特殊字符）
                    description = result.strip()
                    if description.startswith('"') and description.endswith('"'):
                        description = description[1:-1]

                    # 添加到实体数据中
                    entity['relation_description'] = description

                    # 更新进度并显示
                    self.processed_count += 1
                    progress_info = f"({self.processed_count}/{self.total_count})"
                    print(f"关系描述处理完成: {entity['entity_name']} -> {description} {progress_info}")

                    return entity

                except Exception as e:
                    print(f"处理实体 {entity.get('entity_name', 'unknown')} 的关系描述时出错: {e}")
                    entity['relation_description'] = ""  # 出错时设置为空字符串

                    # 更新进度并显示（即使出错也计数）
                    self.processed_count += 1
                    progress_info = f"({self.processed_count}/{self.total_count})"
                    print(f"关系描述处理完成（出错）: {entity.get('entity_name', 'unknown')} -> '' {progress_info}")

                    return entity

        async def process_all_entities():
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [process_single_entity(entity, semaphore, i) for i, entity in enumerate(entities)]
            results = await asyncio.gather(*tasks)
            return results

        print(f"开始批量处理 {len(entities)} 个实体的关系描述，并发数: {concurrency}")
        start_time = time.time()

        processed_entities = asyncio.run(process_all_entities())

        end_time = time.time()
        print(f"所有实体关系描述处理完成，总用时: {(end_time - start_time):.3f}s")

        return processed_entities

    def _process_attribute_descriptions_batch(self, entities, dataset_name, concurrency):
        """批量处理属性描述"""
        self.processed_count = 0
        self.total_count = len(entities)

        async def process_single_entity(entity, semaphore, index):
            async with semaphore:
                try:
                    # 构建属性三元组信息
                    formatted_attributes = []
                    for attributes in entity.get('attributes', []):
                        attribute_name = attributes.get('attribute_name', '')
                        for values in attributes.get('values', []):
                            triple = f"({entity['entity_name']},{attribute_name},{values})"
                            formatted_attributes.append(triple)

                    # 构建实体信息字符串（不包括属性信息）
                    entity_info = "Entity name: " + entity['entity_name'] + "\nSource KG: " + dataset_name + "\nAttribute triple information:\n" + '\n'.join(formatted_attributes)

                    # 构建关系描述提示词
                    prompt =  self.prompt_builder.attribute_description_prompt(entity_info)
                    # 调用LLM服务
                    message = self.llm_service.build_message_for_llm(prompt)
                    result = await self.llm_service.async_chat(message, "Qwen2.5_72b_GPTQ")

                    # 清理结果（移除可能的引号或特殊字符）
                    description = result.strip()
                    if description.startswith('"') and description.endswith('"'):
                        description = description[1:-1]

                    # 添加到实体数据中
                    entity['attribute_description'] = description

                    # 更新进度并显示
                    self.processed_count += 1
                    progress_info = f"({self.processed_count}/{self.total_count})"
                    print(f"属性描述处理完成: {entity['entity_name']} -> {description} {progress_info}")

                    return entity

                except Exception as e:
                    print(f"处理实体 {entity.get('entity_name', 'unknown')} 的属性描述时出错: {e}")
                    entity['attribute_description'] = ""  # 出错时设置为空字符串

                    # 更新进度并显示（即使出错也计数）
                    self.processed_count += 1
                    progress_info = f"({self.processed_count}/{self.total_count})"
                    print(f"属性描述处理完成（出错）: {entity.get('entity_name', 'unknown')} -> '' {progress_info}")

                    return entity

        async def process_all_entities():
            semaphore = asyncio.Semaphore(concurrency)
            tasks = [process_single_entity(entity, semaphore, i) for i, entity in enumerate(entities)]
            results = await asyncio.gather(*tasks)
            return results

        print(f"开始批量处理 {len(entities)} 个实体的属性描述，并发数: {concurrency}")
        start_time = time.time()

        processed_entities = asyncio.run(process_all_entities())

        end_time = time.time()
        print(f"所有实体属性描述处理完成，总用时: {(end_time - start_time):.3f}s")

        return processed_entities

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLM知识图谱处理器')
    parser.add_argument('--json_file', type=str, required=True, help='输入的JSON文件路径')
    parser.add_argument('--dataset_name', type=str, required=True, help='数据集名称,用来放入提示词')
    parser.add_argument('--concurrency', type=int, default=5, help='并发数')
    parser.add_argument('--max_entities', type=int, default=0, help='最大处理实体数（默认为10，设为0则处理全部）')
    parser.add_argument('--process_aliases', action='store_true', help='处理别名')
    parser.add_argument('--process_relations', action='store_true', help='处理关系描述')
    parser.add_argument('--process_attributes', action='store_true', help='处理属性描述')
    parser.add_argument('--process_all', action='store_true', help='处理所有特性（别名、关系描述、属性描述）')

    return parser.parse_args()

if __name__ == '__main__':
    json_file = "XX" 
    dataset = "DWY100K_yg_dbp" 
    concurrency_limit = 10
    max_entities = 0  # 最大处理实体数（默认为10，设为0则处理全部）

    # 创建处理器实例
    processor = LLMKGProcessor()

    print("\n" + "="*50 + "\n开始处理关系描述...")
    result_entities = processor.process_kg_relation_description(
        json_file_path=json_file,
        dataset_name=dataset,
        concurrency=concurrency_limit,
        max_entities=max_entities
    )
    print(f"成功处理 {len(result_entities)} 个实体的关系描述")

    print("\n" + "="*50 + "\n开始处理属性描述...")
    result_entities = processor.process_kg_attribute_description(
        json_file_path=json_file,
        dataset_name=dataset,
        concurrency=concurrency_limit,
        max_entities=max_entities
    )
    print(f"成功处理 {len(result_entities)} 个实体的属性描述")