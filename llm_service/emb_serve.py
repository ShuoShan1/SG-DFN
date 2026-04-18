import openai
import time
from tqdm import tqdm
import asyncio

class TextEmbedService(object):
    """
    文本嵌入服务
    """

    def __init__(self):
        # 嵌入模型客户端
        self.embed_model_client = openai.AsyncClient(api_key='XX', base_url="XX")

    async def text_embedding(self, sentence_list):
        """
        本文嵌入
        :param sentence_list: 句子列表
        :return:
        """
        # 解析结果
        embed_list = []
        # 进行嵌入
        embed_res = await self.embed_model_client.embeddings.create(model="Qwen3-Embedding-8B", input=sentence_list,dimensions= 4096)# Qwen3-Embedding-8B bge-m3
        for embed_item in embed_res.data:
            embed_list.append(embed_item.embedding)
        return embed_list

if __name__ == "__main__":


    text_emb_chat = TextEmbedService()
    text_list = ["1","2"]
    text_emb_result_list = []
    start_time = time.time()
    batch_result = asyncio.run(text_emb_chat.text_embedding(text_list))
    text_emb_result_list.extend(batch_result)
    end_time = time.time()
    print(f'图像开销：{(end_time - start_time)}')
    print("数据维度：",len(text_emb_result_list[0]))


