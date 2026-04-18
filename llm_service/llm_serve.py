import openai
from typing import List
import asyncio

# 暂时可用的大模型接口
VLLM_MODEL_CHOICES = {
                      "xx": "xx"
                    }

class LlmModelClient(object):
    # 同步客户端
    @staticmethod
    def get_model_client(api_base: str):
        return openai.Client(api_key="xx",base_url=api_base)
    # 异步客户端
    @staticmethod
    def get_async_model_client(api_base: str):
        return openai.AsyncClient(api_key="xx",base_url=api_base)


class Llm_Service(object):
    def __init__(self):
        # 初始化服务，为每个不同的图生文模型端点创建独立的客户端
        self.sync_clients = {}
        self.async_clients = {}
        for model_name, endpoint in VLLM_MODEL_CHOICES.items():
            self.sync_clients[model_name] = LlmModelClient.get_model_client(endpoint)
            self.async_clients[model_name] = LlmModelClient.get_async_model_client(endpoint)


    def get_model_client(self, model_name: str):
        # 获取指定模型的同步客户端
        return self.sync_clients.get(model_name)
    def get_async_model_client(self, model_name: str):
        # 获取指定模型的异步客户端
        return self.async_clients.get(model_name)

    def build_message_for_llm(self, text: str) -> List:
        return [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
            }
        ]

    def sync_chat(self, messages: List, model_name: str, max_tokens=5000):
        # 同步回答
        client = self.get_model_client(model_name)
        if not client:
            return f"Error: Model '{model_name}' not found"

        reply = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        return reply.choices[0].message.content

    async def async_chat(self, messages, model_name: str, max_tokens=5000):
        # 异步回答
        client = self.get_async_model_client(model_name)
        if not client:
            return f"Error: Model '{model_name}' not found"

        async_reply = await client.chat.completions.create(
            model=model_name,
            stream=False,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens
        )
        return async_reply.choices[0].message.content

