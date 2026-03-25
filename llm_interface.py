# -*- coding:utf-8 -*-
"""
@file name  : llm_interface.py
@date       : 2026-03-23
@brief      : LLM接口封装，支持OpenAI兼容格式的API调用
              提供查询扩展、结果解释、RAG问答等功能
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI


class LLMInterface:
    """
    LLM接口封装类，支持OpenAI兼容格式的API调用
    用于查询扩展、结果解释、RAG问答等场景
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        初始化LLM接口

        Args:
            base_url: API基础URL，默认从环境变量LLM_BASE_URL获取
            api_key: API密钥，默认从环境变量LLM_API_KEY获取
            model: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 如果没有配置API密钥，标记为不可用
        self._available = bool(self.api_key)

        if self._available:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
            self.client = None

    @property
    def available(self) -> bool:
        """检查LLM是否可用"""
        return self._available

    def expand_query(self, user_query: str, num_expansions: int = 3) -> List[str]:
        """
        查询扩展：基于用户原始查询生成多个搜索意图

        Args:
            user_query: 用户原始查询
            num_expansions: 扩展查询数量

        Returns:
            扩展后的查询列表（包含原始查询）
        """
        if not self._available:
            return [user_query]

        prompt = f"""作为图像搜索助手，请基于用户的查询生成{num_expansions}个不同表达的搜索意图，以提高检索召回率。

用户查询: {user_query}

要求:
1. 保持原始语义，但使用不同的词汇和表达方式
2. 可以从不同角度描述（如风格、场景、对象、颜色等）
3. 每个扩展查询简洁明了，不超过20个字
4. 直接返回扩展查询列表，每行一个，不要编号

输出:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的图像搜索查询扩展助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            expanded = response.choices[0].message.content.strip().split('\n')
            # 过滤空行并去重
            expanded = [q.strip() for q in expanded if q.strip()]
            expanded = list(dict.fromkeys(expanded))  # 去重保持顺序

            # 确保包含原始查询
            if user_query not in expanded:
                expanded.insert(0, user_query)

            return expanded[:num_expansions + 1]

        except Exception as e:
            print(f"[LLMInterface] 查询扩展失败: {e}")
            return [user_query]

    def explain_results(self, query: str, results: List[Dict]) -> str:
        """
        结果解释：为用户解释检索结果的相关性

        Args:
            query: 用户查询
            results: 检索结果列表，每项包含path, distance等信息

        Returns:
            LLM生成的解释文本
        """
        if not self._available:
            return "（LLM未配置，无法生成解释）"

        # 构建结果描述
        result_descriptions = []
        for i, r in enumerate(results[:5], 1):
            desc = f"结果{i}: 图片路径={os.path.basename(r.get('path', 'unknown'))}, 相似度距离={r.get('distance', 'N/A'):.3f}"
            result_descriptions.append(desc)

        results_text = '\n'.join(result_descriptions)

        prompt = f"""作为图像检索系统的AI助手，请为用户解释以下检索结果的相关性。

用户查询: {query}

检索结果（按相似度排序）:
{results_text}

请用简洁友好的语言（100字以内）:
1. 简要说明检索结果与用户查询的匹配程度
2. 指出最相关的结果及其特点
3. 如果有异常情况（如结果不太相关），给出可能的原因

输出:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的图像检索结果分析助手，善于用简洁的语言解释搜索结果。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[LLMInterface] 结果解释失败: {e}")
            return "（解释生成失败）"

    def answer_with_rag(self, query: str, context: List[Dict]) -> str:
        """
        RAG问答：基于检索到的图片上下文回答用户问题

        Args:
            query: 用户问题
            context: 检索到的图片上下文

        Returns:
            LLM基于上下文生成的回答
        """
        if not self._available:
            return "（LLM未配置，无法生成回答。以下是检索到的相关图片信息。）"

        # 构建上下文描述
        context_descriptions = []
        for i, ctx in enumerate(context[:5], 1):
            desc = f"图片{i}: {os.path.basename(ctx.get('path', 'unknown'))} - 相似度: {ctx.get('distance', 0):.3f}"
            context_descriptions.append(desc)

        context_text = '\n'.join(context_descriptions)

        prompt = f"""基于以下检索到的图片信息，回答用户的问题。

检索到的相关图片:
{context_text}

用户问题: {query}

请根据上述图片信息回答问题。如果图片信息不足以回答问题，请说明。
回答要简洁、准确、有帮助。

输出:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个基于多模态检索的智能问答助手，善于根据检索结果回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[LLMInterface] RAG问答失败: {e}")
            return "（回答生成失败）"

    def chat_with_context(
        self,
        query: str,
        context: List[Dict],
        history: Optional[List[Dict]] = None
    ) -> str:
        """
        多轮对话：支持基于检索上下文的多轮问答

        Args:
            query: 当前用户输入
            context: 检索上下文（图片信息）
            history: 对话历史，格式为[{"role": "user/assistant", "content": "..."}]

        Returns:
            LLM回复
        """
        if not self._available:
            return "（LLM未配置）"

        # 构建系统提示
        context_descriptions = []
        for i, ctx in enumerate(context[:5], 1):
            desc = f"图片{i}: {os.path.basename(ctx.get('path', 'unknown'))}"
            context_descriptions.append(desc)

        context_text = '\n'.join(context_descriptions)

        system_prompt = f"""你是一个多模态图像检索助手。用户正在查看以下相关图片:

{context_text}

请基于这些图片信息与用户进行对话，回答用户关于图片的问题。
如果用户的问题与图片无关，可以正常回答但提示用户当前查看的图片信息。"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if history:
            messages.extend(history[-5:])  # 只保留最近5轮

        messages.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[LLMInterface] 对话失败: {e}")
            return "（对话生成失败）"


def get_llm_interface() -> LLMInterface:
    """
    工厂函数：获取LLM接口实例

    Returns:
        LLMInterface实例
    """
    return LLMInterface()
