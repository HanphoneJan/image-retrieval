# -*- coding:utf-8 -*-
"""
@file name  : agent_tools.py
@date       : 2026-03-23
@brief      : Agent工具定义模块
              为AI Agent提供图像检索能力的工具封装
              支持OpenAI Function Calling格式和通用工具格式
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json

try:
    from retrieval_by_faiss import ImageRetrievalModule
    from llm_interface import LLMInterface
except ImportError:
    from .retrieval_by_faiss import ImageRetrievalModule
    from .llm_interface import LLMInterface


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


@dataclass
class AgentTool:
    """Agent工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter]
    func: Callable

    def to_openai_function(self) -> Dict[str, Any]:
        """转换为OpenAI Function Calling格式"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为通用字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "enum": p.enum
                }
                for p in self.parameters
            ]
        }


class ImageRetrievalTools:
    """
    图像检索工具集合

    为Agent提供以下能力：
    1. search_by_text - 文本搜索图片
    2. search_by_image - 以图搜图
    3. explain_results - 解释搜索结果
    4. get_similar_images - 获取相似图片
    """

    def __init__(self, rag_engine):
        """
        初始化工具集合

        Args:
            rag_engine: RAGEngine实例
        """
        self.rag_engine = rag_engine
        self._tools = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具"""
        # 工具1: 文本搜索
        self.register_tool(
            AgentTool(
                name="search_by_text",
                description="根据文本描述搜索相关图片。支持自然语言描述，如'a dog playing in the park'",
                parameters=[
                    ToolParameter(
                        name="query",
                        type="string",
                        description="搜索文本描述",
                        required=True
                    ),
                    ToolParameter(
                        name="topk",
                        type="integer",
                        description="返回结果数量，默认为5",
                        required=False
                    )
                ],
                func=self._search_by_text
            )
        )

        # 工具2: 以图搜图
        self.register_tool(
            AgentTool(
                name="search_by_image",
                description="根据图片搜索相似图片。传入图片路径，返回相似的图片列表",
                parameters=[
                    ToolParameter(
                        name="image_path",
                        type="string",
                        description="查询图片的路径",
                        required=True
                    ),
                    ToolParameter(
                        name="topk",
                        type="integer",
                        description="返回结果数量，默认为5",
                        required=False
                    )
                ],
                func=self._search_by_image
            )
        )

        # 工具3: 解释搜索结果
        self.register_tool(
            AgentTool(
                name="explain_search_results",
                description="为用户解释搜索结果的相关性和特点",
                parameters=[
                    ToolParameter(
                        name="original_query",
                        type="string",
                        description="原始搜索查询",
                        required=True
                    ),
                    ToolParameter(
                        name="results",
                        type="array",
                        description="搜索结果列表，每项包含path和distance",
                        required=True
                    )
                ],
                func=self._explain_results
            )
        )

        # 工具4: 智能问答
        self.register_tool(
            AgentTool(
                name="answer_with_image_context",
                description="基于检索到的图片上下文回答用户问题。适合需要结合图片内容的问答场景",
                parameters=[
                    ToolParameter(
                        name="question",
                        type="string",
                        description="用户问题",
                        required=True
                    ),
                    ToolParameter(
                        name="context_size",
                        type="integer",
                        description="检索上下文数量，默认为3",
                        required=False
                    )
                ],
                func=self._answer_with_context
            )
        )

    def register_tool(self, tool: AgentTool):
        """注册工具"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[AgentTool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """获取OpenAI Function Calling格式的工具列表"""
        return [tool.to_openai_function() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """获取工具描述文本（用于Prompt）"""
        descriptions = []
        for name, tool in self._tools.items():
            desc = f"- {name}: {tool.description}"
            params = ", ".join([f"{p.name}({p.type})" for p in tool.parameters if p.required])
            desc += f"\n  参数: {params}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found"}

        try:
            result = tool.func(**arguments)
            return {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "arguments": arguments,
                "error": str(e),
                "success": False
            }

    # ============ 工具函数实现 ============

    def _search_by_text(self, query: str, topk: int = 5) -> Dict[str, Any]:
        """文本搜索实现"""
        result = self.rag_engine.search_and_explain(
            query=query,
            topk=topk,
            use_expansion=True
        )
        return {
            "query": query,
            "expanded_queries": result.get("expanded_queries", []),
            "results": [
                {
                    "path": r["path"],
                    "similarity_score": 1.0 / (1.0 + r["distance"]),  # 转换为相似度分数
                    "rank": i + 1
                }
                for i, r in enumerate(result["results"])
            ],
            "ai_explanation": result.get("ai_explanation", "")
        }

    def _search_by_image(self, image_path: str, topk: int = 5) -> Dict[str, Any]:
        """以图搜图实现"""
        result = self.rag_engine.analyze_image_similarity(
            image_path=image_path,
            topk=topk
        )
        return {
            "query_image": result["query_image"],
            "results": [
                {
                    "path": r["path"],
                    "similarity_score": 1.0 / (1.0 + r["distance"]),
                    "rank": i + 1
                }
                for i, r in enumerate(result["similar_images"])
            ],
            "ai_analysis": result.get("ai_analysis", "")
        }

    def _explain_results(self, original_query: str, results: List[Dict]) -> Dict[str, Any]:
        """解释搜索结果实现"""
        explanation = self.rag_engine.llm.explain_results(original_query, results)
        return {
            "original_query": original_query,
            "explanation": explanation,
            "key_points": [
                "结果按相似度排序",
                "距离越小表示越相似",
                f"共返回{len(results)}个结果"
            ]
        }

    def _answer_with_context(self, question: str, context_size: int = 3) -> Dict[str, Any]:
        """基于上下文问答实现"""
        result = self.rag_engine.rag_qa(
            question=question,
            topk=context_size
        )
        return {
            "question": question,
            "answer": result["answer"],
            "context_images": [
                {"path": r["path"], "relevance": 1.0 / (1.0 + r["distance"])}
                for r in result["context"]
            ],
            "retrieval_explanation": result.get("ai_explanation", "")
        }


class ToolUsingAgent:
    """
    工具使用Agent演示

    展示如何让LLM使用工具进行多步推理
    """

    def __init__(self, tools: ImageRetrievalTools, llm_interface):
        self.tools = tools
        self.llm = llm_interface

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        运行Agent处理用户输入

        Args:
            user_input: 用户输入

        Returns:
            处理结果
        """
        if not self.llm.available:
            return {"error": "LLM not available"}

        # 构建系统提示
        system_prompt = f"""你是一个智能图像检索助手。你可以使用以下工具帮助用户:

{self.tools.get_tools_description()}

请根据用户需求选择合适的工具。如果需要多个步骤，请逐步执行。
直接以JSON格式返回工具调用:
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}"""

        # 第一次调用：确定使用什么工具
        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3,
                max_tokens=500
            )

            content = response.choices[0].message.content

            # 尝试解析工具调用
            try:
                # 尝试从JSON块中提取
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    json_str = content.split("```")[1].split("```")[0].strip()
                else:
                    json_str = content.strip()

                tool_call = json.loads(json_str)
                tool_name = tool_call.get("tool")
                arguments = tool_call.get("arguments", {})

                # 执行工具
                result = self.tools.execute_tool(tool_name, arguments)

                return {
                    "user_input": user_input,
                    "tool_call": tool_call,
                    "result": result,
                    "raw_response": content
                }

            except json.JSONDecodeError:
                # 如果不是有效的工具调用，直接返回LLM回复
                return {
                    "user_input": user_input,
                    "direct_response": content,
                    "tool_call": None
                }

        except Exception as e:
            return {
                "user_input": user_input,
                "error": str(e)
            }


# 便捷函数
def create_tools_for_rag_engine(rag_engine) -> ImageRetrievalTools:
    """
    为RAG引擎创建工具集合

    Args:
        rag_engine: RAGEngine实例

    Returns:
        ImageRetrievalTools实例
    """
    return ImageRetrievalTools(rag_engine)
