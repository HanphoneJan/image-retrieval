# -*- coding:utf-8 -*-
"""
@file name  : rag_engine.py
@author     : multimodal-rag-engine
@date       : 2026-03-23
@brief      : RAG（检索增强生成）引擎
              整合向量检索与LLM生成能力，实现智能图像检索与问答
"""
import os
from typing import List, Dict, Optional, Any
import numpy as np

try:
    from retrieval_by_faiss import ImageRetrievalModule
    from llm_interface import LLMInterface
except ImportError:
    from .retrieval_by_faiss import ImageRetrievalModule
    from .llm_interface import LLMInterface


class RAGEngine:
    """
    RAG（检索增强生成）引擎

    整合以下能力：
    1. 多模态向量检索（CLIP + Faiss）
    2. LLM查询理解与扩展
    3. 检索结果智能解释
    4. 基于检索上下文的问答
    """

    def __init__(
        self,
        retrieval_module: ImageRetrievalModule,
        llm_interface: Optional[LLMInterface] = None,
        enable_expansion: bool = True,
        enable_explanation: bool = True,
        index_manager: Optional[Any] = None
    ):
        """
        初始化RAG引擎

        Args:
            retrieval_module: 图像检索模块实例
            llm_interface: LLM接口实例，为None时自动创建
            enable_expansion: 是否启用查询扩展
            enable_explanation: 是否启用结果解释
            index_manager: 增量索引管理器（可选），用于支持增量索引搜索
        """
        self.retrieval_module = retrieval_module
        self.llm = llm_interface or LLMInterface()
        self.enable_expansion = enable_expansion and self.llm.available
        self.enable_explanation = enable_explanation and self.llm.available
        self.index_manager = index_manager  # 增量索引管理器

        print("[RAGEngine] 初始化完成")
        print(f"  - LLM可用: {self.llm.available}")
        print(f"  - 查询扩展: {self.enable_expansion}")
        print(f"  - 结果解释: {self.enable_explanation}")
        print(f"  - 增量索引: {self.index_manager is not None}")

    def _search(self, query: str, topk: int) -> tuple:
        """
        内部搜索方法，支持增量索引管理器

        Args:
            query: 查询（文本或图片路径）
            topk: 返回结果数量

        Returns:
            (distances, ids, paths): 距离、ID、路径列表
        """
        if self.index_manager is not None:
            # 使用增量索引管理器搜索
            # 需要先将查询转换为特征向量
            if os.path.exists(query):
                # 图片查询
                feat_vec = self.retrieval_module.clip_model.encode_image_by_path(query)
            else:
                # 文本查询
                feat_vec = self.retrieval_module.clip_model.encode_text_by_string(query)

            feat_vec = feat_vec.astype(np.float32)
            distances, ids, paths = self.index_manager.search(feat_vec, topk)
            return distances, ids, paths
        else:
            # 使用原有检索模块
            return self.retrieval_module.retrieval_func(query, topk)

    def search_and_explain(
        self,
        query: str,
        topk: int = 10,
        use_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        智能检索并解释结果

        Args:
            query: 用户查询（文本或图片路径）
            topk: 返回结果数量
            use_expansion: 是否使用查询扩展

        Returns:
            包含检索结果、扩展查询、AI解释的字典
        """
        result = {
            "original_query": query,
            "expanded_queries": [],
            "results": [],
            "ai_explanation": "",
            "total_results": 0
        }

        # Step 1: 查询扩展（如果启用）
        expanded_queries = []
        if use_expansion and self.enable_expansion and not os.path.exists(query):
            # 只有文本查询才进行扩展
            expanded_queries = self.llm.expand_query(query, num_expansions=3)
            result["expanded_queries"] = expanded_queries
            print(f"[RAGEngine] 查询扩展: {query} -> {expanded_queries}")
        else:
            expanded_queries = [query]

        # Step 2: 多查询检索与融合
        all_results = []
        seen_paths = set()

        for q in expanded_queries:
            try:
                distance_result, index_result, path_list = self._search(
                    q, topk=min(topk * 2, 20)  # 检索更多用于去重
                )

                for dist, idx, path in zip(distance_result, index_result, path_list):
                    if path not in seen_paths and path != 'None':
                        seen_paths.add(path)
                        all_results.append({
                            "path": path,
                            "distance": float(dist),
                            "index": int(idx),
                            "matched_query": q
                        })
            except Exception as e:
                print(f"[RAGEngine] 检索失败 '{q}': {e}")
                continue

        # Step 3: 重排序（按距离排序）
        all_results.sort(key=lambda x: x["distance"])
        final_results = all_results[:topk]

        result["results"] = final_results
        result["total_results"] = len(final_results)

        # Step 4: AI解释
        if self.enable_explanation and final_results:
            explanation = self.llm.explain_results(query, final_results)
            result["ai_explanation"] = explanation
            print(f"[RAGEngine] AI解释: {explanation[:100]}...")

        return result

    def rag_qa(
        self,
        question: str,
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        基于检索的问答（RAG QA）

        Args:
            question: 用户问题
            topk: 检索上下文数量

        Returns:
            包含回答和检索上下文的字典
        """
        # Step 1: 检索相关图片
        retrieval_result = self.search_and_explain(
            query=question,
            topk=topk,
            use_expansion=False  # QA场景通常不需要扩展
        )

        # Step 2: LLM基于上下文回答
        if self.llm.available:
            answer = self.llm.answer_with_rag(
                query=question,
                context=retrieval_result["results"]
            )
        else:
            answer = "（LLM未配置，请配置LLM_API_KEY以启用智能回答）"

        return {
            "question": question,
            "answer": answer,
            "context": retrieval_result["results"],
            "ai_explanation": retrieval_result.get("ai_explanation", "")
        }

    def chat_with_images(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        基于检索图片的多轮对话

        Args:
            query: 当前用户输入
            history: 对话历史
            topk: 检索上下文数量

        Returns:
            包含回复和上下文的字典
        """
        # Step 1: 检索相关图片作为上下文
        retrieval_result = self.search_and_explain(
            query=query,
            topk=topk,
            use_expansion=False
        )

        # Step 2: 基于上下文的对话
        if self.llm.available:
            response = self.llm.chat_with_context(
                query=query,
                context=retrieval_result["results"],
                history=history
            )
        else:
            response = "（LLM未配置，请配置LLM_API_KEY以启用对话功能）"

        return {
            "query": query,
            "response": response,
            "context": retrieval_result["results"],
            "history": history or []
        }

    def analyze_image_similarity(
        self,
        image_path: str,
        topk: int = 5
    ) -> Dict[str, Any]:
        """
        分析图片相似性并生成描述

        Args:
            image_path: 查询图片路径
            topk: 检索数量

        Returns:
            包含相似图片和AI分析的字典
        """
        # 检索相似图片
        retrieval_result = self.search_and_explain(
            query=image_path,
            topk=topk,
            use_expansion=False  # 图片查询不需要扩展
        )

        # 生成AI分析
        if self.llm.available and retrieval_result["results"]:
            # 构建相似图片描述
            similar_descriptions = []
            for i, r in enumerate(retrieval_result["results"][:5], 1):
                desc = f"相似图片{i}: {os.path.basename(r['path'])}, 相似度距离={r['distance']:.3f}"
                similar_descriptions.append(desc)

            similar_text = '\n'.join(similar_descriptions)

            prompt = f"""分析以下图片检索结果，描述查询图片与相似图片的关系。

检索到的相似图片:
{similar_text}

请用简洁的语言（80字以内）:
1. 描述这些图片的共同特征
2. 指出最相似的几张图片
3. 给出可能的场景或类别推断

分析:"""

            try:
                response = self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": "你是一个图像分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=150
                )
                analysis = response.choices[0].message.content.strip()
            except Exception as e:
                analysis = f"（分析生成失败: {e}）"
        else:
            analysis = retrieval_result.get("ai_explanation", "")

        return {
            "query_image": image_path,
            "similar_images": retrieval_result["results"],
            "ai_analysis": analysis
        }


class RAGPipeline:
    """
    RAG Pipeline 构建器

    用于构建标准化的RAG处理流程，支持自定义处理步骤
    """

    def __init__(
        self,
        retrieval_module: ImageRetrievalModule,
        llm_interface: Optional[LLMInterface] = None
    ):
        self.retrieval_module = retrieval_module
        self.llm = llm_interface or LLMInterface()
        self.steps = []

    def add_step(self, name: str, func):
        """添加处理步骤"""
        self.steps.append({"name": name, "func": func})
        return self

    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行Pipeline

        Args:
            query: 输入查询
            **kwargs: 额外参数

        Returns:
            Pipeline执行结果
        """
        context = {
            "query": query,
            "results": [],
            "metadata": {}
        }

        for step in self.steps:
            try:
                context = step["func"](context, self.retrieval_module, self.llm, **kwargs)
                print(f"[Pipeline] 步骤 '{step['name']}' 完成")
            except Exception as e:
                print(f"[Pipeline] 步骤 '{step['name']}' 失败: {e}")
                context["metadata"][f"{step['name']}_error"] = str(e)

        return context


def create_default_rag_pipeline(
    retrieval_module: ImageRetrievalModule,
    llm_interface: Optional[LLMInterface] = None
) -> RAGPipeline:
    """
    创建默认的RAG Pipeline

    Args:
        retrieval_module: 检索模块
        llm_interface: LLM接口

    Returns:
        配置好的RAGPipeline实例
    """
    pipeline = RAGPipeline(retrieval_module, llm_interface)

    # 定义默认处理步骤
    def retrieve_step(ctx, rm, llm, **kwargs):
        """检索步骤"""
        topk = kwargs.get('topk', 10)
        distance, ids, paths = rm.retrieval_func(ctx["query"], topk)
        ctx["results"] = [
            {"path": p, "distance": float(d), "index": int(i)}
            for p, d, i in zip(paths, distance, ids)
        ]
        return ctx

    def explain_step(ctx, rm, llm, **kwargs):
        """解释步骤"""
        if llm.available and ctx["results"]:
            ctx["explanation"] = llm.explain_results(ctx["query"], ctx["results"])
        return ctx

    pipeline.add_step("retrieve", retrieve_step)
    pipeline.add_step("explain", explain_step)

    return pipeline
