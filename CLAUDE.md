# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 安装依赖
pip install -e .
# 或使用 uv（更快）
uv pip install -e .

# 特征提取（首次运行，耗时2-4小时）
python image_feature_extract.py

# 启动Flask服务
python flask_app.py
# 访问 http://localhost:5000

# 运行测试
python -m pytest tests/ -v
python -m pytest tests/test_incremental_index_manager.py -v  # 单个测试文件
```

## 架构总览

这是一个**多模态智能检索引擎**，核心流程：**用户查询 → CLIP编码 → Faiss向量检索 → LLM结果解释**。

分层结构（自顶向下）：

| 层 | 文件 | 职责 |
|---|---|---|
| Web/API | [flask_app.py](flask_app.py) | Flask服务，14个REST端点（检索/RAG/Agent/增量索引），应用组装入口 |
| LLM智能层 | [rag_engine.py](rag_engine.py) | 流程编排：查询扩展 → 多路检索 → 去重融合 → AI解释/问答 |
| LLM接口 | [llm_interface.py](llm_interface.py) | OpenAI兼容API封装，提供查询扩展、结果解释、AI问答、多轮对话 |
| Agent工具 | [agent_tools.py](agent_tools.py) | 标准工具定义（兼容Function Calling格式），4个内置工具 |
| 检索模块 | [retrieval_by_faiss.py](retrieval_by_faiss.py) | 三个核心类：`CLIPModel`(特征提取)、`IndexModule`(Faiss索引)、`ImageRetrievalModule`(组装) |
| 增量索引 | [incremental_index_manager.py](incremental_index_manager.py) | 主索引+缓冲区双缓冲架构，支持动态增删图片 |
| 索引构建 | [image_index_builder.py](image_index_builder.py) | 批量CLIP特征提取 |
| 特征提取脚本 | [image_feature_extract.py](image_feature_extract.py) | 离线批量提取全量图片特征并持久化 |
| 配置 | [config/base_config.py](config/base_config.py) | dataclass配置，支持.env环境变量覆盖 |

### 关键设计决策

1. **优雅降级**：LLM未配置时（无`LLM_API_KEY`），`LLMInterface.available`返回False，LLM增强检索自动退化为传统检索，不报错
2. **双缓冲增量索引**：新图片先进入内存buffer（暴力搜索），定期merge到主Faiss IVF索引。删除为逻辑删除（标记+搜索时过滤），重建时才物理清理
3. **CLIP跨模态**：图像和文本编码到同一512维向量空间，`ImageRetrievalModule.retrieval_func()`统一处理文本/图片路径/ndarray三种输入
4. **特征向量L2归一化**：图像特征必须归一化（`feat /= feat.norm(dim=-1, keepdim=True)`），文本特征不需要

### 数据文件

- 特征矩阵：`data/feat_mat-{dataset}-{backbone}.pkl`（N×512 float32 ndarray）
- 映射字典：`data/map_dict-{dataset}-{backbone}.pkl`（`{id: image_path}`）
- 索引状态：`data/index_state/index_state.json` + `buffer_features.pkl`
- 图片目录由`CFG.image_file_dir`配置（默认`train2017/`），Flask通过`/static/img/<filename>`直接提供图片

### API端点分类

- `/search` — 传统检索（POST，支持text+image）
- `/api/search/rag` — LLM增强检索（查询扩展+AI解释）
- `/api/rag/qa` — 基于检索上下文的问答
- `/api/chat` — 多轮对话
- `/api/tools` + `/api/tools/call` — Agent工具列表与调用
- `/api/index/add|remove|merge|rebuild|status` — 增量索引管理
- `/api/status` — 系统状态
