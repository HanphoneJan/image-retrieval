# -*- coding:utf-8 -*-
"""
@file name  : flask_app.py
@date       : 2026-03-23
@brief      : 多模态RAG检索引擎 - Web服务端
              支持传统检索、RAG增强检索、Agent工具调用
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from datetime import datetime
from retrieval_by_faiss import *
from llm_interface import LLMInterface
from rag_engine import RAGEngine
from agent_tools import create_tools_for_rag_engine
from incremental_index_manager import IncrementalIndexManager
from image_index_builder import ImageIndexBuilder

# 图片目录配置
IMG_ROOT_DIR = CFG.image_file_dir  # 图像数据库所在位置

# 检索模块初始化
with open(CFG.feat_mat_path, 'rb') as f:
    feat_mat = pickle.load(f)
with open(CFG.map_dict_path, 'rb') as f:
    map_dict = pickle.load(f)

ir_model = ImageRetrievalModule(CFG.index_string, CFG.feat_dim, feat_mat, map_dict,
                                CFG.clip_backbone_type, CFG.device)

# 初始化增量索引管理器
index_builder = ImageIndexBuilder(device=CFG.device)
incremental_manager = IncrementalIndexManager(
    main_index=ir_model.index_model,
    index_builder=index_builder,
    buffer_size_threshold=getattr(CFG, 'buffer_size_threshold', 1000),
    state_dir=getattr(CFG, 'index_state_dir', './data/index_state')
)
# 加载之前保存的状态
incremental_manager.load_state(map_dict)
print(f"[FlaskApp] 增量索引管理器已初始化: {incremental_manager.get_status()}")

# 初始化RAG引擎（如果配置了LLM）
llm_interface = LLMInterface(
    base_url=getattr(CFG, 'llm_base_url', None),
    api_key=getattr(CFG, 'llm_api_key', None),
    model=getattr(CFG, 'llm_model', 'gpt-3.5-turbo')
)
rag_engine = RAGEngine(
    retrieval_module=ir_model,
    llm_interface=llm_interface,
    enable_expansion=getattr(CFG, 'rag_enable_expansion', True),
    enable_explanation=getattr(CFG, 'rag_enable_explanation', True),
    index_manager=incremental_manager
)

# 初始化Agent工具（如果启用）
agent_tools = None
if getattr(CFG, 'agent_tools_enabled', True) and llm_interface.available:
    agent_tools = create_tools_for_rag_engine(rag_engine)
    print(f"[FlaskApp] Agent工具已加载: {agent_tools.list_tools()}")

app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    image_file = request.files['image']
    text = request.form['text']
    if image_file:
        def save_img(file, out_dir):
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = '{}-{}'.format(current_time, file.filename)
            path_img = os.path.join(out_dir, file_name)
            file.save(path_img)
            return path_img
        path_to_img = save_img(image_file, app.static_folder)
        query = path_to_img
    else:
        query = text

    # step2: 检索
    distance_result, index_result, path_list = ir_model.retrieval_func(query, CFG.topk)

    # step3: 结果封装
    results = []
    for distance, path in zip(distance_result, path_list):
        basename = os.path.basename(path.rstrip('/\\')) if path and path != 'None' else None
        if not basename:
            continue
        dict_ = {'path': 'static/img/' + basename,
                 'text': f'distance: {distance:.3f}'}
        results.append(dict_)

    # import random
    # names = ['000000000144.jpg', '000000000081.jpg', '000000000154.jpg']
    # name = random.choice(names)
    # paths = os.path.join('static', 'img', name)
    # path_dict = {'path': paths, 'text': 'result1'}
    # results = [path_dict] * 10

    return jsonify(results)


@app.route('/api/search/rag', methods=['POST'])
def search_rag():
    """
    RAG增强检索接口
    支持查询扩展和AI结果解释
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        topk = data.get('topk', CFG.topk)
        use_expansion = data.get('use_expansion', CFG.rag_enable_expansion)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        result = rag_engine.search_and_explain(
            query=query,
            topk=topk,
            use_expansion=use_expansion
        )

        # 转换结果路径为URL可访问格式（使用正斜杠）
        for r in result.get('results', []):
            basename = os.path.basename(r.get('path', '').rstrip('/\\'))
            r['url'] = ('static/img/' + basename) if basename else ''

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/expand', methods=['POST'])
def expand_query():
    """
    查询扩展接口
    使用LLM将用户查询扩展为多个搜索意图
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        num_expansions = data.get('num_expansions', 3)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        if not llm_interface.available:
            return jsonify({"error": "LLM not configured"}), 503

        expanded = llm_interface.expand_query(query, num_expansions=num_expansions)

        return jsonify({
            "original_query": query,
            "expanded_queries": expanded,
            "count": len(expanded)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/rag/qa', methods=['POST'])
def rag_qa():
    """
    RAG问答接口
    基于检索上下文回答用户问题
    """
    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        context_size = data.get('context_size', CFG.rag_context_size)

        if not question:
            return jsonify({"error": "Question is required"}), 400

        result = rag_engine.rag_qa(
            question=question,
            topk=context_size
        )

        # 转换路径
        for r in result.get('context', []):
            basename = os.path.basename(r.get('path', '').rstrip('/\\'))
            r['url'] = ('static/img/' + basename) if basename else ''

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    多轮对话接口
    基于检索上下文进行对话
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '')
        history = data.get('history', [])
        context_size = data.get('context_size', CFG.rag_context_size)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        result = rag_engine.chat_with_images(
            query=query,
            history=history,
            topk=context_size
        )

        # 转换路径
        for r in result.get('context', []):
            basename = os.path.basename(r.get('path', '').rstrip('/\\'))
            r['url'] = ('static/img/' + basename) if basename else ''

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/tools', methods=['GET'])
def list_tools():
    """
    获取可用的Agent工具列表
    """
    if not agent_tools:
        return jsonify({
            "tools": [],
            "llm_available": llm_interface.available
        })

    tools = [tool.to_dict() for tool in agent_tools._tools.values()]

    return jsonify({
        "tools": tools,
        "llm_available": llm_interface.available
    })


@app.route('/api/tools/call', methods=['POST'])
def call_tool():
    """
    调用Agent工具
    """
    try:
        if not agent_tools:
            return jsonify({"error": "Agent tools not available"}), 503

        data = request.get_json() or {}
        tool_name = data.get('tool', '')
        arguments = data.get('arguments', {})

        if not tool_name:
            return jsonify({"error": "Tool name is required"}), 400

        result = agent_tools.execute_tool(tool_name, arguments)

        # 转换结果中的路径
        if result.get('success'):
            tool_result = result.get('result', {})
            if 'results' in tool_result:
                for r in tool_result['results']:
                    if 'path' in r:
                        r['url'] = 'static/img/' + os.path.basename(r['path'])

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """
    获取系统状态
    """
    index_status = incremental_manager.get_status()
    return jsonify({
        "status": "running",
        "llm_available": llm_interface.available,
        "llm_model": llm_interface.model if llm_interface.available else None,
        "rag_enabled": getattr(CFG, 'rag_enable_explanation', True),
        "agent_tools_enabled": getattr(CFG, 'agent_tools_enabled', True) and llm_interface.available,
        "index_type": CFG.index_string,
        "feat_dim": CFG.feat_dim,
        "index": index_status
    })


# ==================== 增量索引管理 API ====================

@app.route('/api/index/add', methods=['POST'])
def index_add():
    """
    添加新图片到索引
    请求体: {"image_paths": ["path1", "path2"]}
    """
    try:
        data = request.get_json() or {}
        image_paths = data.get('image_paths', [])

        if not image_paths:
            return jsonify({"error": "image_paths is required"}), 400

        # 验证路径存在
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        invalid_paths = list(set(image_paths) - set(valid_paths))

        if not valid_paths:
            return jsonify({
                "error": "No valid image paths provided",
                "invalid_paths": invalid_paths
            }), 400

        # 添加到索引
        result = incremental_manager.add_images(valid_paths)

        # 保存状态
        incremental_manager.save_state()

        return jsonify({
            "success": True,
            "added": result["added"],
            "ids": result["ids"],
            "failed": result["failed"] + invalid_paths,
            "status": incremental_manager.get_status()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/index/remove', methods=['POST'])
def index_remove():
    """
    从索引中删除图片（逻辑删除）
    请求体: {"image_ids": [1001, 1002]}
    """
    try:
        data = request.get_json() or {}
        image_ids = data.get('image_ids', [])

        if not image_ids:
            return jsonify({"error": "image_ids is required"}), 400

        # 执行删除
        result = incremental_manager.remove_images(image_ids)

        # 保存状态
        incremental_manager.save_state()

        return jsonify({
            "success": True,
            "removed": result["removed"],
            "not_found": result["not_found"],
            "status": incremental_manager.get_status()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/index/merge', methods=['POST'])
def index_merge():
    """
    将缓冲区合并到主索引
    """
    try:
        result = incremental_manager.merge_buffer()

        # 保存状态
        incremental_manager.save_state()

        return jsonify({
            "success": True,
            "merged": result["merged"],
            "total": result["total"],
            "status": incremental_manager.get_status()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/index/rebuild', methods=['POST'])
def index_rebuild():
    """
    重建索引（清理已删除的数据）
    注意：这是一个耗时操作
    """
    try:
        # 重建索引
        new_index = incremental_manager.rebuild_index(
            all_features=feat_mat,
            all_paths=[map_dict[i] for i in range(len(map_dict))],
            index_string=CFG.index_string
        )

        # 更新ir_model中的索引
        ir_model.index_model = new_index

        # 保存状态
        incremental_manager.save_state()

        return jsonify({
            "success": True,
            "total_vectors": new_index.get_total_count(),
            "status": incremental_manager.get_status()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/index/status', methods=['GET'])
def index_status():
    """
    获取索引状态
    """
    try:
        status = incremental_manager.get_status()
        return jsonify({
            "success": True,
            "status": status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/static/img/<path:filename>')
def serve_image(filename):
    """
    直接提供图片文件，替代软链接方案
    """
    return send_from_directory(IMG_ROOT_DIR, filename)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='localhost', port=5000)
