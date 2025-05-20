import streamlit as st
from pymilvus import MilvusClient, connections, RRFRanker, AnnSearchRequest
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import jieba  # 用于中文分词

# ------------------ 初始化模块 ------------------
@st.cache_resource
def init_components():
    """初始化所有组件（缓存避免重复加载）"""
    # Milvus连接
    client = MilvusClient("/home/prts/milvus_data/milvus.db")

    # 嵌入模型
    embedder = BGEM3EmbeddingFunction(
        model_name_or_path='/mnt/d/models/bge-m3',
        device='cuda',
        use_fp16=False
    )

    # Qwen模型
    model_path = "/mnt/d/models/qwen3_0.6b"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )

    return client, embedder, tokenizer, model

# ------------------ RAG搜索模块 ------------------
def hybrid_search(_client, _embedder, query, limit=5):
    """混合检索 + 关键词覆盖率 reranking"""
    embeddings = _embedder.encode_documents([query])

    # 密集、稀疏向量
    dense_data = np.array(embeddings["dense"][0]).reshape(1, -1)
    sparse_data = embeddings["sparse"][0].tocsr()

    if sparse_data.ndim != 2:
        sparse_data = sparse_data.reshape(1, -1)

    # 检索请求
    reqs = [
        AnnSearchRequest(
            data=dense_data,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        ),
        AnnSearchRequest(
            data=sparse_data,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        )
    ]

    search_results = _client.hybrid_search(
        collection_name="text_collection",
        reqs=reqs,
        ranker=RRFRanker(k=60),
        output_fields=["text"]
    )

    # 初始候选
    candidates = []
    for hit in search_results[0]:
        text = hit['entity']['text']
        score = hit['distance']  # 替代 hit['score']
        candidates.append({"text": text, "score": score})

    # rerank：基于关键词覆盖
    keywords = list(set(query.strip().lower().split()))
    for item in candidates:
        match_count = sum(1 for kw in keywords if kw in item["text"].lower())
        item["score"] += 0.1 * match_count  # 每个关键词 +0.1 分

    # 按最终得分排序
    candidates.sort(key=lambda x: x["score"], reverse=True)

    return [item["text"] for item in candidates[:limit]]


# ------------------ LLM生成模块 ------------------
def generate_response(_tokenizer, _model, context, query):
    """生成回答"""
    SYSTEM_PROMPT = "基于以下上下文回答问题："
    USER_PROMPT = f"""
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """

    # 构建输入
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    # 生成回答
    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=512  # 控制生成长度
    )

    # 解析输出
    response = _tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    ).strip("\n")

    return response

# ------------------ Streamlit界面 ------------------
st.title("RAG+Qwen 智能问答系统 💬")

# 初始化组件
try:
    client, embedder, tokenizer, model = init_components()
except Exception as e:
    st.error(f"组件初始化失败: {str(e)}")
    st.stop()

# 清空聊天按钮（优化项 3）
if st.button("🧹 清空对话"):
    st.session_state.messages = []
    st.rerun()

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！请输入您的问题，我将结合知识库为您解答。"}]

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 下载聊天记录按钮（优化项 4）
chat_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
st.download_button("💬 下载聊天记录", data=chat_text, file_name="chat_history.txt")

# 处理用户输入
if user_input := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 处理回答
    with st.spinner("第 1 步：正在检索知识库..."):
        try:
            context = hybrid_search(client, embedder, user_input)
            context_str = "\n".join(context[:3])  # 取前3条结果

            # 展示参考上下文（优化项 1）
            with st.expander("📚 参考上下文（来自知识库）", expanded=False):
                for i, chunk in enumerate(context[:3]):
                    st.markdown(f"**片段 {i+1}:**\n{chunk}")

            # 生成回答
            with st.spinner("第 2 步：正在调用大模型生成回答..."):
                response = generate_response(tokenizer, model, context_str, user_input)

            # 添加并显示回答
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

        except Exception as e:
            st.error(f"处理失败: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"出错了: {str(e)}"})
