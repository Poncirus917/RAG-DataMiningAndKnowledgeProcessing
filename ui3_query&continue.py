import streamlit as st
from pymilvus import MilvusClient, connections, RRFRanker, AnnSearchRequest
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch

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
def hybrid_search(_client, _embedder, query, limit=5, alpha=0.7):
    """
    混合检索 + 方法三 rerank：标题/摘要语义相似度加权排序
    alpha: 原始Milvus得分占比，(1 - alpha)为query-summary相似度占比
    """
    # 1. 获取 query 的向量表示
    embeddings = _embedder.encode_documents([query])
    dense_data = np.array(embeddings["dense"][0]).reshape(1, -1)
    csr_matrix = embeddings["sparse"][0].tocsr()
    if csr_matrix.ndim != 2:
        csr_matrix = csr_matrix.reshape(1, -1)

    # 2. 构建 Milvus hybrid search 请求
    reqs = [
        AnnSearchRequest(
            data=dense_data,
            anns_field="dense_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        ),
        AnnSearchRequest(
            data=csr_matrix,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit
        )
    ]

    # 3. 执行混合搜索，默认使用 RRF 排序器
    search_results = _client.hybrid_search(
        collection_name="text_collection",
        reqs=reqs,
        ranker=RRFRanker(k=60),
        output_fields=["text"]
    )

    hits = search_results[0]
    texts = [hit.entity['text'] for hit in hits]

    # 4. 计算 query 与每条检索结果摘要的语义相似度
    query_emb = _embedder.encode_queries([query])['dense'][0]
    doc_embs = _embedder.encode_documents(texts)['dense']

    query_emb = np.array(query_emb)
    doc_embs = np.array(doc_embs)

    dot_products = doc_embs @ query_emb
    query_norm = np.linalg.norm(query_emb)
    doc_norms = np.linalg.norm(doc_embs, axis=1)
    cosine_scores = dot_products / (doc_norms * query_norm + 1e-8)

    # 5. 融合原始得分与语义相似度进行 rerank
    reranked = []
    for i, hit in enumerate(hits):
        orig_score = getattr(hit, 'score', 0.0)  # 防止报错
        semantic_score = cosine_scores[i]
        final_score = alpha * orig_score + (1 - alpha) * semantic_score
        reranked.append((final_score, hit.entity['text']))

    reranked.sort(key=lambda x: x[0], reverse=True)

    return [text for _, text in reranked]


# ------------------ LLM生成模块 ------------------
def generate_response(_tokenizer, _model, context, query, enable_thinking=True):
    """生成回答，enable_thinking 控制是否输出 <think> 标签"""
    SYSTEM_PROMPT = "基于以下上下文回答问题："
    USER_PROMPT = f"""
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=512
    )
    response = _tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    ).strip("\n")
    return response


def continue_response(_tokenizer, _model, query, previous_answer):
    """
    续写回答，不带 <think> 标签，生成完整连贯答案
    """
    SYSTEM_PROMPT = (
        "请根据用户提问和之前的回答，生成一个更完整、更连贯、"
        "风格统一的完整答案。请勿输出任何 <think> 内容。"
    )
    USER_PROMPT = f"""
    <question>
    {query}
    </question>
    <previous_answer>
    {previous_answer}
    </previous_answer>
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 续写时关闭 thinking 模式
    )

    inputs = _tokenizer([text], return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=512
    )
    response = _tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    ).strip("\n")
    return response


# ------------------ Streamlit界面 ------------------
st.title("RAG+Qwen 智能问答系统 💬")

# 初始化组件（假设你已有 init_components）
try:
    client, embedder, tokenizer, model = init_components()
except Exception as e:
    st.error(f"组件初始化失败: {str(e)}")
    st.stop()

# 清空聊天按钮
if st.button("🧹 清空对话"):
    st.session_state.messages = []
    st.rerun()

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！请输入您的问题，我将结合知识库为您解答。"}]

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 下载聊天记录按钮
chat_text = "\n\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
st.download_button("💬 下载聊天记录", data=chat_text, file_name="chat_history.txt")

# 处理用户输入（新问题）
if user_input := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.spinner("第 1 步：正在检索知识库..."):
        try:
            context = hybrid_search(client, embedder, user_input)
            context_str = "\n".join(context[:3])  # 取前3条结果

            with st.expander("📚 参考上下文（来自知识库）", expanded=False):
                for i, chunk in enumerate(context[:3]):
                    st.markdown(f"**片段 {i+1}:**\n{chunk}")

            with st.spinner("第 2 步：正在调用大模型生成回答..."):
                # 新问题首次回答，启用 thinking 模式，带 <think>
                response = generate_response(tokenizer, model, context_str, user_input, enable_thinking=True)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

        except Exception as e:
            st.error(f"处理失败: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"出错了: {str(e)}"})

# 续写按钮（存在上一次助手回答时出现）
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    if st.button("继续生成更多内容"):
        # 找到最近一次用户问题和助手回答
        last_assistant_answer = None
        last_user_question = None

        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant" and last_assistant_answer is None:
                last_assistant_answer = msg["content"]
            elif msg["role"] == "user" and last_user_question is None:
                last_user_question = msg["content"]
            if last_assistant_answer and last_user_question:
                break

        if last_user_question and last_assistant_answer:
            with st.spinner("继续生成中..."):
                continued_answer = continue_response(tokenizer, model, last_user_question, last_assistant_answer)

            # 用续写内容替换最后一次助手消息，保证连贯
            st.session_state.messages[-1]["content"] = continued_answer
            st.rerun()