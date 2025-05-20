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
    client = MilvusClient("milvus.db")
    
    # 嵌入模型
    embedder = BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3',
        device='mps',
        use_fp16=False
    )
    
    # Qwen模型
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    return client, embedder, tokenizer, model

# ------------------ RAG搜索模块 ------------------
def hybrid_search(_client, _embedder, query, limit=5):
    """混合检索函数"""
    embeddings = _embedder.encode_documents([query])
    
    # 处理密集向量
    dense_data = np.array(embeddings["dense"][0]).reshape(1, -1)
    
    # 处理稀疏向量
    csr_matrix = embeddings["sparse"][0].tocsr()
    if csr_matrix.ndim != 2:
        csr_matrix = csr_matrix.reshape(1, -1)
    
    # 构建搜索请求
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
    
    # 执行混合搜索
    search_results = _client.hybrid_search(
        collection_name="text_collection",
        reqs=reqs,
        ranker=RRFRanker(k=60),
        output_fields=["text"]
    )
    
    return [hit['entity']['text'] for hit in search_results[0]]

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

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！请输入您的问题，我将结合知识库为您解答。"}]

# 显示历史消息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 处理用户输入
if user_input := st.chat_input("请输入您的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # 处理回答
    with st.spinner("正在检索知识库..."):
        try:
            # RAG检索
            context = hybrid_search(client, embedder, user_input)
            context_str = "\n".join(context[:3])  # 取前3条结果

            # --------- 历史回溯优化：将历史对话作为上下文的一部分传入生成模块 ---------
            # 取最近5轮（用户+助手）的消息（可调整）
            recent_dialogs = st.session_state.messages[-10:]  # 5轮对话，用户和助手各10条消息

            # 拼接历史对话文本，格式为角色+内容，方便模型理解
            history_text = ""
            for msg in recent_dialogs:
                role = "用户" if msg["role"] == "user" else "助手"
                history_text += f"{role}: {msg['content']}\n"

            # 将历史对话和检索到的知识库内容拼接为新的上下文
            enhanced_context = context_str + "\n\n历史对话记录:\n" + history_text
            
            with st.spinner("生成回答中..."):
                response = generate_response(tokenizer, model, enhanced_context, user_input)
            # ----------------------------------------------------------------------

            # 添加并显示回答
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            
        except Exception as e:
            st.error(f"处理失败: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"出错了: {str(e)}"})
