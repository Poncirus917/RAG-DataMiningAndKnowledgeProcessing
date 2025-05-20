from pymilvus import MilvusClient, connections, RRFRanker, AnnSearchRequest
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import numpy as np
import scipy.sparse as sp

# 初始化连接
client = MilvusClient("milvus.db")
embedder = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3', # Specify the model name
    device='mps', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

def hybrid_search(query, limit=5):
    embeddings = embedder.encode_documents([query])

    dense_data = np.array(embeddings["dense"][0]).reshape(1, -1)  # 关键修复点1
    
    # 处理稀疏向量：转换为CSR格式并验证维度
    coo_matrix = embeddings["sparse"][0]
    csr_matrix = coo_matrix.tocsr()
    
    # 确保稀疏矩阵是二维的
    if csr_matrix.ndim != 2:
        csr_matrix = csr_matrix.reshape(1, -1)

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

    ranker = RRFRanker(k=60)

    # Perform hybrid search using MilvusClient
    search_results = client.hybrid_search(
        collection_name="text_collection",
        reqs=reqs,
        ranker=ranker,
        output_fields=["id", "text"]
    )

    final_results = []
    for hits in search_results:
        print("TopK results:")
        for hit in hits:
            result = {
                "id": hit['id'],  # Access the 'id' field from the entity
                "score": hit['distance'],  # Access the 'distance' attribute
                "text": hit['entity']['text']
            }
            final_results.append(result)

    return sorted(final_results, key=lambda x: x["score"], reverse=True)

if __name__ == "__main__":
    query = "如何预防肿瘤？"
    results = hybrid_search(query)
    
    print(f"\n与『{query}』最匹配的{len(results)}篇文章：")
    for i, res in enumerate(results, 1):
        print(f"\n结果{i} (综合得分: {res['score']:.4f})：")
        print(f"ID: {res['id']}")
        print(f"内容摘要：{res['text'][:200]}...")
