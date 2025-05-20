from bs4 import BeautifulSoup
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType
import numpy as np
import os

def extract_chinese_text(html_content):
    """从HTML内容中提取纯文本"""
    soup = BeautifulSoup(html_content, 'lxml')
    # 移除不需要的标签
    for tag in soup(['script', 'style', 'noscript', 'footer', 'header', 'aside', 'nav', 'button', 'input']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)

def process_html_directory(directory):
    """批量处理HTML目录"""
    docs = []
    file_list = []
    
    # 遍历目录获取HTML文件
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                text = extract_chinese_text(html_content)
                if text:  # 过滤空内容
                    docs.append(text)
                    file_list.append(filename)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
    
    print(f"成功加载 {len(docs)} 个文档")
    return docs, file_list

# 初始化嵌入模型
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',
    device='mps',
    use_fp16=True
)

# 处理整个目录
html_dir = "./leuka_data/html"
docs, file_list = process_html_directory(html_dir)

# 批量生成嵌入
if docs:
    docs_embeddings = bge_m3_ef.encode_documents(docs)
    print(f"生成 {len(docs)} 个文档的嵌入向量")

# 转换稀疏向量格式
def coo_to_sparse_dict(coo_array):
    """转换COO稀疏矩阵为字典格式"""
    return {int(col): float(val) for col, val in zip(coo_array.col, coo_array.data)}

# 准备批量数据
batch_data = []
for idx in range(len(docs)):
    sparse_vector = coo_to_sparse_dict(docs_embeddings["sparse"][idx])
    batch_data.append({
        "text": docs[idx],
        "dense_vector": docs_embeddings["dense"][idx].tolist(),
        "sparse_vector": dict(sorted(sparse_vector.items()))  # 保持索引有序
    })

# 连接Milvus
client = MilvusClient("milvus.db")

# 重建集合（生产环境应去掉此步骤）
if client.has_collection("text_collection"):
    client.drop_collection("text_collection")

# 创建集合Schema（定义数据结构）
schema = client.create_schema(
    auto_id=True,   # 主键ID自动生成，无需手动传入
    enable_dynamic_field=False # 禁用动态字段，必须严格按照 schema 定义的字段插入数据
)
# 添加主键字段：ID，自增、64位整型
schema.add_field(
    field_name="id",
    datatype=DataType.INT64, 
    is_primary=True, # 作为主键
    auto_id=True     # 自动生成
)
# 添加文本字段：保存原始文本数据
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
# 添加稠密向量字段：保存BGE-M3生成的嵌入向量
schema.add_field(
    field_name="dense_vector", 
    datatype=DataType.FLOAT_VECTOR,
    dim=bge_m3_ef.dim["dense"] # 嵌入向量维度
)
# 添加稀疏向量字段：保存稀疏表示
schema.add_field(
    field_name="sparse_vector",
    datatype=DataType.SPARSE_FLOAT_VECTOR
)

# 创建带索引的集合
index_params = client.prepare_index_params()
# 为稠密向量添加 IVF_FLAT 索引（适用于中等数据量的快速检索
index_params.add_index(
    field_name="dense_vector",
    index_type="IVF_FLAT",  # 倒排文件索引+线性扫描
    metric_type="IP",       # 相似度计算方式：内积（Inner Product）
    params={"nlist": 128}   # 聚类中心数目，越大越精确但速度稍慢
)
# 为稀疏向量添加 SPARSE_INVERTED_INDEX（适用于高维稀疏特征）
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX", # 稀疏向量专用的倒排索引
    metric_type="IP",                   # 相似度计算方式：内积
    params={"drop_ratio_build": 0.2}    # 构建索引时忽略最频繁的20%项（加速检索）
)
# 创建集合 text_collection
client.create_collection(
    collection_name="text_collection",  # 集合名称
    schema=schema,                      # 使用上面定义的 Schema
    index_params=index_params,          # 使用上面定义的索引
    consistency_level="Strong"          # 强一致性（保证写入后立即可查）
)

# 批量插入数据
if batch_data:
    # 分批插入（每批100条）
    batch_size = 100
    total = len(batch_data)
    for i in range(0, total, batch_size):
        batch = batch_data[i:i+batch_size]
        insert_result = client.insert("text_collection", batch)
        print(f"已插入 {min(i+batch_size, total)}/{total} 条数据，ID范围: {insert_result['ids'][0]}~{insert_result['ids'][-1]}")
        
    print(f"成功插入 {total} 条文档数据")
else:
    print("没有有效数据需要插入")

# 验证数据
print("\n集合信息:", client.describe_collection("text_collection"))
print("总文档数:", client.num_entities("text_collection"))
