from pymilvus import MilvusClient, DataType

client = MilvusClient("milvus.db")

# 修改后的验证部分
print("\n集合信息:", client.describe_collection("text_collection"))

# 获取统计信息
stats = client.get_collection_stats("text_collection")
print("集合统计:", stats)
print("总文档数:", stats["row_count"])

# 抽样查询验证
query_result = client.query(
    collection_name="text_collection",
    limit=3,
    output_fields=["id", "text"]

)
print("\n抽样查询结果:", query_result)
