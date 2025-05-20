# medical_graph/neo4j_loader.py

from py2neo import Graph, Node, Relationship
import networkx as nx

def load_graph_to_neo4j(nx_graph: nx.DiGraph, neo_graph: Graph):
    neo_graph.delete_all()  # 可选：清空 Neo4j 图
    node_map = {}

    for name, attr in nx_graph.nodes(data=True):
        node = Node("Entity", name=name, type=attr.get("type", "未知"))
        neo_graph.create(node)
        node_map[name] = node

    for src, tgt, attr in nx_graph.edges(data=True):
        rel = Relationship(node_map[src], attr.get("relation", "相关"), node_map[tgt])
        neo_graph.create(rel)

    print("图谱已导入 Neo4j。")

if __name__ == "__main__":
    # 加载 GEXF 图谱文件
    graph_path = "data/medical_graph_with_relations.gexf"
    nx_graph = nx.read_gexf(graph_path)

    # 连接本地 Neo4j
    neo_graph = Graph("bolt://localhost:7687", auth=("neo4j", "lzj050211ZS"))

    load_graph_to_neo4j(nx_graph, neo_graph)