import json
import config
import networkx as nx
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from tqdm import tqdm

# 标签映射
LABEL_MAP = {
    "dis": "疾病",
    "sym": "症状",
    "pro": "治疗",
    "bod": "部位",
    "dru": "药物"
}

# 实体之间的关系定义
RELATION_RULES = {
    ("疾病", "症状"): "有症状",
    ("疾病", "治疗"): "使用治疗方法",
    ("疾病", "药物"): "用药",
    ("症状", "部位"): "影响部位"
}

class MedicalGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_counts = defaultdict(int)
        print("加载模型中，请稍候...")
        self.tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
        self.model = AutoModelForTokenClassification.from_pretrained('uer/roberta-base-finetuned-cluener2020-chinese')
        self.model.eval()
        print("模型加载完成")

    def extract_entities(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = []
        current_entity = ""
        current_label = ""
        for token, label_id in zip(tokens, predictions):
            label = self.model.config.id2label[label_id]
            if label.startswith("B-"):
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = token
                current_label = label[2:]
            elif label.startswith("I-") and current_entity:
                current_entity += token
            else:
                if current_entity:
                    entities.append((current_entity, current_label))
                    current_entity = ""
                    current_label = ""
        if current_entity:
            entities.append((current_entity, current_label))
        return entities

    def build(self, data_path: Path):
        files = list(data_path.glob("*_chunks.json"))
        print(f"发现 {len(files)} 个 chunk 文件，开始构建图谱...")

        for file in tqdm(files, desc="📁 处理文件"):
            with open(file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

            for chunk in tqdm(chunks, desc=f"📄 {file.name}", leave=False):
                text = chunk["text"].replace("[[", "").replace("]]", "")  # 去掉标记
                entities = self.extract_entities(text)

                # 添加节点
                for ent_text, ent_type in entities:
                    label = LABEL_MAP.get(ent_type, "其他")
                    self.graph.add_node(ent_text, type=label)
                    self.entity_counts[label] += 1

                # 构建边
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        h_text, h_type = entities[i]
                        t_text, t_type = entities[j]
                        h_label = LABEL_MAP.get(h_type)
                        t_label = LABEL_MAP.get(t_type)
                        if not h_label or not t_label:
                            continue
                        relation = RELATION_RULES.get((h_label, t_label))
                        if relation:
                            self.graph.add_edge(h_text, t_text, relation=relation)

        return self.graph

    def save(self, output_path: Path):
        nx.write_gexf(self.graph, output_path)
        print(f"图谱保存到: {output_path}")

if __name__ == "__main__":
    builder = MedicalGraphBuilder()
    chunk_dir = Path("data/chunks")
    graph = builder.build(chunk_dir)

    print(f"\n构建完成，共 {len(graph.nodes)} 个节点，{len(graph.edges)} 条边")
    for edge in list(graph.edges(data=True))[:5]:
        print("示例关系：", edge)

    builder.save(Path("data/medical_graph_with_relations.gexf"))