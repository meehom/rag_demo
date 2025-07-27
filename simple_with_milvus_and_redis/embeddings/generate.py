import os
import sys
sys.path.append("/xxx")
from tqdm import tqdm
from app.database import create_collection
from app.utils import encode

def load_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def insert_data():
    collection = create_collection()
    file_path = "aa.txt"
    chunks = load_chunks(file_path)

    embeddings = []
    texts = []
    for chunk in tqdm(chunks, desc="生成 embeddings"):
        emb = encode(chunk)
        embeddings.append(emb)
        texts.append(chunk)

    data = [embeddings, texts]
    collection.insert(data)
    print(f"✅ 已插入 {len(chunks)} 条数据到 Milvus")

if __name__ == "__main__":
    insert_data()