import ollama
import numpy as np
from tqdm import tqdm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 连接到 Milvus 服务（默认本地）
connections.connect(host='xx.xx.xx.xx', port='19530')

# 定义常量
DIMENSION = 768  # 根据你的 embedding 模型维度修改（比如 nomic-embed-text 是 768）
COLLECTION_NAME = "faq_collection"

# 创建集合（如果不存在）
def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "FAQ embeddings")
    collection = Collection(COLLECTION_NAME, schema)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 100}
    }
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

# 编码函数
def encode(text):
    return ollama.embeddings(model='nomic-embed-text', prompt=text)['embedding']

# 读取文档并分段
def load_chunks(file_path):
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(line)
    return chunks

# 插入数据到 Milvus
def insert_data(collection, chunks):
    embeddings = []
    texts = []
    for chunk in tqdm(chunks, desc="生成 embeddings"):
        emb = encode(chunk)
        embeddings.append(emb)
        texts.append(chunk)

    data = [
        embeddings,  # embedding 字段
        texts        # text 字段
    ]
    collection.insert(data)
    print(f"已插入 {len(chunks)} 条数据到 Milvus")

# 查询最相似的 chunk
def search_similar(collection, question):
    emb = encode(question)
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([emb], "embedding", search_params, limit=1)

    # 获取最相似的文本
    result_id = results[0].ids[0]
    result_text = collection.query(expr=f"id == {result_id}", output_fields=["text"])[0]['text']
    return result_text

# 主程序
def main():
    collection = create_collection()

    # 如果集合为空，则插入数据
    if collection.num_entities == 0:
        chunks = load_chunks("aa.txt")
        insert_data(collection, chunks)

    while True:
        question = input("请输入一个问题（输入 '退出' 以结束）: ")
        if question == '退出':
            break

        # 检索最相关文本
        relevant_text = search_similar(collection, question)

        # 构造 prompt
        prompt = f'根据参考文档回答问题，回答尽量简洁，不超过20个字\n' \
                 f'问题是："{question}"\n' \
                 f'参考文档是："{relevant_text}"'

        print(f'Prompt:\n{prompt}')

        # 获取答案
        stream = ollama.chat(model='qwen2:0.5b', messages=[{'role': 'user', 'content': prompt}], stream=True)
        print('Answer:')
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
        print()

if __name__ == '__main__':
    main()