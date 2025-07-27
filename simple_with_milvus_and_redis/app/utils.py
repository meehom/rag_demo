import ollama

def encode(text):
    return ollama.embeddings(model='nomic-embed-text', prompt=text)['embedding']

def generate_answer(question, context):
    prompt = f'根据参考文档回答问题，回答尽量简洁，不超过20个字\n' \
             f'问题是："{question}"\n' \
             f'参考文档是："{context}"'
    stream = ollama.chat(model='qwen2:0.5b', messages=[{'role': 'user', 'content': prompt}], stream=True)
    answer = ""
    for chunk in stream:
        answer += chunk['message']['content']
    return answer.strip()