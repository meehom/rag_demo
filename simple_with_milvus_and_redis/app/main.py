
import sys
sys.path.append("/xx")

from fastapi import FastAPI, HTTPException
from app.utils import encode, generate_answer
from app.database import create_collection
from app.cache import cache_get, cache_set
from config.settings import settings
from pydantic import BaseModel


app = FastAPI(title="Q&A API")

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    from app.database import connect_milvus
    connect_milvus()

@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question

    # 1. 先查缓存
    cached = cache_get(question)
    if cached:
        return {"source": "cache", "answer": cached}

    # 2. 向量检索
    try:
        collection = create_collection()  # 注意：这里应该是 get_collection，不是 connect_milvus
        question_emb = encode(question)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = collection.search([question_emb], "embedding", search_params, limit=1)
        result_id = results[0].ids[0]
        result = collection.query(expr=f"id == {result_id}", output_fields=["text"])[0]['text']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus error: {e}")

    # 3. 生成回答
    answer = generate_answer(question, result)

    # 4. 写入缓存
    cache_set(question, answer)

    return {"source": "vector", "answer": answer, "context": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)