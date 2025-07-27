
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MILVUS_HOST: str = "xx.xx.xx.xx"
    MILVUS_PORT: str = "19530"
    REDIS_HOST: str = "xx.xx.xx.xx"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "mypassword"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_MODEL: str = "qwen2:0.5b"
    COLLECTION_NAME: str = "faq_collection"
    DIMENSION: int = 768

    class Config:
        env_file = ".env"

settings = Settings()