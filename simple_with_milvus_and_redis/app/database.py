import sys
sys.path.append("/xx")
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config.settings import settings

def connect_milvus():
    connections.connect(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT
    )
    print("✅ Connected to Milvus")

def create_collection():
    connect_milvus()

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.DIMENSION),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "FAQ embeddings")
    collection = Collection(settings.COLLECTION_NAME, schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 100}
    }
    collection.create_index("embedding", index_params)
    collection.load()
    return collection