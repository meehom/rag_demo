import redis
from config.settings import settings

def get_redis_client():
    return redis.StrictRedis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD,
        decode_responses=True
    )

def cache_set(question: str, answer: str, ttl=3600):
    client = get_redis_client()
    client.setex(f"qa:{question}", ttl, answer)

def cache_get(question: str):
    client = get_redis_client()
    return client.get(f"qa:{question}")