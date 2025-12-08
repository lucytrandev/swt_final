import os
from functools import lru_cache
from typing import Optional, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB", "smart_resume")


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    return MongoClient(MONGODB_URI)


def get_collections() -> Tuple[Collection, Collection]:
    db = get_client()[DB_NAME]
    return db["candidates"], db["jobs"]


def ensure_seed_jobs(seed_data: list) -> None:
    _, jobs = get_collections()
    if jobs.estimated_document_count() == 0 and seed_data:
        jobs.insert_many(seed_data)


def close_client() -> None:
    client = get_client()
    client.close()
    get_client.cache_clear()
