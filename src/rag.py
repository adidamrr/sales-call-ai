from functools import lru_cache
from pathlib import Path
import sys
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import settings


KNOWLEDGE_BASE_DIR = Path("knowledge_base")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_knowledge_documents():
    documents = []

    for file_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.md")):
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "doc_type": file_path.stem,
                "text": text,
            }
        )

    return documents


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def get_qdrant_client():
    return QdrantClient(url=settings.QDRANT_URL)


def get_embedding_dimension(model):
    dimension = model.get_sentence_embedding_dimension()
    if dimension is not None:
        return dimension

    sample_embedding = model.encode("test")
    return len(sample_embedding)


def create_qdrant_collection():
    model = get_embedding_model()
    vector_size = get_embedding_dimension(model)
    client = get_qdrant_client()

    if client.collection_exists(settings.QDRANT_COLLECTION):
        client.delete_collection(settings.QDRANT_COLLECTION)

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def index_knowledge_base():
    documents = load_knowledge_documents()
    model = get_embedding_model()
    client = get_qdrant_client()
    create_qdrant_collection()

    points = []

    for document in documents:
        chunks = split_text_into_chunks(document["text"])

        for chunk_index, chunk in enumerate(chunks):
            embedding = model.encode(chunk, normalize_embeddings=True).tolist()
            point_id = str(uuid5(NAMESPACE_URL, f"{document['source']}:{chunk_index}"))
            payload = {
                "text": chunk,
                "source": document["source"],
                "doc_type": document["doc_type"],
                "chunk_index": chunk_index,
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

    if points:
        client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

    return len(points)


def search_knowledge(query: str, top_k: int = 5):
    model = get_embedding_model()
    client = get_qdrant_client()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    try:
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k,
        )
        points = response.points
    except AttributeError:
        points = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k,
        )

    results = []

    for point in points:
        payload = point.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "score": point.score,
                "source": payload.get("source", ""),
                "doc_type": payload.get("doc_type", ""),
            }
        )

    return results


def main():
    if len(sys.argv) != 2 or sys.argv[1] != "index":
        print("Usage: python -m src.rag index")
        raise SystemExit(1)

    chunks_count = index_knowledge_base()
    print(f"Indexed knowledge base: {chunks_count} chunks")


if __name__ == "__main__":
    main()
