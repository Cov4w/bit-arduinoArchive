
import json
import hashlib
from pathlib import Path

import torch
import chromadb

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 경로 및 설정
# ==========================================

INPUT_PATH = Path("data/processed/arduino_chunks_v3.jsonl")
DB_PATH = Path("data/processed/chroma_db")

COLLECTION_NAME = "arduino_docs_v3"

MODEL_NAME = "BAAI/bge-m3"

BATCH_SIZE = 32


# ==========================================
# 2. 실행 환경 확인
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("Arduino RAG Embedding")
print("=" * 50)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU를 사용할 수 없어 CPU로 실행합니다.")


# ==========================================
# 3. 데이터 로드
# ==========================================

def load_chunks(path):

    chunks = []

    with open(path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file):

            if not line.strip():
                continue

            try:
                data = json.loads(line)

                content = data.get("content", "")

                if not isinstance(content, str) or not content.strip():
                    continue

                metadata = data.get("metadata", {})

                if not isinstance(metadata, dict):
                    metadata = {}

                # metadata가 중첩되어 있지 않은 필드도 보존
                for key, value in data.items():

                    if key not in ("id", "content", "metadata", "embedding"):
                        metadata.setdefault(key, value)

                # ChromaDB에서 허용하는 메타데이터 형식으로 변환
                clean_metadata = {}

                for key, value in metadata.items():

                    if value is None:
                        continue

                    if isinstance(value, (str, int, float, bool)):
                        clean_metadata[str(key)] = value

                    elif isinstance(value, (list, dict)):
                        clean_metadata[str(key)] = json.dumps(
                            value,
                            ensure_ascii=False
                        )

                chunk_id = data.get("id")

                if not chunk_id:
                    unique_text = f"{line_number}:{content}"

                    chunk_id = hashlib.sha256(
                        unique_text.encode("utf-8")
                    ).hexdigest()

                chunks.append({
                    "id": str(chunk_id),
                    "content": content,
                    "metadata": clean_metadata
                })

            except json.JSONDecodeError:
                print(f"JSON 오류: {line_number + 1}번째 줄")

    return chunks


# ==========================================
# 4. BGE-M3 모델 로드
# ==========================================

def load_model():

    print("\n[1] Embedding 모델 로드")

    model = SentenceTransformer(
        MODEL_NAME,
        device=DEVICE
    )

    print("모델 로드 완료")
    print(f"Embedding 차원: {model.get_sentence_embedding_dimension()}")
    print(f"실제 사용 장치: {model.device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU 메모리: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    return model


# ==========================================
# 5. ChromaDB 연결
# ==========================================

def load_collection():

    print("\n[2] ChromaDB 연결")

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Collection: {COLLECTION_NAME}")
    print(f"기존 데이터: {collection.count()}")

    return collection


# ==========================================
# 6. Embedding 생성 및 저장
# ==========================================

def create_embeddings(model, collection, chunks):

    print("\n[3] Embedding 생성 및 저장")

    total = len(chunks)

    for start in tqdm(range(0, total, BATCH_SIZE)):

        batch = chunks[start:start + BATCH_SIZE]

        contents = [
            item["content"] for item in batch
        ]

        embeddings = model.encode(
            contents,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        collection.upsert(
            ids=[
                item["id"] for item in batch
            ],

            documents=contents,

            metadatas=[
                item["metadata"] for item in batch
            ],

            embeddings=embeddings.tolist()
        )

    print("\nEmbedding 저장 완료")
    print(f"저장된 전체 데이터: {collection.count()}")


# ==========================================
# 7. 검색 테스트
# ==========================================

def search_test(model, collection):

    print("\n[4] 검색 테스트")

    query = "아두이노에서 LED를 켜고 끄는 방법은?"

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    print(f"\n질문: {query}")
    print("=" * 50)

    for i in range(len(results["ids"][0])):

        print(f"\n[검색 결과 {i + 1}]")

        print(f"ID: {results['ids'][0][i]}")

        print(
            f"Distance: "
            f"{results['distances'][0][i]:.4f}"
        )

        metadata = results["metadatas"][0][i]

        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Category: {metadata.get('category', 'N/A')}")
        print(f"Type: {metadata.get('content_type', 'N/A')}")

        print("\nContent:")
        print(results["documents"][0][i][:500])

        print("-" * 50)


# ==========================================
# 8. 메인 실행
# ==========================================

def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"입력 파일을 찾을 수 없습니다: {INPUT_PATH}"
        )

    chunks = load_chunks(INPUT_PATH)

    print(f"\n로드된 Chunk: {len(chunks)}")

    model = load_model()

    collection = load_collection()

    create_embeddings(
        model,
        collection,
        chunks
    )

    search_test(
        model,
        collection
    )

    print("\n" + "=" * 50)
    print("Arduino RAG Embedding 완료")
    print("=" * 50)


if __name__ == "__main__":
    main()
