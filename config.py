
from pathlib import Path

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent

# BGE-M3 로컬 모델
MODEL_PATH = BASE_DIR / "Data" / "models" / "bge-m3"

# ChromaDB 저장 경로
CHROMA_DB_PATH = (
    BASE_DIR / "Data" / "data" / "processed" / "chroma_db"
)
