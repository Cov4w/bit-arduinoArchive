
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

# 현재 search_test.py가 있는 Data 폴더
BASE_DIR = Path(__file__).resolve().parent

# 프로젝트 내부에 저장된 모델 경로
MODEL_PATH = BASE_DIR / "models" / "bge-m3"

device = "cuda" if torch.cuda.is_available() else "cpu"

# 로컬 모델만 사용 (인터넷 다운로드 방지)
model = SentenceTransformer(
    str(MODEL_PATH),
    device=device,
    local_files_only=True
)

print(f"모델 경로: {MODEL_PATH}")
print(f"실행 장치: {device}")
