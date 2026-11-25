
import subprocess
import json
import re
from pathlib import Path

# 프로젝트 설정
REPO_DIR = Path("./docs-content")
OUTPUT_DIR = Path("./data/raw")
OUTPUT_FILE = OUTPUT_DIR / "arduino_docs.jsonl"

# 초기 수집 대상
TARGET_CATEGORIES = {
    "hardware",
    "built-in-examples",
    "programming",
    "tutorials",
    "software",
    "learn"
}

def get_title(content):
    """Markdown 첫 번째 H1 제목 추출"""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else "Untitled"

def collect_documents():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Git이 추적하는 파일 목록
    result = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=REPO_DIR
    )

    files = result.decode("utf-8").split("\0")

    documents = []
    skipped = 0

    for file in files:
        if not file:
            continue

        path = Path(file)

        # Markdown 파일만 대상
        if path.suffix.lower() != ".md":
            continue

        # 선택한 카테고리만 수집
        if len(path.parts) < 3:
            continue

        category = path.parts[1]

        if category not in TARGET_CATEGORIES:
            continue

        full_path = REPO_DIR / path

        if not full_path.is_file():
            skipped += 1
            continue

        try:
            content = full_path.read_text(encoding="utf-8")

            if not content.strip():
                skipped += 1
                continue

            document = {
                "id": file.replace("\\", "/"),
                "title": get_title(content),
                "category": category,
                "source": file.replace("\\", "/"),
                "content": content
            }

            documents.append(document)

        except Exception as e:
            print(f"[ERROR] {file}: {e}")
            skipped += 1

    # JSONL 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(
                json.dumps(doc, ensure_ascii=False) + "\n"
            )

    print("=" * 45)
    print("Arduino 문서 수집 완료")
    print("=" * 45)
    print(f"수집 문서: {len(documents)}")
    print(f"건너뛴 문서: {skipped}")
    print(f"저장 경로: {OUTPUT_FILE}")

    # 카테고리별 통계
    for category in sorted(TARGET_CATEGORIES):
        count = sum(
            1 for doc in documents
            if doc["category"] == category
        )
        print(f"{category:20} : {count}")

if __name__ == "__main__":
    collect_documents()
