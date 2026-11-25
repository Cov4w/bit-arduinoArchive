
import json
import re
import hashlib
import yaml

from pathlib import Path
from collections import Counter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# 1. 설정
# ==========================================

INPUT_FILE = Path("data/raw/arduino_docs.jsonl")

OUTPUT_FILE = Path(
    "data/processed/arduino_chunks_v3.jsonl"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# 일반 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# 코드 분할기
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=100,
    separators=["\n\n", "\n", ";", ",", " ", ""]
)


# ==========================================
# 2. YAML Front Matter 추출
# ==========================================

def parse_frontmatter(content):

    pattern = r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)"

    match = re.match(
        pattern,
        content,
        re.DOTALL
    )

    if not match:
        return {}, content

    try:

        metadata = yaml.safe_load(
            match.group(1)
        ) or {}

        if not isinstance(metadata, dict):
            metadata = {}

    except yaml.YAMLError:

        metadata = {}

    return metadata, content[match.end():]


# ==========================================
# 3. 문서 제목 추출
# ==========================================

def get_title(metadata, content):

    title = metadata.get("title")

    if title:
        return str(title)

    match = re.search(
        r"^#\s+(.+)$",
        content,
        re.MULTILINE
    )

    return (
        match.group(1).strip()
        if match
        else "Untitled"
    )


# ==========================================
# 4. YAML 링크 목록 변환
# ==========================================

def extract_link_documents(yaml_data):

    links = yaml_data.get("links", [])

    if not isinstance(links, list):
        return []

    results = []

    current = []

    for item in links:

        if not isinstance(item, dict):
            continue

        link_title = item.get("title", "")
        url = item.get("url", "")

        if not link_title or not url:
            continue

        current.append(
            f"- {link_title}\n"
            f"  URL: {url}"
        )

        # 5개 단위로 그룹화
        if len(current) == 5:

            results.append(
                "\n".join(current)
            )

            current = []

    if current:

        results.append(
            "\n".join(current)
        )

    return results


# ==========================================
# 5. Markdown 구조 분석
# ==========================================

def parse_markdown_blocks(content):

    lines = content.splitlines()

    blocks = []

    current = []

    fence = None

    def flush():

        nonlocal current

        if current:

            value = "\n".join(
                current
            ).strip()

            if value:

                blocks.append((
                    "text",
                    value
                ))

            current = []

    i = 0

    while i < len(lines):

        line = lines[i]

        # 코드 블록 시작
        fence_match = re.match(
            r"^\s*(```+|~~~+)",
            line
        )

        if fence is None and fence_match:

            flush()

            fence = fence_match.group(1)

            current = [line]

            i += 1

            continue

        # 코드 블록 내부
        if fence is not None:

            current.append(line)

            if re.match(
                r"^\s*" + re.escape(fence) + r"\s*$",
                line
            ):

                blocks.append((
                    "code",
                    "\n".join(current)
                ))

                current = []

                fence = None

            i += 1

            continue

        # Markdown 제목
        heading = re.match(
            r"^(#{1,6})\s+(.+)$",
            line
        )

        if heading:

            flush()

            blocks.append((
                "heading",
                line.strip()
            ))

            i += 1

            continue

        # Markdown 표 시작
        if (
            "|" in line
            and i + 1 < len(lines)
            and re.search(
                r"\|[\s:|-]+\|",
                lines[i + 1]
            )
        ):

            flush()

            table_lines = [
                line,
                lines[i + 1]
            ]

            i += 2

            while i < len(lines):

                if "|" not in lines[i]:
                    break

                table_lines.append(
                    lines[i]
                )

                i += 1

            blocks.append((
                "table",
                "\n".join(table_lines)
            ))

            continue

        # 빈 줄
        if not line.strip():

            flush()

            i += 1

            continue

        current.append(line)

        i += 1

    flush()

    # 닫히지 않은 코드 블록 보존
    if fence is not None and current:

        blocks.append((
            "code",
            "\n".join(current)
        ))

    return blocks


# ==========================================
# 6. 텍스트 유효성 검사
# ==========================================

def is_meaningful_text(text):

    return bool(
        re.search(
            r"[A-Za-z0-9가-힣一-龥ぁ-んァ-ン]",
            text
        )
    )


# ==========================================
# 7. 코드 블록 분할
# ==========================================

def split_code(code):

    stripped = code.strip()

    match = re.match(
        r"^(```|~~~)([^\n]*)",
        stripped
    )

    if match:

        fence = match.group(1)

        language = match.group(2).strip()

        inner = stripped[match.end():]

        # 닫는 코드 펜스 제거
        inner = re.sub(
            r"\n\s*" + re.escape(fence) + r"\s*$",
            "",
            inner
        )

    else:

        fence = "```"

        language = ""

        inner = stripped

    pieces = code_splitter.split_text(
        inner
    )

    result = []

    for piece in pieces:

        piece = piece.strip()

        if not piece:
            continue

        if language:

            result.append(
                f"{fence}{language}\n"
                f"{piece}\n"
                f"{fence}"
            )

        else:

            result.append(
                f"{fence}\n"
                f"{piece}\n"
                f"{fence}"
            )

    return result


# ==========================================
# 8. Markdown 표 분할
# ==========================================

def split_table(table):

    lines = table.splitlines()

    if len(lines) < 3:
        return [table]

    header = lines[0]

    separator = lines[1]

    rows = lines[2:]

    chunks = []

    current_rows = []

    for row in rows:

        candidate = (
            header
            + "\n"
            + separator
            + "\n"
            + "\n".join(
                current_rows + [row]
            )
        )

        if (
            len(candidate) > CHUNK_SIZE
            and current_rows
        ):

            chunks.append(
                header
                + "\n"
                + separator
                + "\n"
                + "\n".join(current_rows)
            )

            current_rows = [row]

        else:

            current_rows.append(row)

    if current_rows:

        chunks.append(
            header
            + "\n"
            + separator
            + "\n"
            + "\n".join(current_rows)
        )

    # 단일 행 자체가 긴 경우
    result = []

    for chunk in chunks:

        if len(chunk) > CHUNK_SIZE:

            result.extend(
                text_splitter.split_text(chunk)
            )

        else:

            result.append(chunk)

    return result


# ==========================================
# 9. 메타데이터 생성
# ==========================================

def build_metadata(
    doc,
    yaml_data,
    title,
    section,
    content_type,
    index
):

    tags = yaml_data.get("tags", [])

    compatible_products = yaml_data.get(
        "compatible-products",
        []
    )

    return {

        "title": title,

        "category": doc["category"],

        "source": doc["source"],

        "description": yaml_data.get(
            "description",
            ""
        ),

        "tags": tags if isinstance(
            tags,
            list
        ) else [],

        "difficulty": yaml_data.get(
            "difficulty",
            ""
        ),

        "compatible_products": (
            compatible_products
            if isinstance(compatible_products, list)
            else []
        ),

        "section": section,

        "content_type": content_type,

        "chunk_index": index

    }


# ==========================================
# 10. 공통 Chunk 생성
# ==========================================

def save_chunk(
    chunks,
    doc,
    yaml_data,
    title,
    section,
    content_type,
    content
):

    content = content.strip()

    if not content:
        return

    # 일반 텍스트만 최소 길이 검사
    if content_type == "text":

        if not is_meaningful_text(content):
            return

        if len(content) < 20:
            return

    index = len(chunks)

    unique_id = hashlib.sha256(
        (
            doc["source"]
            + str(index)
            + content
        ).encode("utf-8")
    ).hexdigest()

    metadata = build_metadata(
        doc,
        yaml_data,
        title,
        section,
        content_type,
        index
    )

    chunks.append({

        "id": unique_id,

        "content": content,

        "metadata": metadata

    })


# ==========================================
# 11. 링크 인덱스 Chunk 생성
# ==========================================

def create_link_chunks(
    doc,
    yaml_data,
    title,
    link_groups
):

    chunks = []

    for group in link_groups:

        content = (
            f"Arduino {title}\n\n"
            f"관련 문서:\n"
            f"{group}"
        )

        save_chunk(
            chunks,
            doc,
            yaml_data,
            title,
            title,
            "link_index",
            content
        )

    return chunks


# ==========================================
# 12. 문서별 Chunk 생성
# ==========================================

def create_chunks(doc):

    yaml_data, body = parse_frontmatter(
        doc["content"]
    )

    title = get_title(
        yaml_data,
        body
    )

    # YAML에 links가 존재하는 경우
    link_groups = extract_link_documents(
        yaml_data
    )

    if link_groups:

        return create_link_chunks(
            doc,
            yaml_data,
            title,
            link_groups
        )

    # 일반 Markdown 문서 처리
    blocks = parse_markdown_blocks(body)

    chunks = []

    section_stack = []

    text_buffer = []

    # 일반 텍스트 버퍼 저장
    def flush_text():

        nonlocal text_buffer

        if not text_buffer:
            return

        text = "\n\n".join(
            text_buffer
        )

        section = (
            " > ".join(section_stack)
            or title
        )

        pieces = text_splitter.split_text(
            text
        )

        for piece in pieces:

            save_chunk(
                chunks,
                doc,
                yaml_data,
                title,
                section,
                "text",
                piece
            )

        text_buffer = []

    for block_type, content in blocks:

        # 제목 처리
        if block_type == "heading":

            flush_text()

            match = re.match(
                r"^(#{1,6})\s+(.+)$",
                content
            )

            if match:

                level = len(
                    match.group(1)
                )

                heading_text = match.group(
                    2
                ).strip()

                section_stack[:] = (
                    section_stack[:level - 1]
                )

                section_stack.append(
                    heading_text
                )

            continue

        # 일반 텍스트
        if block_type == "text":

            text_buffer.append(
                content
            )

            continue

        # 코드 및 표 앞에서 텍스트 저장
        flush_text()

        section = (
            " > ".join(section_stack)
            or title
        )

        # 코드 블록
        if block_type == "code":

            pieces = split_code(
                content
            )

            for piece in pieces:

                save_chunk(
                    chunks,
                    doc,
                    yaml_data,
                    title,
                    section,
                    "code",
                    piece
                )

        # 표
        elif block_type == "table":

            pieces = split_table(
                content
            )

            for piece in pieces:

                save_chunk(
                    chunks,
                    doc,
                    yaml_data,
                    title,
                    section,
                    "table",
                    piece
                )

    flush_text()

    return chunks


# ==========================================
# 13. 전체 문서 처리
# ==========================================

def preprocess():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    all_chunks = []

    category_stats = Counter()

    document_stats = Counter()

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            doc = json.loads(line)

            category = doc["category"]

            document_stats[category] += 1

            try:

                chunks = create_chunks(
                    doc
                )

                all_chunks.extend(
                    chunks
                )

                category_stats[category] += len(
                    chunks
                )

            except Exception as e:

                print(
                    f"[ERROR] {doc['source']}: {e}"
                )

    # 동일 출처 + 동일 내용만 중복 제거
    unique_chunks = []

    seen = set()

    for chunk in all_chunks:

        key = (
            chunk["metadata"]["source"],
            chunk["content"]
        )

        if key in seen:
            continue

        seen.add(key)

        unique_chunks.append(chunk)

    # JSONL 저장
    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in unique_chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    # ======================================
    # 결과 출력
    # ======================================

    print("=" * 45)
    print("Arduino RAG V3 전처리 완료")
    print("=" * 45)

    print(
        "원본 문서:",
        sum(document_stats.values())
    )

    print(
        "최종 Chunk:",
        len(unique_chunks)
    )

    print(
        "저장 경로:",
        OUTPUT_FILE
    )

    print("\n카테고리별 문서 수")

    for name, count in sorted(
        document_stats.items()
    ):

        print(
            f"{name:20}: {count}"
        )

    print("\n카테고리별 Chunk")

    for name in sorted(document_stats):

        print(
            f"{name:20}: {category_stats[name]}"
        )

    print("\n콘텐츠 유형별 Chunk")

    type_stats = Counter(
        c["metadata"]["content_type"]
        for c in unique_chunks
    )

    for name, count in type_stats.items():

        print(
            f"{name:20}: {count}"
        )


# ==========================================
# 14. 실행
# ==========================================

if __name__ == "__main__":

    preprocess()
