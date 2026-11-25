# Arduino RAG Chatbot

Arduino 공식 기술 문서를 기반으로 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇 프로젝트입니다.

## 주요 기능

- **문서 검색:** Arduino 공식 문서 기반 벡터 검색
- **다중 쿼리 검색:** 질문을 영문 기술 검색어로 변환하여 검색 정확도 개선
- **AI 답변 생성:** 검색된 문서를 바탕으로 한국어 기술 답변 제공
- **부품 호환성 검토:** 장바구니 부품의 전압, 통신 방식, 핀 구성 등을 고려한 정보 제공
- **출처 제공:** 답변에 참고한 문서의 제목과 URL 반환

## Tech Stack

- Python 3.13
- LangChain
- ChromaDB
- BGE-M3 (Embedding Model)
- OpenAI SDK (학교 API Gateway)
- PyTorch

## Project Structure

```text
arduinoArchive/
├── Data/
│   ├── models/
│   │   └── bge-m3/             # 로컬 임베딩 모델
│   └── data/
│       └── processed/
│           └── chroma_db/      # ChromaDB 벡터 데이터베이스
├── config.py                   # 모델 및 DB 경로 설정
├── .env.example                # 환경 변수 예시
├── requirements.txt            # Python 패키지 목록
└── README.md
```

## Installation

### 1. 가상환경 생성

```bash
conda create -n arduinoArchive python=3.13 -y
conda activate arduinoArchive
```

### 2. PyTorch 설치

CUDA 12.8 환경:

```bash
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

CPU 환경:

```bash
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. 라이브러리 설치

```bash
python -m pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env.example`을 참고하여 `.env` 파일을 생성합니다.

```env
VIRTUAL_KEY=YOUR_API_KEY
SCHOOL_LLM_BASE_URL=YOUR_API_BASE_URL
```

### 5. 모델 및 데이터베이스 준비

다음 경로에 BGE-M3 모델과 사전 구축된 ChromaDB를 배치합니다.

- `Data/models/bge-m3/`
- `Data/data/processed/chroma_db/`

모델과 벡터 데이터베이스는 별도로 제공되며, 실행 시 기존 데이터를 재사용합니다.

## Notes

- 임베딩 모델은 로컬 경로에서 로딩합니다.
- CUDA 사용이 불가능한 환경에서는 CPU로 실행할 수 있습니다.
- 학교 API Gateway 사용을 위해 유효한 API Key와 Base URL이 필요합니다.
- `.env`, 모델 파일, ChromaDB 데이터는 GitHub에 업로드하지 않습니다.
