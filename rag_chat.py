
import os
import torch

from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 루트 config.py에서 경로 가져오기
from config import MODEL_PATH, CHROMA_DB_PATH


class ArduinoRAG:
    def __init__(self):
        # 1. 환경 변수 설정
        load_dotenv()

        api_key = os.getenv("VIRTUAL_KEY")
        base_url = os.getenv("SCHOOL_LLM_BASE_URL")

        self.chat_model = "gemini-3.1-flash-lite"

        if not api_key or not base_url:
            raise ValueError(
                ".env의 VIRTUAL_KEY 또는 SCHOOL_LLM_BASE_URL을 확인하세요."
            )

        # 2. 모델 및 DB 경로 확인
        print("=" * 55)
        print("Arduino RAG 시스템 초기화")
        print("=" * 55)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"BGE-M3 모델 경로를 찾을 수 없습니다:\n{MODEL_PATH}"
            )

        if not CHROMA_DB_PATH.exists():
            raise FileNotFoundError(
                f"ChromaDB 경로를 찾을 수 없습니다:\n{CHROMA_DB_PATH}"
            )

        print(f"모델 경로: {MODEL_PATH}")
        print(f"DB 경로: {CHROMA_DB_PATH}")

        # 3. GPU / CPU 자동 선택
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"실행 장치: {self.device}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # 4. 로컬 BGE-M3 임베딩 모델 로딩
        print("\n[1] BGE-M3 로컬 모델 로딩 중...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=str(MODEL_PATH),
            model_kwargs={
                "device": self.device,
                "local_files_only": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        print("[완료] BGE-M3 모델 로딩 완료")

        # 5. 기존 ChromaDB 연결
        print("\n[2] ChromaDB 연결 중...")

        self.vectorstore = Chroma(
            collection_name="arduino_docs_v3",
            persist_directory=str(CHROMA_DB_PATH),
            embedding_function=self.embeddings,
        )

        document_count = self.vectorstore._collection.count()

        if document_count == 0:
            raise RuntimeError(
                "ChromaDB에 데이터가 없습니다. "
                "경로와 Collection을 확인하세요."
            )

        print(f"[완료] ChromaDB 연결 완료: {document_count:,}개 Chunk")

        # 6. Retriever 설정
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # 7. 학교 API Gateway(OpenAI SDK) 연결
        print("\n[3] 학교 API Gateway 연결 설정 중...")

        self.client = OpenAI(
            api_key=api_key,
            base_url=f"{base_url.rstrip('/')}/api/v1",
        )

        print("[완료] API Gateway 연결 설정 완료")

        # 8. 프롬프트 템플릿 정의
        self.prompt = ChatPromptTemplate.from_template(
            """
너는 Arduino 전문 기술 지원 챗봇이다.
사용자의 질문과 [사용자 장바구니 부품 목록]을 확인하고,
아래 [검색 문서]를 근거로 답변하라.

[답변 규칙]
1. 검색 문서에 포함된 정보를 우선적으로 사용한다.
2. 검색 문서에 없는 내용을 사실처럼 만들어내지 않는다.
3. 근거가 부족하면 정보가 부족하다고 명시한다.
4. 부품 간 호환성 질문 시 동작 전압(3.3V vs 5V),
   통신 방식(I2C/SPI/UART), 핀 구성, 전류 요구량 등을 기준으로 비교한다.
5. 사용자가 질문한 정확한 모델이 검색 문서에 없고
   유사한 파생 모델만 있다면, 두 모델을 동일시하지 말고 명확히 구분한다.
6. 코드가 필요한 경우 실행 가능한 Arduino 코드를 제공하고
   주요 부분에 주석을 작성한다.
7. 답변은 한국어로 작성한다.

[사용자 장바구니 부품 목록]
{cart_context}

[검색 문서]
{context}

[사용자 질문]
{question}

위 내용을 바탕으로 정확하고 이해하기 쉽게 답변하라.
"""
        )

        print("\n" + "=" * 55)
        print("[완료] Arduino RAG 시스템 준비 완료!")
        print("=" * 55)

    def _generate_search_queries(
        self,
        question: str,
        cart_context: str = ""
    ) -> list[str]:
        """
        질문과 장바구니 부품명을 바탕으로
        영문 공식 문서 검색용 쿼리 1~3개 생성
        """

        query_prompt = f"""
You are a search query generator for official Arduino technical documentation.
Convert the user's question into 1 to 3 concise English search queries
optimized for vector database retrieval.

Rules:
- If the user asks about items in their shopping cart
  (e.g., compatibility, specs, wiring), extract the actual
  Arduino board/sensor names from [Cart Items] and generate
  English technical queries for them.
- If the question compares multiple boards or topics,
  split them into separate queries.
- Use exact official Arduino product names and technical terms
  (e.g., "Arduino UNO R3 operating voltage pinout",
  "Nano 33 BLE 3.3V compatibility").
- Output ONLY the search queries, one per line.
  No numbering, no bullets, no extra explanation.

[Cart Items]
{cart_context}

[User Question]
{question}
"""

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "user", "content": query_prompt}
            ],
            temperature=0.0,
        )

        raw_text = response.choices[0].message.content or ""

        queries = [
            line.strip().lstrip("-•1234567890. ")
            for line in raw_text.splitlines()
            if line.strip()
        ]

        return queries[:3] if queries else [question]

    def _multi_query_retrieve(
        self,
        queries: list[str],
        max_docs: int = 8
    ) -> list:
        """
        분해된 각 영어 쿼리로 문서를 검색하고
        본문 중복을 제거하여 병합
        """

        seen_contents = set()
        unique_docs = []

        for query in queries:
            docs = self.retriever.invoke(query)

            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)

                    if len(unique_docs) >= max_docs:
                        return unique_docs

        return unique_docs

    def _format_docs(self, docs) -> str:
        """
        검색된 문서 리스트를 프롬프트용 문자열로 변환
        """

        formatted = []

        for index, doc in enumerate(docs, 1):
            meta = doc.metadata

            content = (
                f"[문서 {index}]\n"
                f"제목: {meta.get('title', '제목 없음')}\n"
                f"카테고리: {meta.get('category', 'unknown')}\n"
                f"유형: {meta.get('content_type', 'unknown')}\n"
                f"URL: {meta.get('url', '')}\n\n"
                f"내용:\n{doc.page_content}"
            )

            formatted.append(content)

        return "\n\n".join(formatted)

    def ask(
        self,
        question: str,
        cart_items: list[dict] | None = None
    ) -> dict:
        """
        질문 + 장바구니 목록
        -> 영문 쿼리 생성
        -> 다중 검색
        -> LLM 답변 생성
        """

        # 1. 장바구니 컨텍스트 생성
        if cart_items:
            cart_context = "\n".join(
                f"- {item.get('title', '')} "
                f"(판매처: {item.get('seller', '')}, "
                f"가격: {item.get('price', 0):,}원)"
                for item in cart_items
            )
        else:
            cart_context = "장바구니가 비어 있음"

        # 2. 영문 검색 쿼리 생성
        search_queries = self._generate_search_queries(
            question,
            cart_context
        )

        print(f"\n[생성된 검색 쿼리] {search_queries}")

        # 3. 다중 쿼리 검색
        docs = self._multi_query_retrieve(
            search_queries,
            max_docs=8
        )

        if not docs:
            return {
                "answer": "관련 문서를 찾지 못했습니다.",
                "queries": search_queries,
                "sources": [],
            }

        print(f"[검색 완료] {len(docs)}개 문서 검색")

        # 4. 검색 문서 컨텍스트 생성
        context = self._format_docs(docs)

        formatted_prompt = self.prompt.format(
            cart_context=cart_context,
            context=context,
            question=question,
        )

        # 5. LLM 답변 생성
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=0.2,
        )

        # 6. 참고 문서 출처 정리
        sources = [
            {
                "title": doc.metadata.get("title", "제목 없음"),
                "category": doc.metadata.get("category", "unknown"),
                "url": doc.metadata.get("url", ""),
            }
            for doc in docs
        ]

        return {
            "answer": response.choices[0].message.content,
            "queries": search_queries,
            "sources": sources,
        }
