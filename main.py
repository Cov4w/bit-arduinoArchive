import os
import re
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from urllib.parse import urlparse
from rag_chat import ArduinoRAG

rag_service: ArduinoRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    # 서버 시작 시 BGE-M3 모델과 ChromaDB 즉시 로딩
    rag_service = ArduinoRAG()
    yield
    rag_service = None


app = FastAPI(lifespan=lifespan)


# 검색 결과 및 장바구니 항목 스키마
class ProductItem(BaseModel):
    title: str
    price: int = 0
    seller: str = ""
    link: str = ""


# 챗봇 요청 스키마 (질문 + 현재 장바구니 목록)
class ChatRequest(BaseModel):
    question: str
    cart_items: list[ProductItem] = []


@app.get("/")
def home():
    return {"message": "Arduino Archive"}


# 부품 검색 API (카카오 검색 API 연동)
@app.get("/search", response_model=list[ProductItem])
async def search_products(query: str = Query(..., min_length=1)):
    kakao_key = os.getenv("KAKAO_REST_API_KEY")
    if not kakao_key:
        raise HTTPException(
            status_code=500,
            detail=".env에 KAKAO_REST_API_KEY가 설정되어 있지 않습니다.",
        )

    url = "https://dapi.kakao.com/v2/search/web"
    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    # 쇼핑/가격 정보가 잘 나오도록 검색어 뒤에 '가격' 키워드를 보조로 붙임
    params = {"query": f"{query} 가격", "size": 15}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"카카오 검색 API 호출 실패: {resp.text}",
            )
        data = resp.json()

    results = []
    for doc in data.get("documents", []):
        # HTML 태그(<b> 등) 제거
        clean_title = re.sub(r"<[^>]+>", "", doc.get("title", "")).strip()
        clean_contents = re.sub(r"<[^>]+>", "", doc.get("contents", "")).strip()
        link = doc.get("url", "")

        # 1) 제목이나 본문에서 '12,000원' 또는 '12000원' 형태의 가격 추출
        combined_text = f"{clean_title} {clean_contents}"
        price_match = re.search(r"(\d{1,3}(?:,\d{3})+|\d{3,7})\s*원", combined_text)
        price = int(price_match.group(1).replace(",", "")) if price_match else 0

        # 2) URL 도메인에서 판매처(사이트명) 추출 (예: www.devicemart.co.kr -> devicemart.co.kr)
        seller = ""
        if link:
            domain = urlparse(link).netloc.replace("www.", "")
            seller = domain

        results.append(
            ProductItem(
                title=clean_title,
                price=price,
                seller=seller,
                link=link,
            )
        )

    # 가격 정보가 추출된 항목을 상단으로 정렬
    results.sort(key=lambda x: x.price > 0, reverse=True)
    return results


# 챗봇 답변 생성 API
@app.post("/chatbot")
def chatbot(request: ChatRequest):
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG 모델이 아직 로딩되지 않았습니다.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문 내용을 입력해주세요.")

    try:
        cart_dicts = [item.model_dump() for item in request.cart_items]
        return rag_service.ask(question, cart_items=cart_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")