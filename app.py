
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS 디자인
# ==========================================
# 2.1 기본 디자인 (예은)
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background-color: #edf6f7;
}

/* 기본 여백 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* 상단 메뉴 숨기기 */
#MainMenu, footer, header {
    visibility: hidden;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background-color: #78c5cd;
    min-width: 230px;
    max-width: 230px;
}

section[data-testid="stSidebar"] > div {
    padding: 25px 15px;
}

/* 사이드바 텍스트 */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: white;
}

/* 로고 */
.logo-area {
    background-color: rgba(255,255,255,0.95);
    border-radius: 18px;
    padding: 17px 10px;
    margin-bottom: 30px;
    text-align: center;
}

/* 사이드바 메뉴 */
.menu-item {
    color: white;
    padding: 13px 15px;
    margin: 8px 0;
    border-radius: 13px;
    font-size: 15px;
    font-weight: 600;
}

.menu-active {
    background-color: rgba(255,255,255,0.27);
}

/* 메인 인사말 */
.greeting {
    font-size: 29px;
    font-weight: 800;
    color: #19334d;
    margin-bottom: 5px;
}

.sub-greeting {
    color: #91a2ae;
    font-size: 13px;
    margin-bottom: 20px;
}

/* 상단 검색창 */
div[data-testid="stTextInput"] input {
    border: none;
    border-radius: 15px;
    background-color: white;
    min-height: 45px;
    box-shadow: 0 3px 15px rgba(30,60,80,0.07);
}

/* 카드 공통 */
.dashboard-card {
    border-radius: 23px;
    padding: 24px;
    min-height: 155px;
    box-shadow: 0 5px 20px rgba(50,80,100,0.07);
    margin-bottom: 18px;
}

.welcome-card {
    background: #43aebe;
    color: white;
}

.result-card {
    background: #83b3f5;
    color: white;
}

.dashboard-card h3 {
    font-size: 20px;
    margin: 0 0 12px 0;
}

.dashboard-card p {
    font-size: 13px;
    line-height: 1.7;
}

/* 기능 카드 */
.feature-card {
    background-color: white;
    border-radius: 20px;
    padding: 17px 12px;
    box-shadow: 0 5px 18px rgba(50,80,100,0.06);
    min-height: 105px;
    text-align: center;
    color: #263c54;
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 8px;
}

.feature-title {
    font-size: 15px;
    font-weight: 800;
}

.feature-desc {
    font-size: 11px;
    color: #a0abb6;
    margin-top: 5px;
}

/* 섹션 제목 */
.section-title {
    font-size: 19px;
    font-weight: 800;
    color: #19334d;
    margin: 15px 0;
}

/* 채팅 영역 */
.st-key-chat_panel {
    background-color: white;
    border-radius: 25px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(50,80,100,0.06);
    min-height: 450px;
}

/* 채팅 메시지 */
div[data-testid="stChatMessage"] {
    border-radius: 15px;
    padding: 13px;
    margin-bottom: 12px;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-user"]
) {
    background-color: #e5f5f5;
}

div[data-testid="stChatMessage"]:has(
    div[data-testid="chatAvatarIcon-assistant"]
) {
    background-color: #f2f6fb;
}

/* 오른쪽 프로필 */
.profile-card {
    background-color: white;
    border-radius: 24px;
    padding: 25px 15px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(50,80,100,0.06);
    margin-bottom: 20px;
}

.profile-icon {
    font-size: 48px;
    background-color: #e9f1fa;
    width: 80px;
    height: 80px;
    line-height: 80px;
    border-radius: 50%;
    margin: auto;
}

.profile-name {
    color: #19334d;
    font-size: 18px;
    font-weight: 800;
    margin-top: 12px;
}

.profile-desc {
    font-size: 12px;
    color: #a1adb8;
}

/* 오른쪽 정보 카드 */
.info-card {
    background-color: white;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(50,80,100,0.05);
}

.info-title {
    color: #19334d;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 12px;
}

/* 버튼 */
.stButton > button {
    border-radius: 12px;
    border: 1px solid #e5edf1;
    background-color: white;
    color: #29445b;
    min-height: 42px;
    font-size: 13px;
    transition: 0.2s;
}

.stButton > button:hover {
    border-color: #55b9c1;
    color: #269eaa;
    background-color: #f0fbfb;
}

/* 채팅 입력창 */
div[data-testid="stForm"] {
    background-color: #f3f6fa;
    border-radius: 15px;
    padding: 10px;
    border: none;
}

/* 파일 업로드 */
div[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 13px;
    padding: 10px;
}

hr {
    border-color: #e9eff2;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 2.2 추가 디자인 (솔: 온보딩·캘린더·사이드바 메뉴·온톨로지·상단 단계 전환)
# ==========================================
st.markdown("""
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent;
    padding: 10px 14px;
    border-radius: 13px;
    margin: 3px 0;
    width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background-color: rgba(255,255,255,0.27);
}
section[data-testid="stSidebar"] div[role="radiogroup"] p {
    color: white;
    font-weight: 600;
    font-size: 15px;
}
.onboard-title { font-size: 34px; font-weight: 800; color: #19334d; text-align: center; margin-top: 30px; }
.onboard-sub { font-size: 15px; color: #7b8d99; text-align: center; margin-bottom: 30px; }
.stage-card {
    background: white; border-radius: 23px; padding: 28px 20px; text-align: center;
    box-shadow: 0 5px 20px rgba(50,80,100,0.07); min-height: 210px;
}
.stage-emoji { font-size: 44px; }
.stage-name { font-size: 20px; font-weight: 800; color: #19334d; margin-top: 8px; }
.stage-desc { font-size: 13px; color: #8a9aa6; line-height: 1.7; margin-top: 8px; }
.cal-wrap { background: white; border-radius: 25px; padding: 20px 22px; box-shadow: 0 5px 20px rgba(50,80,100,0.06); }
.cal-title { font-size: 20px; font-weight: 800; color: #19334d; text-align: center; margin-bottom: 10px; }
table.cal { width: 100%; border-collapse: collapse; text-align: center; }
table.cal th { color: #a6b3bd; font-weight: 500; font-size: 13px; padding: 6px 0; }
table.cal td { font-size: 14px; color: #19334d; padding: 8px 0; height: 44px; vertical-align: top; }
table.cal td.other { color: #cfd8de; }
table.cal td.today span.d { background: #b9e2e6; border-radius: 10px; padding: 4px 8px; }
table.cal td.weekend { background: #f6f9fb; }
.dot { display: block; width: 6px; height: 6px; border-radius: 50%; margin: 3px auto 0; }
.dot-교육 { background: #43aebe; } .dot-취업 { background: #f5b83d; } .dot-포폴 { background: #83b3f5; }
.event-row { background: white; border-radius: 15px; padding: 12px 15px; margin-bottom: 9px;
    box-shadow: 0 4px 15px rgba(50,80,100,0.05); }
.event-title { font-size: 14px; font-weight: 800; color: #19334d; }
.event-date { font-size: 12px; color: #8e9da8; }
.dday { float: right; font-size: 12px; font-weight: 800; color: #269eaa; }
.tag { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 8px; margin-right: 6px; color: white; }
.tag-교육 { background: #43aebe; } .tag-취업 { background: #f5b83d; } .tag-포폴 { background: #83b3f5; }
.page-title { font-size: 26px; font-weight: 800; color: #19334d; margin-bottom: 4px; }
.page-sub { font-size: 13px; color: #91a2ae; margin-bottom: 18px; }
.white-card { background: white; border-radius: 20px; padding: 20px; box-shadow: 0 5px 18px rgba(50,80,100,0.06); margin-bottom: 15px; }
div[data-testid="stButtonGroup"] { justify-content: flex-end; }
.src-box { font-size: 12px; color: #6b7d89; background: #f6f9fb; border-radius: 10px; padding: 8px 10px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 기본 데이터
# ==========================================
import os
import re
import json
import math
import uuid
import calendar
import tempfile
from datetime import date, datetime, timedelta
from collections import Counter

import pandas as pd
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3.1 Streamlit Cloud의 Secrets(OPENAI_API_KEY)를 환경변수로 등록
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# 3.2 프로필 단계별 설정 (예비 에이블러 / 에이블러 / 수료자)
STAGES = {
    "예비": {
        "label": "예비 에이블러", "emoji": "🌱",
        "desc": "지원·합격 발표·입교 준비 단계<br>지원 자격, 트랙 선택, 입교 전 준비",
        "focus": ["지원", "자격", "트랙", "역량", "입교", "파이썬"],
        "rule": "사용자는 아직 입교 전인 예비 지원자임. 일정은 정확한 날짜 대신 '교육 n주차' 또는 '○월 초·중·말 예정'처럼 대략적으로 안내함. 지원을 망설이지 않도록 도전을 응원하는 말로 마무리함.",
        "questions": ["비전공자도 따라갈 수 있나요?", "AI 개발자 트랙과 DX 트랙은 어떻게 달라요?",
                      "지원 자격과 선발 절차를 알려주세요.", "입교 전에 무엇을 준비하면 좋을까요?"],
    },
    "재학": {
        "label": "에이블러", "emoji": "🚀",
        "desc": "교육 과정 진행 중<br>출결·장려금, 미니프로젝트, 취업 일정",
        "focus": ["미니프로젝트", "출결", "장려금", "컨설팅", "포트폴리오", "AICE"],
        "rule": "사용자는 현재 교육 중인 에이블러임. 일정은 월/일 날짜로 정확히 안내하고, 신청 마감이 가까운 순서로 정리함.",
        "questions": ["훈련장려금은 얼마이고 어떤 조건에서 받나요?", "지각, 조퇴 기준과 수료 기준을 알려주세요.",
                      "포트폴리오·자소서·면접 컨설팅은 언제 신청하나요?", "DX 트랙 포트폴리오에는 어떤 프로젝트를 넣나요?"],
    },
    "수료": {
        "label": "수료자", "emoji": "🎓",
        "desc": "수료 후 1년간 맞춤 정보 제공<br>취업 준비, 잡페어, 면접, 선배 사례",
        "focus": ["취업", "잡페어", "면접", "자기소개서", "포트폴리오", "사례"],
        "rule": "사용자는 과정을 마친 수료생임. 일정은 월/일 날짜로 안내하고, 취업 준비와 선배 사례 중심으로 답함.",
        "questions": ["잡페어와 모의면접은 언제인가요?", "면접은 어떻게 준비하면 좋을까요?",
                      "선배들은 어떤 직무로 취업했나요?", "포트폴리오를 어떤 순서로 정리하면 좋을까요?"],
    },
}

# 3.3 10기 일정
EDU_START = date(2026, 9, 29)   # 10기 교육 시작일
EDU_END = date(2027, 3, 5)      # 10기 수료일

# (시작일, 종료일, 제목, 분류) - AIVLE School 백서 10기 기준
EVENTS = [
    (date(2026, 9, 29), date(2026, 9, 29), "교육 시작 (입교)", "교육"),
    (date(2026, 11, 16), date(2026, 11, 17), "미니프로젝트 4차 (RAG 학습 도우미)", "교육"),
    (date(2026, 11, 16), date(2026, 11, 18), "취업지원 플랫폼 회원가입", "취업"),
    (date(2026, 11, 30), date(2026, 12, 2), "포트폴리오 컨설팅 신청", "포폴"),
    (date(2026, 12, 28), date(2027, 1, 3), "자기소개서 컨설팅 신청", "취업"),
    (date(2027, 1, 20), date(2027, 1, 22), "면접 컨설팅 신청", "취업"),
    (date(2027, 2, 24), date(2027, 2, 24), "빅프로젝트 발표회", "교육"),
    (date(2027, 2, 26), date(2027, 2, 26), "AIVLE 잡페어", "취업"),
    (date(2027, 3, 5), date(2027, 3, 5), "수료", "교육"),
]
WEEKDAY = ["월", "화", "수", "목", "금", "토", "일"]

# 3.4 날짜 표시
def fmt_event_date(start, end, stage):
    """재학·수료는 월/일 날짜, 예비는 n주차·○월 초/중/말 예정으로 표시"""
    if stage == "예비":
        week = (start - EDU_START).days // 7 + 1
        part = "초" if start.day <= 10 else ("중" if start.day <= 20 else "말")
        w = f"교육 {week}주차 · " if week >= 1 else ""
        return f"{w}{start.month}월 {part} 예정"
    s = f"{start.month}/{start.day:02d}({WEEKDAY[start.weekday()]})"
    if end != start:
        s += f" ~ {end.month}/{end.day:02d}({WEEKDAY[end.weekday()]})"
    return s


def dday_text(start, end, today):
    if start <= today <= end:
        return "진행 중"
    d = (start - today).days
    return f"D-{d}" if d > 0 else "종료"

# ==========================================
# 4. RAG 설정
# ==========================================
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"
ONTOLOGY_FILE = "aivle_ontology.json"
LOG_FILE = "chat_log.json"
BOT_IMG = "assets/bot.png"   # 챗봇 캐릭터 이미지 (없으면 🤖 이모지)


def bot_avatar():
    return BOT_IMG if os.path.exists(BOT_IMG) else "🤖"


def bot_img_html(size=44):
    if not os.path.exists(BOT_IMG):
        return "🤖"
    import base64
    with open(BOT_IMG, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{data}" width="{size}" style="vertical-align:middle;">'


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4.1 01번 노트북 rag_pipeline() 그대로 사용 (업로드 문서용)
# 백서 컬렉션(aivle_docs)을 지우지 않도록 collection_name 인자만 추가함 (기본값은 원본과 동일)
def rag_pipeline(filepath, collection_name="aivle_docs"):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""])
    chunks = splitter.split_documents(docs)

    # Embedding 모델
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # 벡터저장소 만들기
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory="chroma_db"
    )

    # 기존 데이터 비우기
    vectorstore.reset_collection()

    # 문서 추가
    _ = vectorstore.add_documents(chunks)

    # 검색기 구성
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # LLM 선언
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 프롬프트 구성
    s_msg = """
당신은 KT AIVLE School 교육생과 예비 지원자를 돕는 'AIVLE School 학습 도우미'입니다.
1. 반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 확인되지 않습니다."라고 답한 뒤, 확인할 수 있는 곳(공식 홈페이지, AIVLE-EDU 공지, 담당 매니저 등)을 안내하세요.
3. 문서에 [추정], [후기], [언론] 표기가 있는 내용은 그 표기를 함께 알려 근거 수준을 밝히세요.
4. 일정, 자격, 지원금처럼 바뀔 수 있는 정보는 운영 공지 확인을 함께 안내하세요.
5. 답변은 간결하고 명확하게 목록 형태로 정리하고, 마지막에 바로 실천할 수 있는 한 줄 응원이나 도전 제안을 덧붙이세요.
[문서]
{context}"""
    h_msg = "{input}"
    messages = [("system", s_msg), ("human", h_msg)]
    prompt = ChatPromptTemplate.from_messages(messages)

    # RAG 체인 구성
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# 4.2 백서 검색기: 벡터 + BM25(한글 2-gram) Hybrid, RRF로 합침
def kor_tokenize(text):
    tokens = re.findall(r"[A-Za-z0-9]+|[가-힣]+", text.lower())
    result = []
    for w in tokens:
        result.append(w)
        if re.match(r"[가-힣]", w) and len(w) >= 2:
            result += [w[i:i+2] for i in range(len(w) - 1)]
    return result


class SimpleBM25:
    """rank_bm25와 같은 BM25Okapi 계산 (추가 패키지 없이 구현)"""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.toks = [kor_tokenize(d.page_content) for d in docs]
        self.avgdl = sum(len(t) for t in self.toks) / max(len(self.toks), 1)
        self.tf = [Counter(t) for t in self.toks]
        df = Counter()
        for t in self.toks:
            df.update(set(t))
        n = len(self.toks)
        self.idf = {w: math.log((n - c + 0.5) / (c + 0.5) + 1) for w, c in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k=8):
        q = kor_tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i] for i in order[:k] if scores[i] > 0]


def rrf_merge(result_lists, weights, c=60, top_n=4):
    score, keep = {}, {}
    for docs, w in zip(result_lists, weights):
        for rank, d in enumerate(docs):
            key = d.page_content
            score[key] = score.get(key, 0) + w / (c + rank + 1)
            keep[key] = d
    best = sorted(score, key=score.get, reverse=True)[:top_n]
    return [keep[k] for k in best]


@st.cache_resource
def get_vectorstore():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(collection_name=COLLECTION, embedding_function=embedding,
                  persist_directory=CHROMA_DIR)


@st.cache_resource
def get_bm25():
    data = get_vectorstore()._collection.get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(data["documents"], data["metadatas"])]
    return SimpleBM25(docs) if docs else None


def hybrid_retrieve(query, k=4):
    vec = get_vectorstore().similarity_search(query, k=8)
    bm25 = get_bm25()
    kw = bm25.search(query, k=8) if bm25 else []
    return rrf_merge([kw, vec], [0.5, 0.5], top_n=k)


@st.cache_resource
def get_llm(streaming=True):
    return ChatOpenAI(model="gpt-4o-mini", streaming=streaming, temperature=0)

# 4.3 쿼리 리라이팅 (혜린님 규칙)
REWRITE_SYS = """당신은 KT AIVLE School 문서 검색을 위한 질의 재작성 도우미입니다.
대화 이력과 현재 질문을 참고하여, 이전 대화 없이도 의미를 이해할 수 있는 독립적인 검색 질문 한 개를 작성하세요.
규칙:
1. "그럼", "그 회사", "그 조건"처럼 생략된 대상을 대화에서 명확히 확인할 수 있을 때만 구체적으로 바꾸세요.
2. 현재 질문의 의미와 조건을 유지하세요.
3. 기수, 트랙, 기업명, 인물명, 숫자를 임의로 바꾸거나 추가하지 마세요.
4. 새로운 주제의 질문이면 이전 주제와 억지로 연결하지 마세요.
5. 이미 독립적으로 이해할 수 있는 질문이면 그대로 반환하세요.
6. 대화에 나온 이전 답변을 검증된 사실로 취급하지 마세요.
7. 답을 작성하지 말고 검색 질문만 한 줄로 출력하세요."""


def rewrite_question(question, history):
    if not history:
        return question
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYS), MessagesPlaceholder("history"), ("human", "{input}")])
    chain = prompt | get_llm(False) | StrOutputParser()
    out = chain.invoke({"input": question, "history": history}).strip()
    return out or question

# 4.4 답변 프롬프트: rag_pipeline 규칙 + 프로필 + 현재 화면
ANSWER_SYS = """
당신은 KT AIVLE School 교육생과 예비 지원자를 돕는 'AIVLE School 학습 도우미'입니다.
1. 반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 확인되지 않습니다."라고 답한 뒤, 확인할 수 있는 곳(공식 홈페이지, AIVLE-EDU 공지, 담당 매니저 등)을 안내하세요.
3. 문서에 [추정], [후기], [언론] 표기가 있는 내용은 그 표기를 함께 알려 근거 수준을 밝히세요.
4. 일정, 자격, 지원금처럼 바뀔 수 있는 정보는 운영 공지 확인을 함께 안내하세요.
5. 답변은 간결하고 명확하게 목록 형태로 정리하고, 마지막에 바로 실천할 수 있는 한 줄 응원이나 도전 제안을 덧붙이세요.

[사용자 프로필]
{profile}

[사용자가 보고 있는 화면]
{screen}

[문서]
{context}"""


def answer_stream(question, history, profile_text, screen_text):
    """백서 기준 답변: 리라이팅 → Hybrid 검색 → 프로필·화면 반영 답변 (스트리밍)"""
    search_q = rewrite_question(question, history)
    docs = hybrid_retrieve(search_q, k=4)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYS), MessagesPlaceholder("history"), ("human", "{input}")])
    chain = prompt | get_llm(True) | StrOutputParser()
    stream = chain.stream({"input": question, "history": history, "profile": profile_text,
                           "screen": screen_text or "없음", "context": format_docs(docs)})
    return stream, docs, search_q

# ==========================================
# 5. 대화 저장소 (다시 보기)
# ==========================================
def load_archive():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_archive():
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.archive, f, ensure_ascii=False, indent=1)
    except Exception:
        pass

# ==========================================
# 6. 온톨로지 불러오기
# ==========================================
@st.cache_data
def load_ontology_text(raw):
    data = json.loads(raw)
    nodes = {n["id"]: n.get("type", "기타") for n in data.get("nodes", [])}
    links = data.get("links") or data.get("edges") or []
    edges = []
    for l in links:
        edges.append({"subject": l["source"], "relation": l.get("relation", "관련됨"),
                      "object": l["target"], "evidence": l.get("evidence", ""),
                      "page": l.get("page", ""),
                      "s_type": nodes.get(l["source"], "기타"), "o_type": nodes.get(l["target"], "기타")})
    return edges


TAB_RULES = {
    "교육": ({"과목", "미니프로젝트", "트랙", "규정"}, ["교육", "과목", "미니프로젝트", "출결", "수료", "장려금", "파이썬", "머신러닝", "딥러닝"]),
    "취업": ({"취업프로그램", "플랫폼", "자격"}, ["취업", "컨설팅", "면접", "잡페어", "인적성", "자기소개서", "플랫폼", "AICE"]),
    "포폴": (set(), ["포트폴리오", "프로젝트", "깃허브", "GitHub", "노션", "발표"]),
}


def edge_in_tab(e, tab):
    types, kws = TAB_RULES[tab]
    if e["s_type"] in types or e["o_type"] in types:
        return True
    text = f'{e["subject"]} {e["object"]} {e["evidence"]}'
    return any(k in text for k in kws)

# ==========================================
# 7. 세션 상태
# ==========================================
defaults = {"messages": [], "pending_prompt": None, "profile": None, "page": "🏠 홈",
            "upload_chain": None, "upload_name": None, "answer_source": "AIVLE School 백서",
            "pinned": [], "checks": {}, "send_screen": True, "last_sources": []}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "archive" not in st.session_state:
    st.session_state.archive = load_archive()


def go(page):
    st.session_state.page = page


def ask(q):
    st.session_state.pending_prompt = q


def on_search():
    if st.session_state.search_input.strip():
        st.session_state.pending_prompt = st.session_state.search_input.strip()
    st.session_state.search_input = ""

# ==========================================
# 8. 처음 화면: 프로필 맞춤형 설정 (솔)
# ==========================================
if st.session_state.profile is None:
    st.markdown('<div class="onboard-title">AIVLE School 학습 도우미에 오신 걸 환영해요 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="onboard-sub">지금 나의 단계를 고르면, 일정·추천 질문·답변이 모두 맞춤형으로 바뀌어요.</div>', unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        name = st.text_input("이름(닉네임)", placeholder="예: 솔", key="onboard_name")
        track = st.radio("트랙", ["DX 컨설턴트", "AI 개발자"], horizontal=True, key="onboard_track")
        cols = st.columns(3, gap="medium")
        for col, key in zip(cols, ["예비", "재학", "수료"]):
            info = STAGES[key]
            with col:
                st.markdown(f"""
                <div class="stage-card">
                    <div class="stage-emoji">{info['emoji']}</div>
                    <div class="stage-name">{info['label']}</div>
                    <div class="stage-desc">{info['desc']}</div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"{info['label']}로 시작", key=f"stage_{key}", use_container_width=True):
                    st.session_state.profile = {"stage": key, "name": name.strip() or "에이블러", "track": track}
                    st.rerun()
    st.stop()

profile = st.session_state.profile
stage = profile["stage"]
S = STAGES[stage]
today = date.today()
profile_text = (f"- 이름: {profile['name']}\n- 단계: {S['label']}\n- 트랙: {profile['track']}\n"
                f"- 오늘 날짜: {today.isoformat()}\n- 안내 방식: {S['rule']}")

# ==========================================
# 9. 왼쪽 사이드바 (예은 디자인 + 메뉴·대화 삭제)
# ==========================================
PAGES = ["🏠 홈", "🙋 소개", "📚 교육과정", "💼 취업과정", "📁 포트폴리오", "📤 파일 불러오기", "🕸️ 온톨로지", "💬 대화 기록"]

with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <img src="https://raw.githubusercontent.com/Jangrae/img/master/aivle.png" width="145">
        <p style="color:#9aa8b0; font-size:11px;">AIVLE SCHOOL</p>
    </div>
    """, unsafe_allow_html=True)

    st.radio("메뉴", PAGES, key="page", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ 대화 삭제", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.session_state.last_sources = []
        st.rerun()

    st.markdown('<p style="color:white;font-size:12px;margin-top:30px;">AIVLE School<br>Learning Assistant</p>',
                unsafe_allow_html=True)

# ==========================================
# 10. 화면별 내용 (각 화면은 챗봇에 보낼 '화면 설명'을 돌려줌)
# ==========================================
# 10.0 공통 함수 (일정 목록, 달력)
def upcoming_events(n=4, cat=None):
    evs = [e for e in EVENTS if e[1] >= today and (cat is None or e[3] == cat)]
    return evs[:n]


def render_event_list(evs):
    lines = []
    for s, e, title, cat in evs:
        when = fmt_event_date(s, e, stage)
        st.markdown(f"""
        <div class="event-row">
            <span class="dday">{dday_text(s, e, today)}</span>
            <span class="tag tag-{cat}">{cat}</span><span class="event-title">{title}</span>
            <div class="event-date">{when}</div>
        </div>""", unsafe_allow_html=True)
        lines.append(f"{title}({when}, {dday_text(s, e, today)})")
    if not evs:
        st.caption("남은 일정이 없습니다.")
    return lines


def render_calendar(year, month):
    cal = calendar.Calendar(firstweekday=0)
    marks = {}
    for s, e, title, cat in EVENTS:
        d = s
        while d <= e:
            marks.setdefault(d, set()).add(cat)
            d += timedelta(days=1)
    html = f'<div class="cal-wrap"><div class="cal-title">{year}년 {month}월</div><table class="cal"><tr>'
    html += "".join(f"<th>{w}</th>" for w in WEEKDAY) + "</tr>"
    for week in cal.monthdatescalendar(year, month):
        html += "<tr>"
        for d in week:
            cls = []
            if d.month != month: cls.append("other")
            if d == today: cls.append("today")
            if d.weekday() >= 5: cls.append("weekend")
            dots = "".join(f'<span class="dot dot-{c}"></span>' for c in sorted(marks.get(d, [])))
            html += f'<td class="{" ".join(cls)}"><span class="d">{d.day}</span>{dots}</td>'
        html += "</tr>"
    html += "</table></div>"
    st.markdown(html, unsafe_allow_html=True)

# 10.1 홈
def page_home():
    title_col, search_col = st.columns([5, 4])
    with title_col:
        st.markdown(f"""
        <div class="greeting">Hello, {profile['name']}! 👋</div>
        <div class="sub-greeting">{today.year}년 {today.month}월 {today.day}일 {WEEKDAY[today.weekday()]}요일 · {S['label']}</div>
        """, unsafe_allow_html=True)
    with search_col:
        st.text_input("검색", placeholder="궁금한 내용을 검색해 보세요.", label_visibility="collapsed",
                      key="search_input", on_change=on_search)

    welcome = {
        "예비": "지원부터 입교 준비까지<br>단계별로 필요한 정보를 알려드릴게요.",
        "재학": "출결·장려금, 미니프로젝트, 취업 일정까지<br>지금 필요한 정보를 챙겨드릴게요.",
        "수료": "수료 후 1년 동안<br>취업 준비와 선배 사례를 함께 볼 수 있어요.",
    }[stage]
    nxt = upcoming_events(3)
    nxt_html = "".join(f"<p>• {t} <b>{dday_text(s, e, today)}</b></p>" for s, e, t, c in nxt) or "<p>남은 일정이 없습니다.</p>"

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="dashboard-card welcome-card">
            <h3>Welcome back! ✨</h3>
            <p>{welcome}</p>
            <div style="font-size:27px;text-align:right;">{S['emoji']} 📘</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="dashboard-card result-card">
            <h3>다가오는 일정 📅</h3>
            {nxt_html}
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Menu</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    for col, (icon, title, desc, target) in zip([f1, f2, f3], [
            ("📚", "교육과정", "과목·미니프로젝트", "📚 교육과정"),
            ("💼", "취업과정", "취업지원 일정·체크리스트", "💼 취업과정"),
            ("📁", "포트폴리오", "구성 순서·점검", "📁 포트폴리오")]):
        with col:
            st.markdown(f"""<div class="feature-card"><div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>""",
                        unsafe_allow_html=True)
            st.button(f"{title} 열기", key=f"quick_{title}", on_click=go, args=(target,), use_container_width=True)

    st.markdown('<div class="section-title">📅 일정</div>', unsafe_allow_html=True)
    cal_col, list_col = st.columns([3, 2], gap="medium")
    with cal_col:
        months = [(2026, m) for m in range(9, 13)] + [(2027, m) for m in range(1, 4)]
        default = (today.year, today.month) if (today.year, today.month) in months else months[0]
        ym = st.selectbox("월 선택", months, index=months.index(default),
                          format_func=lambda x: f"{x[0]}년 {x[1]}월", label_visibility="collapsed")
        render_calendar(*ym)
    with list_col:
        lines = render_event_list(upcoming_events(5))
    return "홈 대시보드. 다가오는 일정: " + "; ".join(lines)

# 10.2 소개
def page_intro():
    st.markdown('<div class="page-title">🙋 에이블스쿨 소개</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{S["label"]}에게 맞춘 소개 · 자세한 내용은 오른쪽 챗봇에 물어보세요</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="white-card">
    <b>KT 에이블스쿨(AIVLE School)</b>은 고용노동부 'KDT 첨단산업·디지털 선도기업 아카데미' 사업으로 운영되는
    청년 대상 AI·클라우드 실무 인재 양성 과정임. 10기는 2026.9.29 ~ 2027.3.5, 총 106일 840시간임.
    </div>""", unsafe_allow_html=True)
    t1, t2 = st.columns(2, gap="medium")
    with t1:
        st.markdown("""<div class="white-card"><b>🤖 AI 개발자 트랙</b><br><br>
        AI와 Cloud로 서비스를 구현하는 개발자<br>코드를 짜서 무언가를 만드는 데서 보람을 느끼는 사람</div>""",
                    unsafe_allow_html=True)
    with t2:
        st.markdown("""<div class="white-card"><b>📊 DX 컨설턴트 트랙</b><br><br>
        AI와 Cloud로 기업의 디지털 전환을 주도하는 촉진자<br>문제를 정의하고 근거를 들어 설득하는 데 강점이 있는 사람</div>""",
                    unsafe_allow_html=True)
    return "에이블스쿨 소개 화면(운영 기관, 10기 기간, AI 개발자 트랙과 DX 컨설턴트 트랙 비교)"

# 10.3 교육과정
def page_edu():
    st.markdown('<div class="page-title">📚 교육과정</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">과목 흐름과 교육 일정</div>', unsafe_allow_html=True)
    flow = ["파이썬", "데이터 분석", "머신러닝", "딥러닝", "LLM·RAG", "빅프로젝트"]
    cols = st.columns(len(flow))
    for c, f in zip(cols, flow):
        c.markdown(f'<div class="feature-card"><div class="feature-title">{f}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">교육 일정</div>', unsafe_allow_html=True)
    lines = render_event_list([e for e in EVENTS if e[3] == "교육"])
    if stage == "재학":
        week = (today - EDU_START).days // 7 + 1
        st.info(f"오늘은 교육 {week}주차입니다. 수료까지 {(EDU_END - today).days}일 남았어요.")
    return "교육과정 화면. 과목 흐름: " + " → ".join(flow) + ". 교육 일정: " + "; ".join(lines)

# 10.4 취업과정 (체크리스트 공통 사용)
CHECKLISTS = {
    "취업": {
        "예비": ["지원 자격(연령·학력·미취업) 확인", "국민내일배움카드 발급 가능 여부 확인", "트랙 선택 이유 정리"],
        "재학": ["취업지원 플랫폼 회원가입", "포트폴리오 컨설팅 신청", "자기소개서 컨설팅 신청", "면접 컨설팅 신청", "잡페어 참여 기업 확인"],
        "수료": ["지원 직무 JD 3줄 분석", "포트폴리오 최신화", "면접 답변 경험 8문항 정리", "잡페어 후속 지원"],
    },
    "포폴": {
        "예비": ["지금까지의 프로젝트·경험 목록 만들기", "GitHub 계정 만들기"],
        "재학": ["프로젝트 1: 문제 정의부터 제안까지 간 대표 프로젝트", "프로젝트 2: 데이터로 근거를 만든 프로젝트",
                 "프로젝트 3: 문서화·제안·고객 소통이 보이는 프로젝트", "각 프로젝트에 개요·판단 과정·결과 그림 넣기"],
        "수료": ["지원 직무에 맞춰 프로젝트 순서 바꾸기", "빅프로젝트를 대표 프로젝트로 정리", "결과를 숫자로 표현"],
    },
}


def render_checklist(kind):
    items = CHECKLISTS[kind][stage]
    done = 0
    for i, item in enumerate(items):
        key = f"{kind}_{stage}_{i}"
        v = st.checkbox(item, value=st.session_state.checks.get(key, False), key=f"chk_{key}")
        st.session_state.checks[key] = v
        done += v
    st.progress(done / len(items), text=f"{done}/{len(items)} 완료")
    return f"체크리스트 {done}/{len(items)} 완료. 남은 항목: " + ", ".join(
        it for i, it in enumerate(items) if not st.session_state.checks.get(f"{kind}_{stage}_{i}"))


def page_job():
    st.markdown('<div class="page-title">💼 취업과정</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">10기 취업지원 일정과 나의 체크리스트</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        lines = render_event_list([e for e in EVENTS if e[3] in ("취업", "포폴")])
    with c2:
        st.markdown('<div class="info-title">✅ 나의 체크리스트</div>', unsafe_allow_html=True)
        chk = render_checklist("취업")
    return "취업과정 화면. 취업 일정: " + "; ".join(lines) + ". " + chk

# 10.5 포트폴리오
def page_portfolio():
    st.markdown('<div class="page-title">📁 포트폴리오</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">백서 7.3 기준 구성 순서와 점검</div>', unsafe_allow_html=True)
    lines = render_event_list([e for e in EVENTS if e[3] == "포폴"])
    st.markdown('<div class="info-title">✅ 포트폴리오 점검</div>', unsafe_allow_html=True)
    chk = render_checklist("포폴")
    return "포트폴리오 화면. 일정: " + "; ".join(lines) + ". " + chk

# 10.6 파일 불러오기
def page_upload():
    st.markdown('<div class="page-title">📤 파일 불러오기</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">PDF를 올리면 01번 노트북의 rag_pipeline()으로 바로 질문할 수 있어요</div>', unsafe_allow_html=True)
    up = st.file_uploader("PDF 업로드", type=["pdf"], key="uploader")
    if up and st.button("📥 이 문서로 RAG 만들기", use_container_width=True):
        with st.spinner("문서를 읽고 벡터 저장소를 만드는 중입니다..."):
            tmp = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4().hex}.pdf")
            with open(tmp, "wb") as f:
                f.write(up.getbuffer())
            st.session_state.upload_chain = rag_pipeline(tmp, collection_name="user_docs")
            st.session_state.upload_name = up.name
            st.session_state.answer_source = "업로드 문서"
        st.success(f"{up.name} 준비 완료! 오른쪽 챗봇의 '답변 기준'이 업로드 문서로 바뀌었어요.")
    if st.session_state.upload_name:
        st.info(f"현재 업로드 문서: {st.session_state.upload_name}")
    return f"파일 불러오기 화면. 업로드 문서: {st.session_state.upload_name or '없음'}"

# 10.7 온톨로지 (교육 / 취업 / 포폴 멀티탭)
def page_ontology():
    st.markdown('<div class="page-title">🕸️ 나만의 온톨로지</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">백서에서 뽑은 개체-관계 지도 · {S["label"]} 관심 키워드가 먼저 보여요</div>', unsafe_allow_html=True)
    raw = None
    if os.path.exists(ONTOLOGY_FILE):
        with open(ONTOLOGY_FILE, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        up = st.file_uploader("aivle_ontology.json 업로드 (01번 노트북에서 저장한 파일)", type=["json"])
        if up:
            raw = up.getvalue().decode("utf-8")
    if not raw:
        st.info("온톨로지 파일이 없습니다. 01번 노트북에서 만든 aivle_ontology.json을 저장소에 올리거나 위에서 업로드하세요.")
        return "온톨로지 화면(파일 없음)"

    edges = load_ontology_text(raw)
    entities = sorted({e["subject"] for e in edges} | {e["object"] for e in edges})
    st.multiselect("📌 내 관심 개체 (고정하면 그래프에 항상 표시)", entities, key="pinned")

    shown_lines = []
    tabs = st.tabs(["📚 교육", "💼 취업", "📁 포폴"])
    for tab, name in zip(tabs, ["교육", "취업", "포폴"]):
        with tab:
            tab_edges = [e for e in edges if edge_in_tab(e, name)]
            focus = S["focus"]
            def score(e):
                text = f'{e["subject"]} {e["object"]}'
                pin = 2 if (e["subject"] in st.session_state.pinned or e["object"] in st.session_state.pinned) else 0
                return pin + sum(k in text for k in focus)
            tab_edges.sort(key=score, reverse=True)
            kw = st.text_input("검색", key=f"onto_kw_{name}", placeholder="예: 컨설팅, 미니프로젝트")
            if kw:
                tab_edges = [e for e in tab_edges if kw in e["subject"] or kw in e["object"] or kw in e["evidence"]]
            st.caption(f"관계 {len(tab_edges)}개")

            top = tab_edges[:20]
            if top:
                def q(x):
                    return x.replace('"', "'")[:30]
                dot = ['digraph G {', 'rankdir=LR;', 'node [shape=box, style="rounded,filled", fillcolor="#e5f5f5", color="#43aebe", fontname="sans-serif"];',
                       'edge [color="#83b3f5", fontname="sans-serif", fontsize=10];']
                for e in top:
                    dot.append(f'"{q(e["subject"])}" -> "{q(e["object"])}" [label="{q(e["relation"])}"];')
                dot.append("}")
                st.graphviz_chart("\n".join(dot), use_container_width=True)

                df = pd.DataFrame(top)[["subject", "relation", "object", "evidence", "page"]]
                df.columns = ["개체", "관계", "대상", "근거 원문", "쪽"]
                st.dataframe(df, use_container_width=True, hide_index=True)

                pick = st.selectbox("이 관계를 챗봇에게 자세히 물어보기", range(len(top)), key=f"onto_pick_{name}",
                                    format_func=lambda i: f'{top[i]["subject"]} → {top[i]["relation"]} → {top[i]["object"]}')
                st.button("💬 챗봇에 질문", key=f"onto_ask_{name}", on_click=ask,
                          args=(f'"{top[pick]["subject"]}"의 "{top[pick]["relation"]}"({top[pick]["object"]})에 대해 자세히 알려주세요.',))
                shown_lines += [f'[{name}] {e["subject"]} -{e["relation"]}-> {e["object"]}' for e in top[:8]]
    return "온톨로지 화면. 보이는 관계 일부: " + "; ".join(shown_lines)

# 10.8 대화 기록
def page_archive():
    st.markdown('<div class="page-title">💬 대화 기록</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">지난 질문과 답변을 날짜별로 다시 볼 수 있어요</div>', unsafe_allow_html=True)
    arc = st.session_state.archive
    if not arc:
        st.info("아직 저장된 대화가 없습니다.")
        return "대화 기록 화면(기록 없음)"
    dates = sorted({a["date"] for a in arc}, reverse=True)
    d = st.selectbox("날짜 선택", dates)
    kw = st.text_input("기록 검색", placeholder="질문이나 답변에 들어간 단어")
    rows = [a for a in arc if a["date"] == d and (not kw or kw in a["question"] or kw in a["answer"])]
    for a in reversed(rows):
        with st.expander(f'🕒 {a["time"]} · {a["question"][:40]}'):
            st.caption(f'{a["stage"]} · {a["page"]} · 기준: {a.get("source", "")}')
            st.markdown(f'**Q.** {a["question"]}')
            st.markdown(a["answer"])
            st.button("🔁 다시 질문하기", key=f're_{a["id"]}', on_click=ask, args=(a["question"],))
    c1, c2 = st.columns(2)
    with c1:
        csv = pd.DataFrame(arc).to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 전체 기록 CSV 다운로드", csv, "aivle_chat_log.csv", "text/csv", use_container_width=True)
    with c2:
        if st.button("🗑️ 이 날짜 기록 삭제", use_container_width=True):
            st.session_state.archive = [a for a in arc if a["date"] != d]
            save_archive()
            st.rerun()
    return f"대화 기록 화면. {d} 기록 {len(rows)}건"


PAGE_FUNCS = {"🏠 홈": page_home, "🙋 소개": page_intro, "📚 교육과정": page_edu, "💼 취업과정": page_job,
              "📁 포트폴리오": page_portfolio, "📤 파일 불러오기": page_upload, "🕸️ 온톨로지": page_ontology,
              "💬 대화 기록": page_archive}

# ==========================================
# 11. 메인 레이아웃 (모든 화면 공통)
# ==========================================
# 11.1 맨 오른쪽 위: 단계 전환 (예비 에이블러 / 에이블러 / 수료자)
STAGE_LABELS = {"예비": "🌱 예비 에이블러", "재학": "🚀 에이블러", "수료": "🎓 수료자"}
LABEL_TO_STAGE = {v: k for k, v in STAGE_LABELS.items()}


def on_stage_change():
    v = st.session_state.stage_switch
    if v is None:                                   # 같은 버튼을 다시 눌러 선택이 풀린 경우
        st.session_state.stage_switch = STAGE_LABELS[st.session_state.profile["stage"]]
    else:
        st.session_state.profile["stage"] = LABEL_TO_STAGE[v]


if st.session_state.get("stage_switch") != STAGE_LABELS[stage]:
    st.session_state.stage_switch = STAGE_LABELS[stage]

_, top_right = st.columns([7, 3])
with top_right:
    st.segmented_control("나의 단계", list(STAGE_LABELS.values()), key="stage_switch",
                         on_change=on_stage_change, label_visibility="collapsed")

left, right = st.columns([7, 3], gap="large")

# 11.2 왼쪽: 선택한 메뉴의 화면
with left:
    screen_text = PAGE_FUNCS[st.session_state.page]()

with right:
    # 11.3 오른쪽 프로필
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-icon">{S['emoji']}</div>
        <div class="profile-name">{profile['name']}</div>
        <div class="profile-desc">{S['label']} · {profile['track']}</div>
    </div>""", unsafe_allow_html=True)
    with st.popover("✏️ 프로필 수정", use_container_width=True):
        new_name = st.text_input("이름(닉네임)", value=profile["name"], key="edit_name")
        tracks = ["DX 컨설턴트", "AI 개발자"]
        new_track = st.radio("트랙", tracks, index=tracks.index(profile["track"]), horizontal=True, key="edit_track")
        if st.button("저장", key="edit_save", use_container_width=True):
            profile["name"] = new_name.strip() or profile["name"]
            profile["track"] = new_track
            st.rerun()
        if st.button("처음 화면으로", key="edit_reset", use_container_width=True):
            st.session_state.profile = None
            st.rerun()

    # 11.4 챗봇 (모든 화면 공통)

    with st.container(key="chat_panel"):
        st.markdown(f"""
        <div style="font-size:17px;font-weight:800;color:#19334d;margin-bottom:6px;">{bot_img_html(44)} AIVLE 학습 도우미</div>
        <div style="font-size:12px;color:#9baab4;margin-bottom:10px;">지금 보고 있는 화면을 함께 보고 답해요.</div>
        """, unsafe_allow_html=True)

        sources = ["AIVLE School 백서"] + (["업로드 문서"] if st.session_state.upload_chain else [])
        if st.session_state.answer_source not in sources:
            st.session_state.answer_source = sources[0]
        st.radio("답변 기준", sources, key="answer_source", horizontal=True)
        st.toggle("현재 화면 내용 함께 보내기", key="send_screen")

        if not st.session_state.messages:
            st.info(f"안녕하세요, {profile['name']}님! 😊 무엇이 궁금하신가요?")

        for message in st.session_state.messages[-8:]:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("assistant", avatar=bot_avatar()):
                    st.markdown(message.content)

        if st.session_state.last_sources:
            with st.expander("📄 직전 답변의 근거 문서"):
                for src in st.session_state.last_sources:
                    st.markdown(f'<div class="src-box">{src}</div>', unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            question = st.text_input("질문 입력", placeholder="예: 이 화면에서 제일 급한 일정이 뭐야?",
                                     label_visibility="collapsed")
            submitted = st.form_submit_button("질문 보내기 ➤", use_container_width=True)

        # ----- 질문 처리 -----
        prompt_text = None
        if submitted and question.strip():
            prompt_text = question.strip()
        elif st.session_state.pending_prompt:
            prompt_text = st.session_state.pending_prompt

        if prompt_text:
            st.session_state.pending_prompt = None
            history = st.session_state.messages[-6:]
            with st.chat_message("user"):
                st.markdown(prompt_text)
            try:
                with st.chat_message("assistant", avatar=bot_avatar()):
                    if st.session_state.answer_source == "업로드 문서" and st.session_state.upload_chain:
                        full = st.write_stream(st.session_state.upload_chain.stream({"input": prompt_text}))
                        st.session_state.last_sources = [f"업로드 문서: {st.session_state.upload_name}"]
                    else:
                        screen = screen_text if st.session_state.send_screen else ""
                        stream, docs, search_q = answer_stream(prompt_text, history, profile_text, screen)
                        full = st.write_stream(stream)
                        st.session_state.last_sources = [
                            f'p.{d.metadata.get("page", "?")} · {d.page_content[:120]}…' for d in docs]
                st.session_state.messages.append(HumanMessage(content=prompt_text))
                st.session_state.messages.append(AIMessage(content=full))
                now = datetime.now()
                st.session_state.archive.append({
                    "id": uuid.uuid4().hex[:8], "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"),
                    "stage": S["label"], "page": st.session_state.page, "source": st.session_state.answer_source,
                    "question": prompt_text, "answer": full})
                save_archive()
            except Exception as e:
                st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
            st.rerun()

    # 11.5 추천 질문 (단계별)
    st.markdown('<div class="section-title">💡 추천 질문</div>', unsafe_allow_html=True)
    for i, q in enumerate(S["questions"]):
        st.button(q, key=f"recommend_{i}", on_click=ask, args=(q,), use_container_width=True)
