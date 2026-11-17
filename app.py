
import streamlit as st
import calendar
import io
import os
import uuid
import json
from datetime import date, datetime
from html import escape

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter


# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="AIVLE School Learning Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"


# =========================================================
# 2. CSS 디자인
# =========================================================

st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background: linear-gradient(120deg, #f5fbfc, #eaf7f8);
    color: #193653;
    font-family: 'Malgun Gothic', sans-serif;
}

/* 기본 여백: Streamlit 상단 툴바와 본문이 겹치지 않도록 공통 적용 */
.block-container,
.main .block-container,
div[data-testid="stAppViewBlockContainer"],
.stAppViewBlockContainer {
    padding-top: 4.25rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* Streamlit 상단 헤더 공간 확보 */
header[data-testid="stHeader"] {
    height: 3rem;
    background: transparent;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #55bdc5, #63c6cc);
    min-width: 210px;
    max-width: 210px;
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* 바로가기 링크: 흰색 카드 위에 진한 글씨가 항상 보이도록 고정 */
section[data-testid="stSidebar"] a.sidebar-shortcut-link,
section[data-testid="stSidebar"] a.sidebar-shortcut-link:visited {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 6px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    min-height: 48px !important;
    padding: 10px 11px !important;
    margin: 7px 0 !important;
    border-radius: 11px !important;
    border: 1px solid rgba(255,255,255,0.85) !important;
    background: #ffffff !important;
    color: #173653 !important;
    -webkit-text-fill-color: #173653 !important;
    text-decoration: none !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
    opacity: 1 !important;
    box-shadow: 0 2px 7px rgba(25,70,90,0.08) !important;
}
section[data-testid="stSidebar"] a.sidebar-shortcut-link span {
    color: #173653 !important;
    -webkit-text-fill-color: #173653 !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] .shortcut-label {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
    min-width: 0 !important;
}
section[data-testid="stSidebar"] .shortcut-arrow {
    flex: 0 0 auto !important;
    color: #159eae !important;
    -webkit-text-fill-color: #159eae !important;
}
section[data-testid="stSidebar"] a.sidebar-shortcut-link:hover {
    background: #eafafa !important;
    color: #087f8b !important;
    -webkit-text-fill-color: #087f8b !important;
}

/* 사이드바 메뉴 버튼 */
section[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    min-height: 43px;
    text-align: left;
    padding: 8px 13px;
    margin: 2px 0;
    border: 1px solid transparent;
    border-radius: 12px;
    background: transparent;
    color: white !important;
    font-size: 15px;
    font-weight: 800;
    box-shadow: none;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.22);
    border-color: rgba(255,255,255,0.35);
    color: white !important;
}

/* 선택된 메뉴: Streamlit 버전에 따라 kind / data-testid가 달라질 수 있어 함께 지정 */
section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] div.stButton > [data-testid="stBaseButton-primary"] {
    background: #eafafa !important;
    border: 1px solid #eafafa !important;
    color: #193653 !important;
}

/* 선택된 메뉴 안쪽의 p, span 등에도 글자색을 직접 적용
   (사이드바 전체 흰색 글자 규칙이 선택 메뉴에 상속되는 현상 방지) */
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] *,
section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-primary"] *,
section[data-testid="stSidebar"] div.stButton > [data-testid="stBaseButton-primary"] * {
    color: #193653 !important;
    -webkit-text-fill-color: #193653 !important;
}

/* 비선택 메뉴는 흰색 글자 유지 */
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"],
section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"] {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] *,
section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* 로고는 사이드바 기본 글자색 규칙의 영향을 받지 않도록 고정 */
.logo-box .logo-main {
    color: #173956 !important;
}
.logo-box .logo-sub {
    color: #28b7c0 !important;
}

/* 사이드바 로고 */
.logo-box {
    background: white;
    border-radius: 15px;
    padding: 17px 8px;
    text-align: center;
    margin-bottom: 25px;
}

.logo-main {
    color: #173956;
    font-size: 27px;
    font-weight: 900;
}

.logo-sub {
    color: #28b7c0;
    font-size: 12px;
}

/* 프로필 설정 화면 */
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.72);
    border: 1px solid #dce9ed;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 5px 18px rgba(31,82,105,0.06);
}
div[data-testid="stForm"] label,
div[data-testid="stForm"] p {
    color: #193653 !important;
    font-weight: 700;
}
div[data-testid="stForm"] input {
    border-radius: 9px;
}
div[data-testid="stForm"] button[kind="primary"] {
    background: #ff4b4b !important;
    border: 0 !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 9px !important;
    min-height: 42px;
}

/* 제목 */
.page-title {
    font-size: 30px;
    font-weight: 900;
    color: #173653;
    margin-bottom: 3px;
}

.page-subtitle {
    color: #71869c;
    font-size: 14px;
    margin-bottom: 20px;
}

/* 배너 */
.hero {
    background: linear-gradient(110deg, #d9f5f5, #c8eff2);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 20px;
    min-height: 145px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.hero h1 {
    font-size: 29px;
    font-weight: 900;
    color: #173653;
    margin: 0 0 8px 0;
}

.hero h3 {
    color: #159eae;
    font-size: 18px;
    margin: 0 0 12px 0;
}

.hero p {
    color: #29465f;
    font-size: 14px;
    line-height: 1.8;
    margin: 0;
}

.hero-icon {
    font-size: 75px;
    padding: 10px;
}

/* 섹션 제목 */
.section-title {
    font-size: 19px;
    font-weight: 900;
    color: #173653;
    margin: 20px 0 12px 0;
}

.section-number {
    display: inline-block;
    background: linear-gradient(90deg, #20bdc4, #51d2d1);
    color: white;
    padding: 4px 11px;
    border-radius: 9px;
    font-size: 13px;
    margin-right: 8px;
}

/* 일반 카드 */
.info-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 15px rgba(35,90,110,0.07);
    height: 100%;
}

.info-card h3 {
    color: #173653;
    font-size: 17px;
    font-weight: 900;
    margin: 5px 0 8px 0;
}

.info-card p {
    color: #71869c;
    font-size: 13px;
    line-height: 1.7;
    margin: 0;
}

.card-icon {
    font-size: 35px;
    text-align: center;
    margin-bottom: 7px;
}

/* 트랙 카드 */
.track-card {
    color: white;
    border-radius: 19px;
    padding: 22px;
    min-height: 130px;
}

.track-card h2 {
    font-size: 21px;
    margin: 5px 0;
}

.track-card p {
    font-size: 13px;
    line-height: 1.6;
    color: white;
}

.track-ai {
    background: linear-gradient(120deg, #27b9c0, #3abac5);
}

.track-dx {
    background: linear-gradient(120deg, #5c9fee, #79adf6);
}

/* 교육 로드맵 */
.roadmap-card {
    background: white;
    border: 1px solid #e2f0f4;
    border-radius: 17px;
    padding: 17px 10px;
    text-align: center;
    min-height: 150px;
    box-shadow: 0 4px 12px rgba(35,90,110,0.05);
}

.roadmap-icon {
    font-size: 34px;
}

.step {
    display: inline-block;
    background: #21b8c0;
    color: white;
    border-radius: 20px;
    padding: 4px 13px;
    font-size: 11px;
    font-weight: 800;
    margin: 5px;
}

.roadmap-card h3 {
    font-size: 16px;
    color: #173653;
    margin: 5px 0;
}

.roadmap-card p {
    font-size: 12px;
    color: #71869c;
    line-height: 1.5;
}

/* 학습 분야 */
.subject-card {
    background: white;
    border-radius: 17px;
    padding: 15px 8px;
    text-align: center;
    min-height: 120px;
    box-shadow: 0 4px 12px rgba(35,90,110,0.06);
    border: 1px solid #e7f1f4;
}

.subject-icon {
    font-size: 32px;
}

.subject-card h4 {
    color: #173653;
    font-size: 15px;
    margin: 5px 0;
}

.subject-card p {
    color: #71869c;
    font-size: 11px;
    line-height: 1.5;
    margin: 0;
}

/* 프로젝트 카드 */
.project-card {
    background: white;
    border-radius: 17px;
    padding: 18px;
    border: 1px solid #e4eff3;
    box-shadow: 0 4px 12px rgba(35,90,110,0.05);
    margin-bottom: 10px;
}

.project-card h3 {
    color: #173653;
    font-size: 17px;
}

.project-card p {
    color: #71869c;
    font-size: 12px;
    line-height: 1.8;
}

/* 하단 배너 */
.bottom-banner {
    background: linear-gradient(100deg, #117d8b, #168b99);
    border-radius: 18px;
    padding: 20px;
    color: white;
    margin-top: 22px;
}

.bottom-banner h2 {
    color: white;
    font-size: 22px;
    margin: 0 0 5px 0;
}

.bottom-banner p {
    color: white;
    margin: 0;
    font-size: 14px;
}

/* 오른쪽 패널 */
.right-card {
    background: white;
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 17px;
    box-shadow: 0 5px 15px rgba(35,90,110,0.06);
}

.assistant-icon {
    background: #e9f2fc;
    border-radius: 50%;
    width: 85px;
    height: 85px;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 42px;
}

.right-card h3 {
    color: #173653;
    font-size: 17px;
    margin-bottom: 12px;
}

.right-card p {
    color: #8294a8;
    font-size: 12px;
    line-height: 1.8;
}

.question-item {
    background: white;
    border: 1px solid #e6eff2;
    border-radius: 12px;
    padding: 11px;
    text-align: center;
    font-size: 12px;
    color: #36536e;
    margin-bottom: 9px;
}

.small-text {
    color: #91a5b5;
    font-size: 11px;
}

/* 홈 환영 카드 */
.welcome-card {
    border-radius: 20px;
    padding: 22px;
    min-height: 155px;
    color: white;
}

.welcome-card h3 {
    font-size: 17px;
    color: white;
}

.welcome-card p {
    color: white;
    font-size: 12px;
    line-height: 1.8;
}

.welcome-one {
    background: #39afbd;
}

.welcome-two {
    background: #80acf0;
}

/* 달력 */
.calendar-box {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 5px 15px rgba(35,90,110,0.06);
}

.calendar-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 7px;
    text-align: center;
}

.calendar-day {
    padding: 11px 3px;
    border: 1px solid #e8f0f4;
    border-radius: 10px;
    font-size: 11px;
    color: #36536e;
}

.calendar-head {
    color: #8499aa;
    font-weight: 800;
    padding: 8px;
    font-size: 12px;
}

.today {
    background: #32b3bf;
    color: white;
    font-weight: 900;
}

/* Streamlit 버튼 */
.stButton > button {
    border-radius: 12px;
    border: 1px solid #dcebf0;
    background: white;
    color: #244763;
    font-weight: 700;
}

.stButton > button:hover {
    background: #e7f8f9;
    color: #159eae;
    border-color: #40c1c8;
}

/* 입력창 */
.stChatInput textarea {
    color: #193653;
}

div[data-testid="stChatMessage"] {
    background: white;
    border-radius: 14px;
}

hr {
    border-color: #d9eaee;
}

/* 사이드바 선택 메뉴 색상 최종 보정 */
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: #eafafa !important;
    border-color: #eafafa !important;
    color: #193653 !important;
    -webkit-text-fill-color: #193653 !important;
}
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] *,
section[data-testid="stSidebar"] button[kind="primary"] * {
    color: #193653 !important;
    -webkit-text-fill-color: #193653 !important;
}


.profile-stage-badge {
    display: inline-block;
    margin-top: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    background: #e4f7f7;
    color: #159eaa;
    border: 1px solid #c7eeee;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.4;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. RAG 설정
# =========================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource
def get_retriever():

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


@st.cache_resource
def get_chain():

    retriever = get_retriever()

    s_msg = """
    당신은 주어진 문서를 기반으로 답변하는
    AIVLE School 학습 도우미입니다.

    반드시 제공된 문서를 우선적으로 참고하여 답변하세요.
    문서에 없는 내용은 추측하지 말고 안내하세요.

    [문서]
    {context}
    """

    h_msg = "{input}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg),
        ("human", h_msg)
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0
    )

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


# 3-1. 솔 챗봇 파이프라인 (01번 노트북 rag_pipeline 고도화 버전)
#      Hybrid 검색(벡터 + BM25 한글 2-gram, RRF) → 쿼리 리라이팅 → 프로필·화면 반영 답변
import re
import math
from collections import Counter
from langchain_core.prompts import MessagesPlaceholder

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



# =========================================================
# 4. 세션 상태
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_date" not in st.session_state:
    st.session_state.selected_date = date.today()

if "calendar_year" not in st.session_state:
    st.session_state.calendar_year = date.today().year

if "calendar_month" not in st.session_state:
    st.session_state.calendar_month = date.today().month

if "menu" not in st.session_state:
    st.session_state.menu = "🏠 홈"

# 최초 접속 시 프로필 설정을 먼저 보여주기 위한 상태
if "profile_completed" not in st.session_state:
    st.session_state.profile_completed = False

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "",
        "stage": "에이블러(교육생)",
        "track": "DX 컨설턴트 트랙",
        "company": "",
        "job": "",
    }

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "document_preview" not in st.session_state:
    st.session_state.document_preview = None

if "ontology_tab" not in st.session_state:
    st.session_state.ontology_tab = "지식 그래프"

if "ontology_query_result" not in st.session_state:
    st.session_state.ontology_query_result = None


# =========================================================
# 5. 공통 HTML 함수
# =========================================================

def section_title(number, title):

    st.markdown(
        f"""
        <div class="section-title">
            <span class="section-number">{number:02d}.</span>
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )


def info_card(icon, title, description):

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-icon">{icon}</div>
            <h3 style="text-align:center;">{title}</h3>
            <p style="text-align:center;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def subject_card(icon, title, description):

    st.markdown(
        f"""
        <div class="subject-card">
            <div class="subject-icon">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 6. 프로필 설정 및 맞춤 추천 질문
# =========================================================

def get_recommended_questions():
    """사용자 단계, 트랙, 희망 기업/직무를 반영한 추천 질문을 반환합니다."""
    profile = st.session_state.get("user_profile", {})
    stage = profile.get("stage", "에이블러(교육생)")
    track = profile.get("track", "DX 컨설턴트 트랙")
    company = profile.get("company", "").strip()
    job = profile.get("job", "").strip()

    if stage == "예비 에이블러":
        stage_question = "KT 에이블스쿨 지원 자격과 선발 절차를 알려주세요."
        stage_question_2 = "교육 시작 전 미리 준비하면 좋은 기초 역량은 무엇인가요?"
    elif stage == "수료자(1년)":
        stage_question = "수료 후 취업 준비 로드맵을 단계별로 알려주세요."
        stage_question_2 = "프로젝트 경험을 자기소개서와 면접에서 활용하는 방법은?"
    else:
        stage_question = "현재 교육 단계에서 우선적으로 복습해야 할 내용은 무엇인가요?"
        stage_question_2 = "미니프로젝트 결과물을 포트폴리오로 정리하는 방법은?"

    if "AI 개발자" in track:
        track_question = "AI 개발자에게 필요한 Python·머신러닝·딥러닝 역량은 무엇인가요?"
        track_question_2 = "AI 개발자 트랙 프로젝트를 기술 면접에서 설명하는 방법은?"
    else:
        track_question = "DX 컨설턴트에게 필요한 데이터 분석 및 비즈니스 문제 해결 역량은?"
        track_question_2 = "데이터 분석 프로젝트를 DX 컨설턴트 직무 성과로 표현하는 방법은?"

    questions = [stage_question, track_question]
    if company:
        questions.append(f"{company}의 채용 준비 시 중점적으로 확인해야 할 역량은?")
    if job:
        questions.append(f"희망 직무인 {job}에 맞는 포트폴리오 구성 방법은?")
    questions.extend([stage_question_2, track_question_2])
    return questions[:5]


def page_profile():
    profile = st.session_state.user_profile
    st.markdown("""
    <div class="hero" style="min-height:110px; margin-bottom:18px;">
        <div>
            <h1 style="font-size:27px;">AIVLE School 학습 도우미 👋</h1>
            <p style="margin:8px 0 0; color:#55718a;">먼저 프로필을 설정하면, 단계와 트랙에 맞는 안내와 챗봇 답변을 보여드려요.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, form_col = st.columns([0.42, 1.58], gap="large")
    with left:
        st.markdown("""
        <div class="right-card" style="text-align:center; padding:30px 18px;">
            <div style="font-size:55px;">👩🏻‍💻</div>
            <h3 style="color:#193653;">나에게 맞는 학습 도우미</h3>
            <p>현재 상황과 목표를 알려주시면<br>맞춤형 학습·취업 정보를 추천해 드립니다.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card" style="margin-top:14px;">
            <h3>🔒 프로필 정보 안내</h3>
            <p>입력한 정보는 이 앱의 추천 질문과 화면 안내를 개인화하는 데 사용됩니다.</p>
        </div>
        """, unsafe_allow_html=True)

    with form_col:
        st.markdown('<div class="section-title"><span class="section-number">01.</span> 프로필 설정</div>', unsafe_allow_html=True)
        with st.form("profile_setup_form"):
            name = st.text_input("이름 (닉네임)", value=profile.get("name", ""), placeholder="예: 홍길동")
            stage = st.radio(
                "현재 단계",
                ["예비 에이블러", "에이블러(교육생)", "수료자(1년)"],
                index=["예비 에이블러", "에이블러(교육생)", "수료자(1년)"].index(profile.get("stage", "에이블러(교육생)")),
                horizontal=True
            )
            track = st.radio(
                "트랙",
                ["DX 컨설턴트 트랙", "AI 개발자 트랙"],
                index=0 if "DX" in profile.get("track", "DX") else 1,
                horizontal=True
            )
            company = st.text_input("희망 기업 (선택)", value=profile.get("company", ""), placeholder="예: KT, 금융권, 공기업")
            job = st.text_input("희망 직무 (선택)", value=profile.get("job", ""), placeholder="예: DX 컨설턴트, 데이터 분석가")
            submitted = st.form_submit_button("시작하기 →", use_container_width=True, type="primary")

            if submitted:
                st.session_state.user_profile = {
                    "name": name.strip(),
                    "stage": stage,
                    "track": track,
                    "company": company.strip(),
                    "job": job.strip(),
                }
                st.session_state.profile_completed = True
                st.session_state.menu = "🏠 홈"
                st.rerun()


# =========================================================
# 7. 왼쪽 사이드바
# =========================================================

menu_items = [
    "🏠 홈",
    "👋 소개",
    "📚 교육과정",
    "💼 취업과정",
    "📁 포트폴리오",
    "📂 파일 불러오기",
    "🕸️ 온톨로지"
]

if st.session_state.profile_completed:
    with st.sidebar:

        # AIVLE 로고
        st.markdown("""
        <div class="logo-box">
            <div class="logo-main">◢ AIVLE</div>
            <div class="logo-sub">AIVLE SCHOOL</div>
        </div>
        """, unsafe_allow_html=True)

        # 라디오 버튼 대신 일반 버튼으로 메뉴 구성
        for item in menu_items:
            if st.button(
                item,
                key=f"nav_{item}",
                use_container_width=True,
                type="primary" if st.session_state.menu == item else "secondary"
            ):
                st.session_state.menu = item
                st.rerun()

        st.markdown("<br><hr>", unsafe_allow_html=True)

        # 바로가기: 페이지 이름을 한글로 표시하고 글자색을 진하게 고정
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("#### 🔗 바로가기")

        shortcut_links = [
            ("🎓", "에이블 에듀 LMS", "https://aivle.edu.kt.co.kr"),
            ("🏫", "에이블스쿨 공식 홈페이지", "https://aivle.kt.co.kr/home/main/indexMain"),
            ("📸", "에이블스쿨 인스타그램", "https://www.instagram.com/aivlestory?igshid=YmMyMTA2M2Y%3D"),
            ("💼", "취업지원 플랫폼", "https://aivle10.univjob.co.kr"),
        ]

        for link_icon, link_label, link_url in shortcut_links:
            st.markdown(
                f"""
                <a class="sidebar-shortcut-link"
                   href="{link_url}" target="_blank" rel="noopener noreferrer">
                    <span class="shortcut-label">
                        <span class="shortcut-icon">{link_icon}</span>
                        <span>{link_label}</span>
                    </span>
                    <span class="shortcut-arrow">↗</span>
                </a>
                """,
                unsafe_allow_html=True
            )

        st.markdown("""
        <br><br>
        <div style="font-size:12px; line-height:1.8; color:white;">
            AIVLE School<br>
            Learning Assistant
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 7-0. 오른쪽 챗봇 (모든 화면 공통, 백서 RAG)
# =========================================================

def profile_text_for_chat():
    p = st.session_state.get("user_profile", {})
    return (f"- 이름: {p.get('name') or '에이블러'}\n- 단계: {p.get('stage')}\n- 트랙: {p.get('track')}\n"
            f"- 희망 기업: {p.get('company') or '미입력'}\n- 희망 직무: {p.get('job') or '미입력'}\n"
            f"- 오늘 날짜: {date.today().isoformat()}\n"
            "- 안내 방식: 예비 에이블러는 지원·입교 준비 위주로 쉽게, 에이블러는 출결·일정·프로젝트 위주로 날짜까지, "
            "수료자는 취업·포트폴리오·면접 위주로 안내함.")


def right_chatbot():

    st.markdown("""
    <div style="font-size:18px;font-weight:900;color:#193653;margin:22px 0 10px;">
        💬 AIVLE 챗봇 <span style="font-size:12px;color:#28b7c0;">백서 기반 RAG</span>
    </div>
    """, unsafe_allow_html=True)

    box = st.container(height=420, border=True)

    with box:
        if not st.session_state.messages:
            st.caption("백서를 근거로 답해요. 아래에 질문을 입력하거나 추천 질문을 눌러 보세요.")
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

    with st.form("right_chat_form", clear_on_submit=True):
        typed = st.text_input("질문", placeholder="예: 정상수료 기준이 뭐예요?",
                              label_visibility="collapsed", key="right_chat_text")
        sent = st.form_submit_button("보내기 ➤", use_container_width=True, type="primary")

    question = typed.strip() if sent and typed.strip() else st.session_state.pop("chat_pending", None)

    if question:
        history = st.session_state.messages[-6:]
        st.session_state.messages.append(HumanMessage(content=question))
        with box:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    stream, docs, search_q = answer_stream(
                        question, history, profile_text_for_chat(),
                        f"현재 메뉴: {st.session_state.get('menu', '홈')}")
                    reply = st.write_stream(stream)
                    st.session_state.messages.append(AIMessage(content=reply))
                    if docs:
                        with st.expander(f"📎 근거 문서 {len(docs)}개 · 검색어: {search_q}"):
                            for d in docs:
                                st.caption(d.page_content[:160].replace("\n", " ") + " …")
                except Exception as e:
                    st.session_state.messages.pop()
                    st.error(f"답변 생성 중 오류가 발생했습니다: {e}")


# =========================================================
# 7. 오른쪽 공통 패널
# =========================================================

def right_panel():

    # 홈 화면의 Assistant 카드에 프로필에서 선택한 현재 단계를 표시합니다.
    current_stage = st.session_state.get("user_profile", {}).get("stage", "에이블러(교육생)")
    stage_html = ""
    if st.session_state.get("menu") == "🏠 홈":
        stage_html = f'<div class="profile-stage-badge">현재 단계&nbsp; · &nbsp;{escape(current_stage)}</div>'

    st.markdown(f"""
    <div class="right-card" style="text-align:center;">
        <div class="assistant-icon">👩🏻‍💻</div>
        <h3 style="margin-top:12px;">AIVLE Assistant</h3>
        <p>Your Learning Partner</p>
        {stage_html}
    </div>

    <div class="right-card">
        <h3>📘 학습 도우미 안내</h3>
        <p>
            AIVLE School 교육 자료를 기반으로
            질문에 대한 답변을 제공합니다.
        </p>
        <b style="color:#193653;font-size:13px;">
            RAG 기반 문서 검색
        </b>
        <p>관련 문서를 검색하고 답변을 생성합니다.</p>
    </div>

    <div style="font-size:18px;font-weight:900;
                color:#193653;margin:20px 0 12px;">
        💡 맞춤 추천 질문
    </div>

    """, unsafe_allow_html=True)

    # 추천 질문: 누르면 바로 챗봇에 질문
    for i, question in enumerate(get_recommended_questions()):
        if st.button(question, key=f"rq_{i}", use_container_width=True):
            st.session_state.chat_pending = question

    right_chatbot()

    st.markdown("""
    <div class="right-card" style="margin-top:25px;">
        <h3>🕒 Recent Activity</h3>
        <p>최근 대화 기록</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.messages:

        for msg in reversed(st.session_state.messages[-4:]):
            if isinstance(msg, HumanMessage):
                st.caption("🙋 " + msg.content[:35])

    else:
        st.caption("아직 대화 기록이 없습니다.")


# =========================================================
# 8. 홈 화면
# =========================================================

def page_home():

    st.markdown(f"""
    <div class="page-title">Hello, {escape(st.session_state.user_profile.get("name") or "AIVLE")}! 👋</div>
    <div class="page-subtitle">{escape(st.session_state.user_profile.get("stage", "에이블러(교육생)"))} · {escape(st.session_state.user_profile.get("track", "DX 컨설턴트 트랙"))}에 맞춘 학습을 시작해 보세요.</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("""
        <div class="welcome-card welcome-one">
            <h3>Welcome back! ✨</h3>
            <p>
                AIVLE School 학습 도우미입니다.<br>
                교육 과정, 강의 내용, 미니프로젝트 등<br>
                궁금한 내용을 자유롭게 질문해 보세요!
            </p>
            <div style="text-align:right;font-size:25px;">💻 💬</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="welcome-card welcome-two">
            <h3>Learning Guide 📚</h3>
            <p>
                여러분의 학습을 도와드릴게요.<br>
                문서 기반 RAG 시스템을 활용하여<br>
                필요한 정보를 빠르게 찾아보세요.
            </p>
            <div style="text-align:right;font-size:25px;">💡 📚</div>
        </div>
        """, unsafe_allow_html=True)

    section_title(1, "Quick Menu")

    q1, q2, q3 = st.columns(3)

    with q1:
        info_card("📚", "학습 자료", "교육 문서 기반 질문")

    with q2:
        info_card("🧠", "학습 Q&A", "AI 문서 검색 및 답변")

    with q3:
        info_card("🚀", "미니프로젝트", "프로젝트 정보 확인")

    st.markdown("<br>", unsafe_allow_html=True)

    section_title(2, "출결 & 일정 관리")

    st.markdown("""
    <p class="small-text">
        교육 일정과 개인 학습 일정을 확인하고 출결 상태를 기록해 보세요.
    </p>
    """, unsafe_allow_html=True)

    # 달력 월 이동
    prev_col, month_col, next_col = st.columns([1, 5, 1])

    with prev_col:
        if st.button("‹", key="prev_month"):
            st.session_state.calendar_month -= 1

            if st.session_state.calendar_month < 1:
                st.session_state.calendar_month = 12
                st.session_state.calendar_year -= 1

            st.rerun()

    with month_col:
        st.markdown(
            f"""
            <h3 style="text-align:center;color:#193653;">
                {st.session_state.calendar_year}년
                {st.session_state.calendar_month}월
            </h3>
            """,
            unsafe_allow_html=True
        )

    with next_col:
        if st.button("›", key="next_month"):
            st.session_state.calendar_month += 1

            if st.session_state.calendar_month > 12:
                st.session_state.calendar_month = 1
                st.session_state.calendar_year += 1

            st.rerun()

    st.markdown('<div class="calendar-box">', unsafe_allow_html=True)

    cal = calendar.monthcalendar(
        st.session_state.calendar_year,
        st.session_state.calendar_month
    )

    weekdays = ["일", "월", "화", "수", "목", "금", "토"]

    html = '<div class="calendar-grid">'

    for day in weekdays:
        html += f'<div class="calendar-head">{day}</div>'

    for week in cal:
        for day in week:

            if day == 0:
                html += '<div class="calendar-day" style="border:none;"></div>'

            else:
                today_class = ""

                if (
                    day == date.today().day
                    and st.session_state.calendar_month == date.today().month
                    and st.session_state.calendar_year == date.today().year
                ):
                    today_class = "today"

                html += f'<div class="calendar-day {today_class}">{day}</div>'

    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("📌 오늘 날짜는 강조 표시됩니다.")


# =========================================================
# 9. 소개 페이지
# =========================================================

def page_intro():

    st.markdown("""
    <div class="page-title">👋 소개</div>

    <div class="hero">
        <div>
            <h1>AIVLE School 학습 도우미 소개</h1>
            <h3>AIVLE School 교육생을 위한 맞춤형 학습 지원 서비스</h3>
            <p>
                KT AIVLE School의 교육 과정, 학습 자료, 프로젝트, 취업 준비까지<br>
                교육생이 궁금한 내용을 빠르고 정확하게 안내하는<br>
                RAG 기반 AI 학습 도우미입니다.
            </p>
        </div>
        <div class="hero-icon">🤖💡</div>
    </div>
    """, unsafe_allow_html=True)

    section_title(1, "AIVLE School이란?")

    st.markdown("""
    <div class="info-card">
        <div style="display:flex;align-items:center;gap:20px;">
            <div style="font-size:36px;font-weight:900;color:#333;">
                kt
            </div>
            <div style="width:1px;height:55px;background:#dce8ee;"></div>
            <div style="flex:1;">
                <h3>KT AIVLE School</h3>
                <p style="font-size:14px;">
                    KT가 운영하는 AI·DX 실무형 인재 양성 프로그램으로,<br>
                    기업 현장에서 요구하는 디지털 역량을 갖춘 인재를 육성합니다.
                </p>
            </div>
            <div style="background:#e8f8f9;border-radius:15px;padding:15px;">
                <b style="color:#17aeb7;">⚙️ AI</b><br>
                <small>인공지능 역량</small>
            </div>
            <div style="background:#edf5ff;border-radius:15px;padding:15px;">
                <b style="color:#5089e8;">◈ DX</b><br>
                <small>디지털 전환 역량</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:

        section_title(2, "교육 트랙 소개")

        st.markdown("""
        <div class="track-card track-ai">
            <div style="font-size:35px;">💻</div>
            <h2>AI 개발자</h2>
            <p>
                AI 모델과 서비스 개발을 통해<br>
                실무형 AI 서비스 구현 역량을 강화
            </p>
        </div>
        <br>
        <div class="track-card track-dx">
            <div style="font-size:35px;">📊</div>
            <h2>DX 컨설턴트</h2>
            <p>
                데이터 분석과 AI를 활용한<br>
                기업 문제 해결 및 디지털 전환 기획
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        section_title(3, "주요 교육 분야")

        subjects = [
            ("🐍", "프로그래밍·데이터 분석", "Python, 데이터 처리 및 시각화"),
            ("🧠", "머신러닝·딥러닝", "AI 모델 학습과 평가"),
            ("🤖", "생성형 AI", "LLM 및 프롬프트 엔지니어링"),
            ("☁️", "클라우드", "클라우드 인프라와 서비스")
        ]

        for row in range(2):

            cols = st.columns(2)

            for col_idx in range(2):
                icon, title, desc = subjects[row * 2 + col_idx]

                with cols[col_idx]:
                    subject_card(icon, title, desc)

    section_title(4, "교육 진행 과정")

    steps = [
        ("📖", "01", "이론 및 실습", "기초 기술과 핵심 개념 학습"),
        ("💻", "02", "미니 프로젝트", "학습한 기술을 활용한 문제 해결 경험"),
        ("🚀", "03", "빅 프로젝트", "실제 문제 발굴부터 결과물 구현 및 발표")
    ]

    cols = st.columns(3)

    for i, (icon, number, title, desc) in enumerate(steps):

        with cols[i]:
            st.markdown(f"""
            <div class="roadmap-card">
                <div class="roadmap-icon">{icon}</div>
                <span class="step">STEP {number}</span>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    section_title(5, "교육을 통해 기르는 핵심 역량")

    skills = [
        ("💡", "문제 해결 역량", "실무 문제를 정의하고 데이터·AI로 해결"),
        ("📊", "데이터 활용 역량", "데이터 수집·분석·해석을 통한 인사이트 도출"),
        ("👥", "협업·소통 역량", "팀 프로젝트를 통한 효과적인 협업과 의사소통"),
        ("🎯", "프로젝트 수행 역량", "기획부터 구현, 발표까지 프로젝트 경험")
    ]

    cols = st.columns(4)

    for i, (icon, title, desc) in enumerate(skills):

        with cols[i]:
            info_card(icon, title, desc)

    st.markdown("""
    <div class="bottom-banner">
        <div style="display:flex;align-items:center;gap:25px;">
            <div style="font-size:40px;">◢</div>
            <div>
                <h2>Your Next Step with AIVLE School!</h2>
                <p>AIVLE School과 함께, AI와 DX로 더 큰 성장의 기회를 만들어보세요.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 10. 교육과정 페이지
# =========================================================

def page_education():

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div class="page-title">📚 교육과정</div>
            <div class="page-subtitle">AIVLE School Curriculum</div>
        </div>
        <div style="
            background:#34b58e;color:white;padding:7px 18px;
            border-radius:20px;font-size:13px;font-weight:900;">
            CURRICULUM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div>
            <h1>배움에서 실무까지, 성장의 여정 🚀</h1>
            <p style="font-size:16px;">
                체계적인 교육과 프로젝트를 통해<br>
                AI·DX 실무 역량을 키워보세요.
            </p>
        </div>
        <div class="hero-icon">👩🏻‍💻📊</div>
    </div>
    """, unsafe_allow_html=True)

    # 교육 트랙 선택
    section_title(1, "교육 트랙 선택")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="track-card track-ai" style="min-height:140px;">
            <div style="display:flex;align-items:center;gap:18px;">
                <div style="font-size:45px;">💻</div>
                <div>
                    <h2>AI 개발자 트랙</h2>
                    <p>AI 모델 개발 및 서비스 구현 중심 교육</p>
                    <span style="
                        background:white;color:#2399ac;
                        border-radius:20px;padding:5px 14px;
                        font-size:12px;font-weight:800;">
                        AI Developer
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="track-card track-dx" style="min-height:140px;">
            <div style="display:flex;align-items:center;gap:18px;">
                <div style="font-size:45px;">📊</div>
                <div>
                    <h2>DX 컨설턴트 트랙</h2>
                    <p>데이터 분석과 AI를 활용한 비즈니스 문제 해결</p>
                    <span style="
                        background:white;color:#518be0;
                        border-radius:20px;padding:5px 14px;
                        font-size:12px;font-weight:800;">
                        DX Consultant
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 교육 로드맵
    section_title(2, "교육 진행 로드맵")

    roadmap = [
        ("💻", "STEP 01", "기초 역량 학습", "Python, SQL,<br>데이터 분석 등"),
        ("⚙️", "STEP 02", "AI·DX 심화 학습", "머신러닝, 딥러닝,<br>생성형 AI, 클라우드"),
        ("👥", "STEP 03", "미니 프로젝트", "학습한 기술을 활용한<br>실습 및 팀 프로젝트"),
        ("🚀", "STEP 04", "빅 프로젝트", "실제 비즈니스 문제 해결 및<br>서비스 구현")
    ]

    cols = st.columns(4)

    for i, (icon, step, title, desc) in enumerate(roadmap):

        with cols[i]:

            st.markdown(f"""
            <div class="roadmap-card" style="min-height:150px;">
                <div class="roadmap-icon">{icon}</div>
                <span class="step">{step}</span>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

            if i != 3:
                st.markdown(
                    "<div style='text-align:center;color:#4e91be;"
                    "font-size:23px;font-weight:900;'>→</div>",
                    unsafe_allow_html=True
                )

    # 주요 학습 분야
    left, right = st.columns([1.5, 1])

    with left:

        section_title(3, "주요 학습 분야")

        subjects = [
            ("🐍", "Python", "프로그래밍 기초 및 데이터 처리"),
            ("📊", "데이터 분석", "데이터 전처리·분석·시각화"),
            ("🗄️", "SQL", "데이터베이스·데이터 조회"),
            ("🤖", "머신러닝", "AI 모델 학습과 평가"),
            ("🧠", "딥러닝", "신경망 기반 모델 학습"),
            ("✨", "생성형 AI", "LLM, 프롬프트 엔지니어링, RAG 활용")
        ]

        for row in range(2):

            cols = st.columns(3)

            for j in range(3):

                icon, title, desc = subjects[row * 3 + j]

                with cols[j]:
                    subject_card(icon, title, desc)

    with right:

        section_title(4, "프로젝트 학습")

        st.markdown("""
        <div class="project-card">
            <div style="display:flex;gap:15px;">
                <div style="font-size:35px;">📋</div>
                <div>
                    <h3>미니 프로젝트</h3>
                    <p>
                        ✅ 학습 내용을 실제로 적용<br>
                        ✅ 개인 및 팀 기반 프로젝트 수행<br>
                        ✅ 결과 발표 및 피드백
                    </p>
                </div>
            </div>
        </div>

        <div class="project-card">
            <div style="display:flex;gap:15px;">
                <div style="font-size:35px;">🚀</div>
                <div>
                    <h3>빅 프로젝트</h3>
                    <p>
                        ✅ 실제 비즈니스 문제 발굴<br>
                        ✅ AI 기반 서비스 기획 및 구현<br>
                        ✅ 최종 결과물 발표 및 성과 공유
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 하단 배너
    st.markdown("""
    <div class="bottom-banner">
        <div style="display:flex;align-items:center;gap:25px;">
            <div style="font-size:45px;">◢</div>
            <div>
                <h2>Your Future Starts Here!</h2>
                <p>AIVLE School과 함께 실무형 인재로 성장해 보세요.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)




# =========================================================
# 11. 취업과정 페이지
# =========================================================

def page_career():

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div class="page-title">💼 취업과정</div>
            <div class="page-subtitle">Career &amp; Job Preparation</div>
        </div>
        <div style="
            background:linear-gradient(90deg,#20b8c2,#159eae);
            color:white;padding:7px 18px;border-radius:20px;
            font-size:13px;font-weight:900;">
            JOB PREPARATION
        </div>
    </div>

    <div class="hero" style="min-height:145px;">
        <div>
            <h1>AIVLE School 취업 준비 가이드</h1>
            <h3>학습부터 포트폴리오, 면접까지<br>
                실무형 취업 역량을 완성해 보세요!</h3>
        </div>
        <div class="hero-icon">🧑🏻‍💼📈</div>
    </div>
    """, unsafe_allow_html=True)

    # 01. 취업 준비 로드맵
    section_title(1, "취업 준비 로드맵")

    roadmap = [
        ("🎯", "직무 탐색", "관심 직무 이해 및<br>요구 역량 파악"),
        ("📚", "역량 강화", "교육 과정 기반<br>핵심 역량 확보"),
        ("💻", "포트폴리오", "프로젝트 정리 및<br>성과 도출"),
        ("📄", "서류 준비", "이력서·자기소개서<br>작성"),
        ("👥", "면접 대비", "기술·인성 면접 및<br>실전 연습")
    ]
    cols = st.columns(5)
    for i, (icon, title, desc) in enumerate(roadmap):
        with cols[i]:
            st.markdown(f"""
            <div class="roadmap-card" style="
                min-height:130px;padding:13px 7px;
                background:linear-gradient(145deg,#e4f9f7,#edf4ff);">
                <div class="roadmap-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if i < 4:
                st.markdown(
                    "<div style='text-align:center;color:#249eb0;"
                    "font-size:20px;font-weight:900;'>→</div>",
                    unsafe_allow_html=True
                )

    # 02. 취업 지원 프로그램 / 03. 직무별 정보
    left, right = st.columns([1, 1.35], gap="medium")

    with left:
        section_title(2, "주요 취업 지원 프로그램")
        programs = [
            ("🎓", "취업 특강", "현직자 직무 특강 및<br>채용 트렌드 안내"),
            ("📄", "자소서 클리닉", "자기소개서 작성법과<br>1:1 피드백"),
            ("👥", "모의 면접", "실전형 면접 연습 및<br>개별 피드백"),
            ("🏢", "채용 연계", "KT 및 협력사<br>채용 정보 제공")
        ]
        for row in range(2):
            cols = st.columns(2)
            for j in range(2):
                icon, title, desc = programs[row * 2 + j]
                with cols[j]:
                    st.markdown(f"""
                    <div class="info-card" style="
                        min-height:112px;padding:13px 8px;margin-bottom:10px;">
                        <div class="card-icon">{icon}</div>
                        <h3 style="text-align:center;font-size:14px;">{title}</h3>
                        <p style="text-align:center;">{desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

    with right:
        section_title(3, "직무별 정보")
        tab_dx, tab_ai = st.tabs(["DX 컨설턴트", "AI 개발자"])

        with tab_dx:
            st.markdown("""
            <div class="project-card" style="min-height:220px;">
                <div style="display:flex;gap:15px;align-items:center;">
                    <div style="font-size:38px;">📊</div>
                    <div>
                        <h3>DX 컨설턴트 (DX Consultant)</h3>
                        <p>데이터와 AI를 활용해 기업의 비즈니스 문제를 분석하고,
                        디지털 전환을 기획·제안하는 직무입니다.</p>
                    </div>
                </div>
                <hr>
                <p><b style="color:#16aeb8;">✓ 주요 업무</b>　데이터 분석, DX 과제 기획, 솔루션 제안</p>
                <p><b style="color:#16aeb8;">✓ 요구 역량</b>　데이터 분석력, 비즈니스 이해, 커뮤니케이션</p>
                <p><b style="color:#16aeb8;">✓ 연관 직무</b>　데이터 분석가, IT기획, PM, BI 개발자 등</p>
            </div>
            """, unsafe_allow_html=True)

        with tab_ai:
            st.markdown("""
            <div class="project-card" style="min-height:220px;">
                <div style="display:flex;gap:15px;align-items:center;">
                    <div style="font-size:38px;">🤖</div>
                    <div>
                        <h3>AI 개발자 (AI Developer)</h3>
                        <p>AI 모델과 서비스를 개발하고 데이터 기반 문제 해결을
                        위한 인공지능 기능을 구현하는 직무입니다.</p>
                    </div>
                </div>
                <hr>
                <p><b style="color:#16aeb8;">✓ 주요 업무</b>　AI 모델 개발, 데이터 전처리, 서비스 구현</p>
                <p><b style="color:#16aeb8;">✓ 요구 역량</b>　Python, 머신러닝·딥러닝, 알고리즘</p>
                <p><b style="color:#16aeb8;">✓ 연관 직무</b>　AI 엔지니어, ML 엔지니어, 데이터 엔지니어 등</p>
            </div>
            """, unsafe_allow_html=True)

    # 04. 취업 준비 자료실 / 05. 추천 학습 콘텐츠
    left, right = st.columns([1.25, 1], gap="medium")

    with left:
        section_title(4, "취업 준비 자료실")
        resources = [
            ("📄", "이력서 템플릿", "경험과 역량을 구조화해 정리"),
            ("🖊️", "자기소개서 가이드", "문항별 작성 전략과 사례"),
            ("💬", "면접 질문 모음", "직무·인성 면접 예상 질문"),
            ("📁", "포트폴리오 예시", "프로젝트 구성 및 성과 표현")
        ]
        cols = st.columns(4)
        for i, (icon, title, desc) in enumerate(resources):
            with cols[i]:
                st.markdown(f"""
                <div class="info-card" style="
                    min-height:115px;padding:13px 6px;">
                    <div class="card-icon">{icon}</div>
                    <h3 style="text-align:center;font-size:13px;">{title}</h3>
                    <p style="text-align:center;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    with right:
        section_title(5, "추천 학습 콘텐츠")
        st.markdown("""
        <div class="project-card" style="
            background:linear-gradient(120deg,#f0fbf9,#f5f9ff);">
            <p style="font-size:14px;line-height:2.0;color:#29465f;">
                <b style="color:#18b58a;">✓</b>　프로젝트 경험 정리 및 성과 수치화 방법<br>
                <b style="color:#18b58a;">✓</b>　데이터·AI 직무 면접 대비 예상 질문<br>
                <b style="color:#18b58a;">✓</b>　포트폴리오 구성 및 발표 팁<br>
                <b style="color:#18b58a;">✓</b>　기업별 채용 공고 및 자격 요건 정리
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bottom-banner">
        <div style="display:flex;align-items:center;gap:25px;">
            <div style="font-size:42px;">◢</div>
            <div>
                <h2>Next Career with AIVLE School!</h2>
                <p>AIVLE School과 함께, 실무 역량을 취업으로 연결하세요.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)




# =========================================================
# 12. 포트폴리오 페이지
# =========================================================

# 포트폴리오 피드백: 텍스트 추출 + 피드백 체인
def read_portfolio_text(uploaded_file):
    """PDF는 PyMuPDF(백서 로더와 같은 패키지)로 먼저 읽고, 나머지는 extract_document_text 사용"""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".pdf":
        try:
            import fitz  # langchain-pymupdf4llm 설치 시 함께 설치됨
            pdf = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            return "\n\n".join(f"[{i}페이지]\n{p.get_text()}" for i, p in enumerate(pdf, start=1))
        except ImportError:
            pass
    return extract_document_text(uploaded_file)


PORTFOLIO_FEEDBACK_SYS = """
당신은 KT AIVLE School 취업 컨설턴트입니다. 교육생의 포트폴리오를 읽고 지원 직무 관점에서 피드백합니다.

[지원자 프로필]
{profile}

[백서의 포트폴리오·취업 가이드 (근거)]
{guide}

아래 형식 그대로, 포트폴리오 원문에 있는 문장을 근거로 짧게 인용하며 작성하세요. 원문에 없는 성과를 지어내지 마세요.

## 총평
(2~3문장, 10점 만점 점수 포함)

## 항목별 평가
| 항목 | 점수(5점) | 근거 | 개선 제안 |
|---|---|---|---|
| 프로젝트 배경 및 문제 정의 | | | |
| 수행 내용 및 문제 해결 과정 | | | |
| 주요 성과 및 결과 (정량·정성) | | | |
| 배운 점 및 개선할 점 | | | |
| 구성 · 가독성 · 디자인 | | | |

## 지원 직무 관점 강점 3가지

## 꼭 보완할 점 3가지 (고쳐 쓴 예시 문장 포함)
"""


def portfolio_feedback_stream(pf_text, file_name):
    p = st.session_state.get("user_profile", {})
    profile = (f"- 단계: {p.get('stage')}\n- 트랙: {p.get('track')}\n"
               f"- 희망 기업: {p.get('company') or '미입력'}\n- 희망 직무: {p.get('job') or '미입력'}")
    # 백서에서 포트폴리오 작성 기준을 RAG로 가져와 근거로 사용
    try:
        guide_docs = hybrid_retrieve(f"{p.get('track', '')} 포트폴리오 작성 방법 구성 성과 표현", k=4)
        guide = format_docs(guide_docs)
    except Exception:
        guide = "백서 검색 결과 없음"
    prompt = ChatPromptTemplate.from_messages([
        ("system", PORTFOLIO_FEEDBACK_SYS),
        ("human", "파일명: {name}\n\n[포트폴리오 원문]\n{text}")
    ])
    chain = prompt | get_llm(True) | StrOutputParser()
    return chain.stream({"profile": profile, "guide": guide, "name": file_name, "text": pf_text[:6000]})


def page_portfolio():

    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div class="page-title">📁 포트폴리오</div>
            <div class="page-subtitle">Portfolio</div>
        </div>
        <div style="
            background:linear-gradient(90deg,#20b8b8,#159eae);
            color:white;padding:8px 18px;border-radius:22px;
            font-size:13px;font-weight:900;">
            MY PORTFOLIO
        </div>
    </div>

    <div class="hero" style="min-height:145px;">
        <div>
            <h1>프로젝트 경험을 가치 있는 포트폴리오로! ✨</h1>
            <h3 style="color:#536d87;">
                작성부터 AI 피드백, 선배 사례까지<br>
                AIVLE School이 포트폴리오 완성을 도와드립니다.
            </h3>
        </div>
        <div class="hero-icon">👩🏻‍💻📚</div>
    </div>
    """, unsafe_allow_html=True)

    # 상단 바로가기 카드
    quick_actions = [
        ("✏️", "포트폴리오 작성", "템플릿으로 쉽게 작성", "작성하기"),
        ("✨", "AI 피드백 받기", "완성도 분석 및 개선 제안", "피드백"),
        ("📂", "선배 포트폴리오", "우수 사례 확인 및 참고", "사례"),
        ("📖", "작성 가이드", "구성·작성법·예시 제공", "가이드")
    ]
    action_cols = st.columns(4)
    for idx, (icon, title, desc, action) in enumerate(quick_actions):
        with action_cols[idx]:
            st.markdown(f"""
            <div class="info-card" style="
                min-height:108px;text-align:center;padding:13px 8px;
                border:1px solid #e6f0f4;">
                <div style="font-size:34px;margin-bottom:3px;">{icon}</div>
                <h3 style="font-size:16px;margin:3px 0 5px;">{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("바로가기  ›", key=f"portfolio_quick_{idx}",
                         use_container_width=True):
                st.session_state.portfolio_section = action
                st.rerun()

    st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

    # 초기 상태
    if "portfolio_section" not in st.session_state:
        st.session_state.portfolio_section = "작성하기"
    if "portfolio_items" not in st.session_state:
        st.session_state.portfolio_items = []

    # 작성 영역 / AI 피드백 영역
    left_col, right_col = st.columns([1, 1.03], gap="medium")

    with left_col:
        section_title(1, "포트폴리오 작성하기")

        write_tab, saved_tab = st.tabs(["✏️ 새로 작성하기", "📁 내 포트폴리오"])

        with write_tab:
            st.markdown("""
            <div style="
                background:#eef5fc;border-radius:13px;padding:9px 12px;
                color:#294763;font-weight:800;text-align:center;
                margin-bottom:12px;">
                프로젝트 경험을 정리하고 나만의 포트폴리오를 만들어 보세요.
            </div>
            """, unsafe_allow_html=True)

            template = st.radio(
                "작성 템플릿 선택",
                ["프로젝트 중심 템플릿", "취업용 종합 템플릿", "데이터·AI 특화 템플릿"],
                horizontal=False,
                key="portfolio_template"
            )

            template_desc = {
                "프로젝트 중심 템플릿": "프로젝트별 문제 정의, 수행 과정, 결과와 성과를 중심으로 작성합니다.",
                "취업용 종합 템플릿": "자기소개, 핵심 역량, 프로젝트, 자격증 및 경험을 종합적으로 구성합니다.",
                "데이터·AI 특화 템플릿": "데이터 수집·전처리, 분석·모델링, 평가 지표와 비즈니스 적용을 강조합니다."
            }
            st.info(template_desc[template])

            with st.form("portfolio_create_form", clear_on_submit=False):
                project_title = st.text_input(
                    "프로젝트명 *",
                    placeholder="예: 공공데이터 기반 지역 주차 문제 분석"
                )
                c1, c2 = st.columns(2)
                with c1:
                    project_role = st.text_input(
                        "담당 역할",
                        placeholder="예: 데이터 분석 및 시각화"
                    )
                with c2:
                    project_period = st.text_input(
                        "진행 기간",
                        placeholder="예: 2026.10 ~ 2026.11"
                    )

                tech_stack = st.text_input(
                    "사용 기술",
                    placeholder="예: Python, Pandas, Streamlit, SQL"
                )
                problem = st.text_area(
                    "프로젝트 배경 및 문제 정의",
                    placeholder="어떤 문제를 해결하기 위해 프로젝트를 진행했나요?",
                    height=85
                )
                process = st.text_area(
                    "수행 내용 및 문제 해결 과정",
                    placeholder="데이터 수집, 분석, 모델링, 서비스 구현 등 수행 과정을 작성하세요.",
                    height=100
                )
                result = st.text_area(
                    "주요 성과 및 결과",
                    placeholder="정량적·정성적 성과와 프로젝트 결과를 작성하세요.",
                    height=85
                )
                lesson = st.text_area(
                    "배운 점 및 개선할 점",
                    placeholder="프로젝트를 통해 배운 점과 향후 개선 방향을 작성하세요.",
                    height=75
                )

                submitted = st.form_submit_button(
                    "✏️ 포트폴리오 저장하기",
                    use_container_width=True,
                    type="primary"
                )

                if submitted:
                    if not project_title.strip():
                        st.warning("프로젝트명을 입력해 주세요.")
                    else:
                        st.session_state.portfolio_items.append({
                            "title": project_title.strip(),
                            "role": project_role.strip(),
                            "period": project_period.strip(),
                            "tech": tech_stack.strip(),
                            "problem": problem.strip(),
                            "process": process.strip(),
                            "result": result.strip(),
                            "lesson": lesson.strip(),
                            "template": template
                        })
                        st.success("포트폴리오가 내 포트폴리오에 저장되었습니다.")

        with saved_tab:
            saved_items = st.session_state.portfolio_items
            if saved_items:
                st.markdown(f"**저장된 포트폴리오 {len(saved_items)}개**")
                for idx, item in enumerate(saved_items):
                    with st.expander(f"📁 {item['title']}", expanded=False):
                        st.caption(
                            f"{item.get('role') or '역할 미입력'} · "
                            f"{item.get('period') or '기간 미입력'}"
                        )
                        st.markdown(f"**사용 기술**  \n{item.get('tech') or '-'}")
                        st.markdown(f"**문제 정의**  \n{item.get('problem') or '-'}")
                        st.markdown(f"**수행 과정**  \n{item.get('process') or '-'}")
                        st.markdown(f"**성과**  \n{item.get('result') or '-'}")
                        st.markdown(f"**배운 점**  \n{item.get('lesson') or '-'}")
                        if st.button("삭제", key=f"delete_portfolio_{idx}"):
                            st.session_state.portfolio_items.pop(idx)
                            st.rerun()
            else:
                st.markdown("""
                <div class="project-card" style="text-align:center;padding:35px 12px;">
                    <div style="font-size:42px;">📂</div>
                    <h3>아직 저장된 포트폴리오가 없습니다.</h3>
                    <p>새로 작성하기 탭에서 프로젝트 경험을 등록해 보세요.</p>
                </div>
                """, unsafe_allow_html=True)

    with right_col:
        section_title(2, "AI 포트폴리오 피드백")

        uploaded_portfolio = st.file_uploader(
            "포트폴리오 파일 업로드",
            type=["pdf", "ppt", "pptx", "docx"],
            accept_multiple_files=False,
            key="portfolio_feedback_upload",
            help="PDF, PPT, PPTX, DOCX 파일을 업로드할 수 있습니다."
        )

        if uploaded_portfolio:
            st.success(f"업로드 파일: {uploaded_portfolio.name}")
            st.caption(f"파일 크기: {uploaded_portfolio.size / 1024:.1f} KB")

        st.markdown("""
        <div class="project-card" style="
            background:linear-gradient(120deg,#e9fbf9,#e9f5ff);
            padding:17px;">
            <h3>✨ AI가 이런 항목을 중심으로 피드백해요!</h3>
            <p style="font-size:14px;line-height:2.1;color:#29465f;">
                <b style="color:#18b58a;">✓</b> 프로젝트 설명의 구체성과 논리 흐름 분석<br>
                <b style="color:#18b58a;">✓</b> 성과(정량·정성) 표현의 적절성 평가<br>
                <b style="color:#18b58a;">✓</b> 지원 직무 관점의 강점과 보완점 제안<br>
                <b style="color:#18b58a;">✓</b> 구성, 가독성, 디자인 개선 가이드 제공
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ AI 피드백 받기", key="portfolio_feedback_btn",
                     use_container_width=True, type="primary"):
            if uploaded_portfolio is None:
                st.warning("피드백을 받을 포트폴리오 파일을 먼저 업로드해 주세요.")
            else:
                try:
                    with st.spinner("포트폴리오 내용을 읽는 중..."):
                        pf_text = read_portfolio_text(uploaded_portfolio)
                    if len(pf_text.strip()) < 50:
                        st.warning("파일에서 글자를 거의 찾지 못했습니다. 이미지로만 된 PDF라면 텍스트가 들어간 PDF로 다시 저장해 주세요.")
                    else:
                        st.caption(f"추출한 글자 수: {len(pf_text):,}자 (앞 6,000자 기준으로 분석)")
                        st.session_state.portfolio_feedback = st.write_stream(
                            portfolio_feedback_stream(pf_text, uploaded_portfolio.name))
                except Exception as e:
                    st.error(f"피드백 생성 중 오류가 발생했습니다: {e}")

        # 이전 피드백 다시 보기 + 다운로드
        if st.session_state.get("portfolio_feedback"):
            st.download_button("⬇️ 피드백 다운로드 (.md)", st.session_state.portfolio_feedback,
                               file_name="portfolio_feedback.md", key="pf_feedback_down",
                               use_container_width=True)

    # 선배 포트폴리오 사례
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    examples_col, guide_col = st.columns([1.8, 1], gap="medium")

    with examples_col:
        section_title(3, "선배들의 포트폴리오")

        category = st.radio(
            "직무별 사례",
            ["전체", "DX 컨설턴트", "AI 개발자"],
            horizontal=True,
            key="portfolio_example_category"
        )

        examples = [
            {
                "track": "DX 컨설턴트",
                "title": "VOC 분석 기반 AI 고객상담 지원 서비스",
                "desc": "고객 VOC를 분석해 상담 업무 효율화 방안을 제안한 프로젝트",
                "tags": ["데이터분석", "RAG", "서비스기획"],
                "cover": "🌊"
            },
            {
                "track": "AI 개발자",
                "title": "이미지 분류 기반 품질 검사 시스템",
                "desc": "이미지 데이터를 활용한 품질 검사 자동화 프로젝트",
                "tags": ["딥러닝", "모델링", "Flask"],
                "cover": "🧠"
            },
            {
                "track": "DX 컨설턴트",
                "title": "공공데이터 분석을 통한 지역 복지 서비스 개선",
                "desc": "공공데이터 분석을 통한 지역 현안 및 개선 방향 도출",
                "tags": ["데이터분석", "시각화", "정책제안"],
                "cover": "📊"
            },
            {
                "track": "AI 개발자",
                "title": "LLM 기반 문서 질의응답 시스템",
                "desc": "문서 검색 증강 생성 기반 질의응답 서비스 구현",
                "tags": ["LLM", "RAG", "LangChain"],
                "cover": "🤖"
            }
        ]

        filtered = [
            x for x in examples
            if category == "전체" or x["track"] == category
        ]
        card_cols = st.columns(4 if category == "전체" else 2)
        for idx, item in enumerate(filtered):
            with card_cols[idx % len(card_cols)]:
                st.markdown(f"""
                <div class="info-card" style="
                    padding:10px 9px;min-height:190px;margin-bottom:8px;">
                    <div style="
                        height:64px;border-radius:9px;
                        background:linear-gradient(120deg,#0e7589,#1ca8ad);
                        display:flex;align-items:center;justify-content:center;
                        font-size:34px;color:white;">
                        {item['cover']}
                    </div>
                    <div style="
                        display:inline-block;background:#e8f4ff;color:#2677bb;
                        border-radius:6px;padding:2px 6px;font-size:10px;
                        font-weight:800;margin-top:7px;">
                        {item['track']}
                    </div>
                    <h3 style="font-size:13px;margin:5px 0;">{item['title']}</h3>
                    <p style="font-size:11px;">{item['desc']}</p>
                    <div style="margin-top:6px;">
                        {''.join(f"<span style='display:inline-block;background:#edf3fa;color:#526d87;border-radius:9px;padding:3px 6px;margin:2px;font-size:9px;'>{tag}</span>" for tag in item['tags'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with guide_col:
        section_title(4, "포트폴리오 작성 가이드")
        guide_items = [
            ("☷", "좋은 포트폴리오의 구성요소",
             "지원 직무와 연결되는 핵심 역량, 프로젝트, 성과를 명확하게 구성하세요."),
            ("</>", "프로젝트 작성 예시",
             "문제 정의 → 분석 및 수행 과정 → 결과 → 배운 점 순서로 작성하면 좋아요."),
            ("📊", "성과 지표 표현 방법",
             "가능하면 건수, 비율, 시간 단축 등 객관적인 수치로 성과를 표현하세요."),
            ("🎨", "디자인·가독성 팁",
             "한 페이지에 하나의 핵심 메시지를 담고 글꼴과 색상을 일관되게 사용하세요."),
            ("❔", "자주 묻는 질문 (FAQ)",
             "팀 프로젝트에서 본인의 역할과 기여도를 구체적으로 구분해 작성하세요.")
        ]
        for idx, (icon, title, desc) in enumerate(guide_items):
            with st.expander(f"{icon}  {title}"):
                st.write(desc)

    st.markdown("""
    <div class="bottom-banner">
        <div style="display:flex;align-items:center;gap:22px;">
            <div style="font-size:42px;">📁</div>
            <div>
                <h2>Build Your Future Portfolio!</h2>
                <p>프로젝트 경험을 나만의 경쟁력 있는 포트폴리오로 완성해 보세요.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



# =========================================================
# 11. 파일 불러오기 페이지
# =========================================================

def extract_document_text(uploaded_file):
    """업로드한 PDF/TXT/DOCX/PPT/PPTX 파일에서 텍스트를 추출합니다."""
    file_name = uploaded_file.name
    extension = os.path.splitext(file_name)[1].lower()
    file_bytes = uploaded_file.getvalue()

    if extension == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[{page_number}페이지]\n{page_text}")
        return "\n\n".join(pages)

    if extension == ".txt":
        try:
            return file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            return file_bytes.decode("cp949", errors="replace")

    if extension == ".docx":
        from docx import Document as DocxDocument
        document = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())

    if extension in [".pptx", ".ppt"]:
        from pptx import Presentation
        presentation = Presentation(io.BytesIO(file_bytes))
        slide_texts = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    texts.append(shape.text.strip())
            if texts:
                slide_texts.append(
                    f"[슬라이드 {slide_number}]\n" + "\n".join(texts)
                )
        return "\n\n".join(slide_texts)

    raise ValueError("지원하지 않는 파일 형식입니다.")


def add_document_to_vectorstore(uploaded_file, category, description):
    """문서를 청크로 분할하고 ChromaDB에 저장합니다."""
    text = extract_document_text(uploaded_file)

    if not text or not text.strip():
        raise ValueError(
            "문서에서 텍스트를 추출하지 못했습니다. "
            "스캔된 PDF라면 OCR 처리가 필요할 수 있습니다."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    base_document = Document(
        page_content=text,
        metadata={
            "source": uploaded_file.name,
            "category": category,
            "description": description or ""
        }
    )
    chunks = splitter.split_documents([base_document])

    ids = [str(uuid.uuid4()) for _ in chunks]

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        persist_directory=CHROMA_DIR
    )
    vectorstore.add_documents(chunks, ids=ids)

    return {
        "name": uploaded_file.name,
        "category": category,
        "description": description,
        "size": len(uploaded_file.getvalue()),
        "uploaded_at": datetime.now().strftime("%Y.%m.%d %H:%M"),
        "status": "완료",
        "chunk_count": len(chunks),
        "ids": ids,
        "preview": text[:5000]
    }


def page_file_upload():

    st.markdown("""
    <div class="page-title">📂 파일 불러오기</div>
    <div class="page-subtitle">Document Upload · 나만의 학습 자료를 등록하고 AI와 대화해 보세요.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero" style="min-height:140px;">
        <div>
            <h1>학습 자료를 업로드하고, <span style="color:#159eae;">AI와 대화해 보세요!</span></h1>
            <p>
                AIVLE School 백서, 교안, 프로젝트 자료 등<br>
                나만의 문서를 업로드하여 RAG 기반으로 질문하고 답변을 받을 수 있습니다.
            </p>
        </div>
        <div class="hero-icon">📄⬆️</div>
    </div>
    """, unsafe_allow_html=True)

    feature_cols = st.columns(4, gap="small")
    feature_items = [
        ("📤", "문서 업로드", "PDF, PPT, DOCX 지원"),
        ("📚", "자동 처리", "문서 분할 및 임베딩"),
        ("💬", "AI 기반 질의응답", "업로드한 문서로 답변"),
        ("📑", "지식 관리", "등록 문서 확인 및 삭제")
    ]

    for col, (icon, title, desc) in zip(feature_cols, feature_items):
        with col:
            st.markdown(f"""
            <div class="info-card" style="padding:15px 8px; min-height:105px;">
                <div class="card-icon">{icon}</div>
                <h3 style="text-align:center; font-size:15px;">{title}</h3>
                <p style="text-align:center;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    section_title(1, "문서 업로드")

    upload_col, setting_col = st.columns([1.2, 0.9], gap="medium")

    with upload_col:
        st.markdown("""
        <div class="info-card" style="min-height:245px;">
            <div style="text-align:center; padding:8px 0 2px;">
                <div style="font-size:48px;">☁️</div>
                <h3>파일을 선택하여 업로드하세요</h3>
                <p>PDF, PPTX, DOCX, TXT 파일을 지원합니다.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "파일 선택",
            type=["pdf", "pptx", "docx", "txt"],
            key="main_document_uploader",
            help="파일당 최대 200MB를 권장합니다."
        )

        st.caption("지원 형식: PDF · PPTX · DOCX · TXT  |  파일당 최대 200MB")

    with setting_col:
        st.markdown("""
        <div class="info-card" style="min-height:245px;">
            <h3>⚙️ 업로드 설정</h3>
        """, unsafe_allow_html=True)

        category = st.selectbox(
            "📤 문서 카테고리",
            ["교육 자료", "프로젝트 자료", "취업 자료", "포트폴리오", "기타"],
            key="document_category"
        )

        description = st.text_area(
            "🗂️ 문서 설명 (선택)",
            placeholder="문서 내용을 간단히 입력하세요.\n(예: DX 교안, 미니프로젝트 자료 등)",
            height=95,
            key="document_description"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button(
            "⬆️ 업로드 및 처리",
            type="primary",
            use_container_width=True,
            key="process_uploaded_document"
        ):
            if uploaded_file is None:
                st.warning("먼저 업로드할 파일을 선택해 주세요.")
            elif any(
                item["name"] == uploaded_file.name
                for item in st.session_state.uploaded_documents
            ):
                st.warning("이미 등록된 파일명입니다. 기존 문서를 삭제한 후 다시 등록해 주세요.")
            else:
                try:
                    with st.spinner("문서를 읽고 청크 분할 및 임베딩을 진행하고 있습니다..."):
                        saved_doc = add_document_to_vectorstore(
                            uploaded_file,
                            category,
                            description
                        )
                        st.session_state.uploaded_documents.append(saved_doc)

                    st.success(
                        f"'{uploaded_file.name}' 업로드 및 처리가 완료되었습니다. "
                        f"({saved_doc['chunk_count']}개 청크)"
                    )
                    st.rerun()

                except Exception as error:
                    st.error("문서 처리 중 오류가 발생했습니다.")
                    st.caption(str(error))

    section_title(2, "등록된 문서 관리")

    documents = st.session_state.uploaded_documents

    filter_col, count_col, refresh_col = st.columns([2.2, 1, 0.45])

    with filter_col:
        search_name = st.text_input(
            "문서 검색",
            placeholder="🔍 문서명을 검색하세요.",
            label_visibility="collapsed",
            key="document_search"
        )

    with count_col:
        category_filter = st.selectbox(
            "카테고리",
            ["전체", "교육 자료", "프로젝트 자료", "취업 자료", "포트폴리오", "기타"],
            label_visibility="collapsed",
            key="document_filter"
        )

    with refresh_col:
        if st.button("🔄", help="목록 새로고침", use_container_width=True):
            st.rerun()

    filtered_docs = [
        doc for doc in documents
        if (search_name.lower() in doc["name"].lower())
        and (category_filter == "전체" or doc["category"] == category_filter)
    ]

    st.markdown("""
    <div style="
        display:grid;
        grid-template-columns: 2.4fr 1fr 1.2fr 0.8fr 0.8fr 1fr;
        gap:8px;
        background:#d9f2f3;
        border-radius:10px;
        padding:12px;
        color:#193653;
        font-weight:800;
        font-size:13px;">
        <div>파일명</div><div>카테고리</div><div>업로드일</div>
        <div>크기</div><div>처리 상태</div><div>작업</div>
    </div>
    """, unsafe_allow_html=True)

    if not filtered_docs:
        st.markdown("""
        <div class="info-card" style="text-align:center; margin-top:8px;">
            <div style="font-size:32px;">📂</div>
            <h3>등록된 문서가 없습니다.</h3>
            <p>위 영역에서 학습 자료를 업로드해 주세요.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for index, doc in enumerate(filtered_docs):
            row_cols = st.columns([2.4, 1, 1.2, 0.8, 0.8, 1], gap="small")

            with row_cols[0]:
                st.markdown(f"📄 **{doc['name']}**")

            with row_cols[1]:
                st.markdown(f"`{doc['category']}`")

            with row_cols[2]:
                st.caption(doc["uploaded_at"])

            with row_cols[3]:
                size_mb = doc["size"] / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")

            with row_cols[4]:
                st.markdown("<span style='color:#16a36a;font-weight:800;'>🟢 완료</span>", unsafe_allow_html=True)
                st.caption(f"{doc['chunk_count']} chunks")

            with row_cols[5]:
                action_cols = st.columns(3, gap="small")

                with action_cols[0]:
                    if st.button("👁️", key=f"preview_doc_{index}", help="문서 미리보기"):
                        st.session_state.document_preview = doc["name"]

                with action_cols[1]:
                    if st.button("💬", key=f"ask_doc_{index}", help="이 문서에 대해 질문하기"):
                        st.session_state.menu = "🧠 학습 Q&A"
                        st.session_state.document_question_hint = (
                            f"'{doc['name']}' 문서의 주요 내용을 요약해 주세요."
                        )
                        st.rerun()

                with action_cols[2]:
                    if st.button("🗑️", key=f"delete_doc_{index}", help="문서 삭제"):
                        try:
                            vectorstore = Chroma(
                                collection_name=COLLECTION,
                                embedding_function=OpenAIEmbeddings(
                                    model="text-embedding-3-small"
                                ),
                                persist_directory=CHROMA_DIR
                            )
                            if doc.get("ids"):
                                vectorstore.delete(ids=doc["ids"])

                            st.session_state.uploaded_documents = [
                                item for item in st.session_state.uploaded_documents
                                if item["name"] != doc["name"]
                            ]

                            if st.session_state.document_preview == doc["name"]:
                                st.session_state.document_preview = None

                            st.success(f"'{doc['name']}' 문서를 삭제했습니다.")
                            st.rerun()

                        except Exception as error:
                            st.error("문서 삭제 중 오류가 발생했습니다.")
                            st.caption(str(error))

    if st.session_state.document_preview:
        selected_doc = next(
            (
                item for item in st.session_state.uploaded_documents
                if item["name"] == st.session_state.document_preview
            ),
            None
        )

        if selected_doc:
            with st.expander(
                f"📖 문서 미리보기: {selected_doc['name']}",
                expanded=True
            ):
                st.caption(
                    f"카테고리: {selected_doc['category']} · "
                    f"업로드일: {selected_doc['uploaded_at']}"
                )
                if selected_doc.get("description"):
                    st.write(selected_doc["description"])
                st.text(selected_doc["preview"])
                if st.button("미리보기 닫기", key="close_document_preview"):
                    st.session_state.document_preview = None
                    st.rerun()

    st.caption(
        f"현재 세션에 등록된 문서: {len(documents)}개 · "
        f"저장된 청크: {sum(doc['chunk_count'] for doc in documents)}개"
    )


# =========================================================
# 12. 학습 Q&A 페이지
# =========================================================

def page_chat():

    st.markdown("""
    <div class="page-title">🧠 학습 Q&A</div>
    <div class="page-subtitle">
        AIVLE School 학습 자료를 기반으로 궁금한 내용을 질문해 보세요.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3>🤖 AIVLE AI Assistant</h3>
        <p>
            업로드된 교육 자료를 검색하여 근거 기반 답변을 제공합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 기존 대화 출력
    for message in st.session_state.messages:

        role = "user" if isinstance(message, HumanMessage) else "assistant"

        with st.chat_message(role):
            st.markdown(message.content)

    # 질문 입력
    user_input = st.chat_input("궁금한 내용을 입력하세요.")

    if user_input:

        st.session_state.messages.append(
            HumanMessage(content=user_input)
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            try:
                chain = get_chain()

                response_stream = chain.stream({
                    "input": user_input
                })

                full_response = st.write_stream(response_stream)

                st.session_state.messages.append(
                    AIMessage(content=full_response)
                )

            except Exception as e:

                st.error(
                    "답변 생성 중 오류가 발생했습니다. "
                    "OpenAI API 키와 ChromaDB 설정을 확인해 주세요."
                )

                st.caption(str(e))


# =========================================================
# 13. 나머지 페이지 임시 화면
# =========================================================

def page_placeholder(title, description, icon):

    st.markdown(f"""
    <div class="page-title">{icon} {title}</div>
    <div class="page-subtitle">{description}</div>

    <div class="hero">
        <div>
            <h1>{title}</h1>
            <h3>AIVLE School Learning Assistant</h3>
            <p>
                해당 페이지는 현재 준비 중입니다.<br>
                추후 관련 기능을 추가할 예정입니다.
            </p>
        </div>
        <div class="hero-icon">{icon}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================

# =========================================================
# 14. 온톨로지 페이지
# =========================================================

ONTOLOGY_CLASSES = [
    ("AIVLESchool", "AIVLE School", "교육 기관"),
    ("Curriculum", "교육과정", "교육 트랙 및 학습 단계"),
    ("Job", "직무", "AI 개발자 · DX 컨설턴트"),
    ("Project", "프로젝트", "미니 프로젝트 · 빅 프로젝트"),
    ("Skill", "학습기술", "Python · SQL · 데이터 분석 등"),
    ("Outcome", "성과", "포트폴리오 · 취업 연계"),
    ("Document", "자료", "백서 · 교안 · 프로젝트 문서"),
]

ONTOLOGY_PROPERTIES = [
    ("hasCurriculum", "교육 기관은 교육과정을 운영한다"),
    ("hasJob", "교육과정은 직무와 연결된다"),
    ("hasProject", "교육과정은 프로젝트를 포함한다"),
    ("hasSkill", "프로젝트는 학습 기술을 활용한다"),
    ("hasOutcome", "프로젝트는 성과로 이어진다"),
    ("hasDocument", "교육 지식은 문서 자료로 뒷받침된다"),
]

ONTOLOGY_TRIPLES = [
    ("AIVLE School", "hasCurriculum", "교육과정"),
    ("AIVLE School", "hasJob", "직무"),
    ("AIVLE School", "hasProject", "프로젝트"),
    ("교육과정", "hasSkill", "기초 역량"),
    ("교육과정", "hasSkill", "AI-DX 심화"),
    ("교육과정", "hasProject", "미니 프로젝트"),
    ("교육과정", "hasProject", "빅 프로젝트"),
    ("직무", "hasSkill", "Python"),
    ("직무", "hasSkill", "데이터 분석"),
    ("직무", "hasSkill", "생성형 AI"),
    ("프로젝트", "hasSkill", "SQL"),
    ("프로젝트", "hasSkill", "시각화"),
    ("프로젝트", "hasOutcome", "포트폴리오"),
    ("프로젝트", "hasOutcome", "취업 연계"),
    ("교육과정", "hasDocument", "AIVLE School 백서"),
    ("프로젝트", "hasDocument", "미니프로젝트 교안"),
]

def build_ontology_graph(show_classes=True, show_properties=True):
    """온톨로지 관계를 Graphviz DOT 그래프로 구성합니다."""
    nodes = [
        ('aivle', 'AIVLE\\nSchool', '#20b7b5', 'white'),
        ('curriculum', '교육과정', '#8b79e8', 'white'),
        ('job', '직무', '#38c9b8', 'white'),
        ('project', '프로젝트', '#ffc94f', '#173653'),
        ('skill', '학습기술', '#65aaf4', 'white'),
        ('outcome', '성과', '#ec8ab8', 'white'),
        ('document', '자료', '#8ac8dc', '#173653'),
    ]

    dot = [
        'digraph AIVLE {',
        'graph [rankdir=LR, bgcolor="transparent", pad="0.25", nodesep="0.55", ranksep="0.8"];',
        'node [shape=ellipse, style="filled", fontname="Malgun Gothic", fontsize=13, penwidth=1.5, color="#ffffff", margin="0.22,0.14"];',
        'edge [fontname="Malgun Gothic", fontsize=10, color="#8296b8", fontcolor="#425b78", arrowsize=0.8, penwidth=1.4];'
    ]

    if show_classes:
        for node_id, label, fill, font in nodes:
            dot.append(f'{node_id} [label="{label}", fillcolor="{fill}", fontcolor="{font}"];')
    else:
        dot.append('aivle [label="AIVLE\\nSchool", fillcolor="#20b7b5", fontcolor="white"];')

    if show_classes and show_properties:
        edges = [
            ('aivle', 'curriculum', 'hasCurriculum'),
            ('aivle', 'job', 'hasJob'),
            ('aivle', 'project', 'hasProject'),
            ('curriculum', 'skill', 'hasSkill'),
            ('curriculum', 'project', 'hasProject'),
            ('job', 'skill', 'hasSkill'),
            ('project', 'skill', 'usesSkill'),
            ('project', 'outcome', 'hasOutcome'),
            ('curriculum', 'document', 'hasDocument'),
        ]
    elif show_classes:
        edges = [
            ('aivle', 'curriculum', ''),
            ('aivle', 'job', ''),
            ('aivle', 'project', ''),
            ('curriculum', 'skill', ''),
            ('curriculum', 'project', ''),
            ('job', 'skill', ''),
            ('project', 'skill', ''),
            ('project', 'outcome', ''),
            ('curriculum', 'document', ''),
        ]
    else:
        edges = []

    for source, target, label in edges:
        dot.append(f'{source} -> {target} [label="{label}"];')

    dot.append('}')
    return "\n".join(dot)


def search_ontology(keyword):
    """개념(클래스 이름·설명)·관계 이름·트리플에서 검색어가 들어간 관계를 찾음"""
    if not keyword:
        return []
    kw = keyword.lower().replace(" ", "")
    # 클래스 설명에 걸리면 그 클래스 이름도 검색어로 포함
    names = {label for cid, label, desc in ONTOLOGY_CLASSES
             if kw in (cid + label + desc).lower().replace(" ", "")}
    hits = []
    for a, p, b in ONTOLOGY_TRIPLES:
        text = (a + p + b).lower().replace(" ", "")
        if kw in text or a in names or b in names:
            hits.append((a, p, b))
    return hits


RELATION_SYS = """당신은 KT AIVLE School 백서에서 지식 관계를 뽑는 도우미입니다.
[문서]에서 검색어와 관련된 사실만 (주체, 관계, 대상, 근거) 형태로 최대 15개 뽑으세요.
규칙:
1. 관계는 짧은 한국어 동사구로 씀 (예: 포함함, 필요함, 신청 기간, 대상, 연결됨).
2. 근거는 [문서]에 있는 문장을 그대로 30자 이내로 옮겨 씀. 문서에 없는 내용은 뽑지 않음.
3. 아래 JSON 형식만 출력함.
{{"triples": [{{"s": "주체", "p": "관계", "o": "대상", "evidence": "근거 원문"}}]}}

[문서]
{context}"""


def _norm(t):
    return re.sub(r"[\s\W_]+", "", str(t)).lower()


def rag_relation_graph(keyword):
    """검색어 → Hybrid 검색(백서) → LLM 관계 추출 → 근거가 원문에 있는 관계만 남김"""
    docs = hybrid_retrieve(keyword, k=6)
    context = format_docs(docs)
    prompt = ChatPromptTemplate.from_messages([("system", RELATION_SYS), ("human", "검색어: {kw}")])
    raw = (prompt | get_llm(False) | StrOutputParser()).invoke({"context": context, "kw": keyword})
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0)) if m else {"triples": []}
    src = _norm(context)
    kept, dropped = [], 0
    for t in data.get("triples", []):
        if not all(k in t for k in ("s", "p", "o", "evidence")):
            dropped += 1
            continue
        ev = _norm(t["evidence"])
        # 근거 검증: 근거 문장(앞 12자 이상)이 백서 원문에 실제로 있어야 남김
        if ev and (ev in src or ev[:12] in src):
            kept.append(t)
        else:
            dropped += 1
    return {"keyword": keyword, "triples": kept, "dropped": dropped, "n_docs": len(docs)}


def build_search_graph(hits, keyword):
    """검색 결과 관계만 그린 그래프 (검색어가 들어간 노드는 강조)"""
    kw = keyword.lower().replace(" ", "")
    dot = ['digraph S {', 'graph [rankdir=LR, bgcolor="transparent"];',
           'node [shape=ellipse, style="filled", fontname="Malgun Gothic", fontsize=12, fillcolor="#EAF7F3", color="#2BB5A3"];',
           'edge [fontname="Malgun Gothic", fontsize=10, color="#8296b8", fontcolor="#425b78"];']
    nodes = {n for a, _, b in hits for n in (a, b)}
    for n in nodes:
        if kw in n.lower().replace(" ", ""):
            dot.append(f'"{n}" [fillcolor="#20b7b5", fontcolor="white", penwidth=2];')
    for a, p, b in hits:
        dot.append(f'"{a}" -> "{b}" [label="{p}"];')
    dot.append('}')
    return "\n".join(dot)


def run_ontology_sample_query(query):
    """화면 예시 질의를 AIVLE 샘플 트리플 데이터에 적용합니다."""
    q = (query or "").lower()

    if "project" in q and "skill" in q:
        rows = [
            {"project": "VOC 분석 서비스", "skill": "Python, 데이터 분석, RAG"},
            {"project": "주정차 데이터 분석", "skill": "Python, SQL, 시각화"},
        ]
    elif "curriculum" in q or "교육과정" in q:
        rows = [
            {"subject": "AIVLE School", "relation": "hasCurriculum", "object": "교육과정"},
            {"subject": "교육과정", "relation": "hasProject", "object": "미니 프로젝트"},
            {"subject": "교육과정", "relation": "hasProject", "object": "빅 프로젝트"},
        ]
    elif "job" in q or "직무" in q:
        rows = [
            {"직무": "DX 컨설턴트", "관련 기술": "데이터 분석, SQL, 생성형 AI"},
            {"직무": "AI 개발자", "관련 기술": "Python, 머신러닝, 딥러닝"},
        ]
    else:
        rows = [
            {"subject": s, "predicate": p, "object": o}
            for s, p, o in ONTOLOGY_TRIPLES
        ]

    return rows


def page_ontology():
    st.markdown("""
    <div class="page-title">🕸️ 온톨로지</div>
    <div class="page-subtitle">Ontology & Knowledge Graph · AIVLE School 지식 관계 탐색</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero" style="min-height:140px;">
        <div>
            <h1>AIVLE School 지식 구조를 한눈에!</h1>
            <h3 style="color:#159eae;">교육과정, 직무, 프로젝트의 관계를 연결합니다.</h3>
            <p>
                교육과정, 직무, 프로젝트 등 주요 개념의 관계를<br>
                온톨로지(지식 그래프)로 시각화하고 탐색할 수 있습니다.
            </p>
        </div>
        <div class="hero-icon">🕸️💻</div>
    </div>
    """, unsafe_allow_html=True)

    # 상단 탭 버튼
    tab_names = ["지식 그래프", "클래스(개념)", "속성(Property)", "SPARQL 질의", "관리"]
    tab_icons = ["📊", "📁", "🔗", "📄", "⚙️"]
    tab_cols = st.columns(5, gap="small")

    for col, name, icon in zip(tab_cols, tab_names, tab_icons):
        with col:
            if st.button(
                f"{icon} {name}",
                key=f"ontology_tab_{name}",
                use_container_width=True,
                type="primary" if st.session_state.ontology_tab == name else "secondary"
            ):
                st.session_state.ontology_tab = name
                st.rerun()

    active_tab = st.session_state.ontology_tab

    if active_tab == "지식 그래프":
        section_title(1, "지식 그래프 시각화")

        # 관계망 검색: 개념·관계 이름으로 찾기
        onto_kw = st.text_input(
            "🔍 관계망 검색",
            placeholder="개념이나 관계를 입력하세요 (예: 프로젝트, Python, hasSkill, 포트폴리오)",
            key="onto_search"
        ).strip()
        onto_hits = search_ontology(onto_kw)
        if onto_kw:
            if onto_hits:
                st.success(f"'{onto_kw}' 관련 관계 {len(onto_hits)}건")
                st.graphviz_chart(build_search_graph(onto_hits, onto_kw), use_container_width=True)
                st.dataframe([{"주체": a, "관계": p, "대상": b} for a, p, b in onto_hits],
                             use_container_width=True, hide_index=True)
                if st.button(f"💬 '{onto_kw}' 챗봇에 물어보기", key="onto_ask"):
                    st.session_state.chat_pending = f"{onto_kw}에 대해 백서 기준으로 알려줘."
                    st.rerun()
            else:
                st.info(f"'{onto_kw}'와 일치하는 예시 관계가 없습니다. 아래 '백서에서 관계망 만들기'를 눌러 보세요.")

            # RAG 기반 관계망: 학습 백서에서 검색어 관련 지식을 찾아 관계망으로 보여줌
            if st.button(f"📘 '{onto_kw}' 백서에서 관계망 만들기 (RAG)", key="onto_rag_btn",
                         type="primary", use_container_width=True):
                with st.spinner("백서에서 관련 내용을 찾고 관계를 뽑는 중..."):
                    try:
                        st.session_state.onto_rag = rag_relation_graph(onto_kw)
                    except Exception as e:
                        st.session_state.onto_rag = None
                        st.error(f"관계망 생성 중 오류가 발생했습니다: {e}")

            rag_res = st.session_state.get("onto_rag")
            if rag_res and rag_res["keyword"] == onto_kw:
                st.markdown(f"#### 📘 백서 기반 관계망 · '{onto_kw}'")
                if rag_res["triples"]:
                    st.caption(f"백서 근거 청크 {rag_res['n_docs']}개 · 원문 근거가 확인된 관계 "
                               f"{len(rag_res['triples'])}개 (근거 불일치 {rag_res['dropped']}개 제외)")
                    st.graphviz_chart(
                        build_search_graph([(t["s"], t["p"], t["o"]) for t in rag_res["triples"]], onto_kw),
                        use_container_width=True)
                    st.dataframe([{"주체": t["s"], "관계": t["p"], "대상": t["o"], "백서 근거": t["evidence"]}
                                  for t in rag_res["triples"]], use_container_width=True, hide_index=True)
                else:
                    st.info("백서에서 근거가 확인되는 관계를 찾지 못했습니다.")

        graph_col, info_col = st.columns([2.15, 0.85], gap="medium")

        with graph_col:
            with st.container(border=True):
                filter_cols = st.columns([1, 1, 1.5])
                with filter_cols[0]:
                    show_classes = st.checkbox("개념(클래스)", value=True, key="onto_show_classes")
                with filter_cols[1]:
                    show_properties = st.checkbox("관계(속성)", value=True, key="onto_show_properties")
                with filter_cols[2]:
                    st.caption("노드를 선택하는 대신 오른쪽 목록에서 개념을 탐색할 수 있습니다.")

                try:
                    st.graphviz_chart(
                        build_ontology_graph(show_classes, show_properties),
                        use_container_width=True
                    )
                except Exception:
                    st.warning("그래프를 표시할 수 없습니다. 아래 관계 목록을 확인해 주세요.")
                    st.dataframe(
                        [{"주체": s, "관계": p, "대상": o} for s, p, o in ONTOLOGY_TRIPLES],
                        use_container_width=True,
                        hide_index=True
                    )

        with info_col:
            with st.container(border=True):
                st.markdown("#### 🧩 온톨로지 정보")
                class_tab, property_tab = st.tabs(["클래스 (7)", "속성 (6)"])
                with class_tab:
                    for _, label, description in ONTOLOGY_CLASSES:
                        st.markdown(f"🔹 **{label}**")
                        st.caption(description)
                with property_tab:
                    for prop, description in ONTOLOGY_PROPERTIES:
                        st.markdown(f"🔗 **{prop}**")
                        st.caption(description)

        lower_left, lower_right = st.columns([1.2, 0.9], gap="medium")
        with lower_left:
            section_title(2, "관계 데이터 미리보기")
            st.dataframe(
                [{"주체": s, "관계": p, "대상": o} for s, p, o in ONTOLOGY_TRIPLES],
                use_container_width=True,
                hide_index=True
            )
        with lower_right:
            section_title(3, "예시 질의")
            examples = [
                "DX 컨설턴트에게 필요한 핵심 역량은?",
                "미니프로젝트 주제와 사용 기술은?",
                "교육과정 간의 관계를 알려줘.",
                "특정 프로젝트에 사용된 기술은?"
            ]
            for i, example in enumerate(examples):
                if st.button(f"🔎 {example}", key=f"onto_example_{i}", use_container_width=True):
                    st.session_state.ontology_tab = "SPARQL 질의"
                    st.session_state.ontology_query_text = (
                        "SELECT ?project ?skill WHERE { ?project a ex:Project ; ex:useSkill ?skill . }"
                        if i == 1 or i == 3 else
                        "SELECT ?subject ?relation ?object WHERE { ?subject ?relation ?object . }"
                    )
                    st.rerun()

    elif active_tab == "클래스(개념)":
        section_title(1, "온톨로지 클래스")
        st.markdown("AIVLE School의 주요 개념을 유형별로 정리했습니다.")
        cols = st.columns(3, gap="medium")
        for i, (class_id, label, description) in enumerate(ONTOLOGY_CLASSES):
            with cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"### 🧊 {label}")
                    st.caption(f"클래스 ID: `{class_id}`")
                    st.write(description)
                    related = [t for t in ONTOLOGY_TRIPLES if t[0] == label or t[2] == label]
                    st.caption(f"연결된 관계 데이터: {len(related)}건")

    elif active_tab == "속성(Property)":
        section_title(1, "관계 속성")
        st.markdown("개념과 개념 사이의 연결 의미를 정의합니다.")
        st.dataframe(
            [{"속성": prop, "설명": description} for prop, description in ONTOLOGY_PROPERTIES],
            use_container_width=True,
            hide_index=True
        )
        section_title(2, "관계 트리플")
        st.dataframe(
            [{"주체(Subject)": s, "속성(Predicate)": p, "대상(Object)": o}
             for s, p, o in ONTOLOGY_TRIPLES],
            use_container_width=True,
            hide_index=True
        )

    elif active_tab == "SPARQL 질의":
        section_title(1, "SPARQL 질의로 지식 탐색하기")
        default_query = st.session_state.get(
            "ontology_query_text",
            "SELECT ?project ?skill WHERE {\\n  ?project a ex:Project ; ex:useSkill ?skill .\\n}"
        )
        query = st.text_area(
            "SPARQL 예시 질의",
            value=default_query,
            height=125,
            key="ontology_query_editor"
        )
        run_col, help_col = st.columns([1, 3])
        with run_col:
            run_query = st.button("▶ 실행", type="primary", use_container_width=True)
        with help_col:
            st.caption("현재 데모는 AIVLE 샘플 지식 데이터에 대한 예시 질의를 제공합니다.")

        if run_query:
            st.session_state.ontology_query_result = run_ontology_sample_query(query)

        if st.session_state.ontology_query_result is not None:
            result = st.session_state.ontology_query_result
            section_title(2, f"질의 결과 ({len(result)}건)")
            st.dataframe(result, use_container_width=True, hide_index=True)

        with st.expander("사용 가능한 예시 질의"):
            st.code(
                "SELECT ?project ?skill WHERE {\\n"
                "  ?project a ex:Project ; ex:useSkill ?skill .\\n"
                "}\\n\\n"
                "# 교육과정 관계 조회 예시\\n"
                "SELECT ?subject ?relation ?object WHERE {\\n"
                "  ?subject ?relation ?object .\\n"
                "}",
                language="sparql"
            )

    elif active_tab == "관리":
        section_title(1, "온톨로지 관리")
        st.info("현재 화면은 예시 지식 그래프 데이터로 구성되어 있습니다. 실제 운영 데이터와 연결하기 전 개념 및 관계를 검토하는 관리 화면입니다.")
        st.metric("클래스 수", len(ONTOLOGY_CLASSES))
        st.metric("속성 수", len(ONTOLOGY_PROPERTIES))
        st.metric("관계 데이터 수", len(ONTOLOGY_TRIPLES))
        section_title(2, "등록된 개념")
        st.dataframe(
            [{"클래스 ID": cid, "개념": label, "설명": desc}
             for cid, label, desc in ONTOLOGY_CLASSES],
            use_container_width=True,
            hide_index=True
        )
        section_title(3, "등록된 관계")
        st.dataframe(
            [{"주체": s, "관계": p, "대상": o} for s, p, o in ONTOLOGY_TRIPLES],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# 15. 최종 화면 구성
# =========================================================

if not st.session_state.profile_completed:
    page_profile()
else:
    main_col, side_col = st.columns(
        [3.2, 1.15],
        gap="large"
    )

    with main_col:
        menu = st.session_state.menu

        if menu == "🏠 홈":
            page_home()
        elif menu == "👋 소개":
            page_intro()
        elif menu == "📚 교육과정":
            page_education()
        elif menu == "💼 취업과정":
            page_career()
        elif menu == "📁 포트폴리오":
            page_portfolio()
        elif menu == "📂 파일 불러오기":
            page_file_upload()
        elif menu == "🕸️ 온톨로지":
            page_ontology()
        elif menu == "🧠 학습 Q&A":
            page_chat()
        else:
            page_home()

    with side_col:
        right_panel()
