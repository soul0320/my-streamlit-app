import streamlit as st
import calendar
from datetime import date, datetime
from html import escape

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
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

/* 기본 여백 */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 100%;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #55bdc5, #63c6cc);
}

/* 펼쳤을 때만 너비 고정 (닫힌 상태까지 고정하면 다시 펼쳐지지 않음) */
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 210px;
    max-width: 210px;
}

/* 닫힌 상태에서 다시 펼치는 버튼(헤더 왼쪽 ») 항상 보이게 */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999;
}
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"] {
    background: #55bdc5 !important;
    color: white !important;
    border-radius: 10px;
}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stExpandSidebarButton"] svg {
    fill: white !important;
    color: white !important;
}

/* 프로필 설정 화면 */
.profile-hero {
    background: linear-gradient(110deg, #d9f5f5, #c8eff2);
    border-radius: 22px;
    padding: 32px 36px;
    margin: 10px 0 24px;
}
.profile-hero h1 { color: #173653; font-size: 34px; font-weight: 900; margin: 0 0 8px; }
.profile-hero p { color: #4c6478; font-size: 16px; margin: 0; }

section[data-testid="stSidebar"] * {
    color: white;
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

section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: #eafafa;
    border-color: #eafafa;
    color: #193653 !important;
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

    [사용자 프로필]
    {profile}
    - 예비 에이블러: 지원 자격, 선발 절차, 교육 전 준비 위주로 쉽게 설명하세요.
    - 에이블러: 출결, 일정, 미니/빅프로젝트, 학습 전략 위주로 구체적으로 설명하세요.
    - 수료자: 취업 지원, 포트폴리오, 면접 위주로 설명하세요.

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
            "input": itemgetter("input"),
            "profile": itemgetter("profile")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


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

# 공통 데이터: profile (공통 규칙 4번)
if "profile" not in st.session_state:
    st.session_state.profile = None


# =========================================================
# 4-1. 프로필 설정 (처음 접속 시 먼저 표시)
# =========================================================

STAGES = ["🌱 예비 에이블러", "🎓 에이블러 (교육생)", "🏆 수료자 (1년)"]
TRACKS = ["DX 컨설턴트 트랙", "AI 개발자 트랙"]

def page_profile_setup():

    st.markdown("""
    <div class="profile-hero">
        <h1>AIVLE School 학습 도우미 👋</h1>
        <p>먼저 프로필을 설정하면, 단계에 맞는 안내와 챗봇 답변을 보여드려요.</p>
    </div>
    """, unsafe_allow_html=True)

    prev = st.session_state.profile or {}

    left, center, right = st.columns([1, 2, 1])

    with center:
        with st.form("profile_form"):
            name = st.text_input(
                "이름 (닉네임)",
                value=prev.get("name", ""),
                key="profile_name"
            )
            stage = st.radio(
                "현재 단계",
                STAGES,
                index=STAGES.index(prev["stage"]) if prev.get("stage") in STAGES else 1,
                horizontal=True,
                key="profile_stage"
            )
            track = st.radio(
                "트랙",
                TRACKS,
                index=TRACKS.index(prev["track"]) if prev.get("track") in TRACKS else 0,
                horizontal=True,
                key="profile_track"
            )
            company = st.text_input(
                "희망 기업 (선택)",
                value=prev.get("company", ""),
                placeholder="예: KT, 금융권, 공기업",
                key="profile_company"
            )
            job = st.text_input(
                "희망 직무 (선택)",
                value=prev.get("job", ""),
                placeholder="예: DX 컨설턴트, 데이터 분석가",
                key="profile_job"
            )

            submitted = st.form_submit_button(
                "시작하기 ➜",
                type="primary",
                use_container_width=True
            )

        if submitted:
            st.session_state.profile = {
                "name": name.strip() or "에이블러",
                "stage": stage,
                "track": track,
                "company": company.strip(),
                "job": job.strip()
            }
            st.session_state.menu = "🏠 홈"
            st.rerun()


# 프로필이 없으면 설정 화면만 보여주고 아래 화면은 그리지 않음
if st.session_state.profile is None:
    page_profile_setup()
    st.stop()

profile = st.session_state.profile


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
# 6. 왼쪽 사이드바
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

with st.sidebar:

    # AIVLE 로고
    st.markdown("""
    <div class="logo-box">
        <div class="logo-main">◢ AIVLE</div>
        <div class="logo-sub">AIVLE SCHOOL</div>
    </div>
    """, unsafe_allow_html=True)

    # 내 프로필 + 변경 버튼
    st.markdown(
        f"""
        <div style="font-size:13px;line-height:1.7;margin:-10px 0 8px;">
            <b>{escape(profile['name'])}</b> 님<br>
            {profile['stage']}<br>{profile['track']}
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("✏️ 프로필 변경", key="profile_edit", use_container_width=True):
        st.session_state.profile = None
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

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

    # 관리자 메뉴는 제거하고 문서 업로드/관리는 유지
    st.markdown("#### 📄 백서 및 학습 문서 업로드")

    uploaded_file = st.file_uploader(
        "파일 선택",
        type=["pdf", "txt", "docx"],
        help="200MB per file"
    )

    st.caption("200MB per file · PDF, TXT, DOCX")

    st.markdown("---")
    st.markdown("#### 📑 등록된 문서 관리")

    st.selectbox(
        "문서 선택",
        ["등록된 문서 없음"],
        label_visibility="collapsed"
    )

    st.caption("저장된 청크: 0개")

    if st.button("🗑️ 대화 전체 삭제", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <br><br>
    <div style="font-size:12px; line-height:1.8; color:white;">
        AIVLE School<br>
        Learning Assistant
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 7. 오른쪽 공통 패널
# =========================================================

def right_panel():

    st.markdown("""
    <div class="right-card" style="text-align:center;">
        <div class="assistant-icon">👩🏻‍💻</div>
        <h3 style="margin-top:12px;">AIVLE Assistant</h3>
        <p>Your Learning Partner</p>
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
        💡 추천 질문
    </div>

    <div class="question-item">KT 에이블스쿨 지원 자격은?</div>
    <div class="question-item">미니프로젝트는 어떻게 진행되나요?</div>
    <div class="question-item">교육 과정은 어떻게 구성되나요?</div>
    <div class="question-item">수료 기준은 무엇인가요?</div>

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

    st.markdown("""
    <div class="page-title">Hello, {name}! 👋</div>
    <div class="page-subtitle">{stage} · {track} | 오늘도 새로운 지식을 함께 배워봐요.</div>
    """.format(
        name=escape(profile["name"]),
        stage=profile["stage"],
        track=profile["track"]
    ), unsafe_allow_html=True)

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
# 11. 학습 Q&A 페이지
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
                    "input": user_input,
                    "profile": f"{profile['stage']} / {profile['track']} / 희망: {profile['company']} {profile['job']}"
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
# 12. 나머지 페이지 임시 화면
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
# 13. 최종 화면 구성
# =========================================================

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
        page_placeholder(
            "취업과정", "취업 준비와 커리어 개발", "💼"
        )

    elif menu == "📁 포트폴리오":
        page_placeholder(
            "포트폴리오", "프로젝트 및 개인 포트폴리오 관리", "📁"
        )

    elif menu == "📂 파일 불러오기":
        page_placeholder(
            "파일 불러오기", "학습 자료 업로드 및 관리", "📂"
        )

    elif menu == "🕸️ 온톨로지":
        page_placeholder(
            "온톨로지", "교육 및 취업 지식 관계 탐색", "🕸️"
        )

with side_col:
    right_panel()
