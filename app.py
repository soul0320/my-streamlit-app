
import os, json, smtplib, re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import calendar as pycal
import pandas as pd
import streamlit as st
from openai import OpenAI
from email.mime.text import MIMEText

# ── 기본 설정 ─────────────────────────────────────
# 키는 환경변수에서 자동
client = OpenAI()

# 이메일 요청 시 받을 주소 (음성 인식 오류를 피하려고 코드에 고정)
MY_EMAIL = "soulsky0218@gmail.com" #수정필요 (기본수신 이메일)

# 에이블 대표 이미지 경로 (app.py와 같은 폴더에 둔 파일 이름)
IMAGE_PATH = "able.png"

# 사이드바 메뉴 (아래 셀들도 이 이름으로 화면을 나눔)
PAGES = ["🎙️ AI 비서", "📊 업무 우선순위 분석", "📅 마감 캘린더", "📨 팔로업 메일", "🎧 출근길 브리핑",
         "🍃 오늘의 할 일", 🗄️ 회의록 아카이브]

# 대화 기록 최대 개수 (질문 1 + 답변 1 = 2개 → 10이면 최근 5번의 대화)
MAX_HISTORY = 10

# 주석 추가 ~~~
def kst_today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()

def kst_now_str():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")


# 음성용 텍스트 정리: 마크다운 기호를 빼야 TTS가 매끄럽게 읽음
def to_speech_text(text):
    text = re.sub(r"[#*`>|]", "", text)              # 제목·굵게 등 기호 제거
    text = re.sub(r"^\s*-\s*", "", text, flags=re.M)  # 줄 앞 목록 기호 '-' 제거
    return text[:4000]                                # TTS 입력 길이 한도

# 에이블 역할
sys_role = """
당신은 '김 대리'의 개인 비서 '에이블'입니다.
김 대리가 회의에 참석하지 못했을 때, 회의 내용을 정리해 업무를 놓치지 않도록 돕습니다.

[주요 업무]
1. 회의록을 분석하고 핵심 내용을 요약합니다.
2. 회의록에서 김 대리에게 배정된 업무 지시사항을 찾아 자세히 정리합니다.
3. 김 대리의 다양한 질문에 답변합니다.

[답변 규칙]
- 한국어로 정중하고 간결하게 답변합니다.
- 회의록에 없는 내용은 지어내지 않습니다.
- 요약을 주제 → 논의 내용 → 결정 사항 순서로 구역을 나눠서 진행합니다.
- 김 대리 담당 업무는 요약과 별도로 정리하고, 업무마다 업무명, 세부 내용, 기한을 함께 적습니다. 기한이 언급되지 않은 업무는 '기한 미정'으로 표시합니다.
- 결정 사항에는 예산, 일정, 채널처럼 구체적인 수치와 항목을 포함합니다.
- 업무 세부 내용에는 회의에서 언급된 방향과 조건을 포함합니다.
- 기한은 회의에서 해당 업무에 대해 명시적으로 언급된 경우에만 적습니다. 추측하지 않습니다.
- 이메일은 김 대리가 요청한 경우에만 보내고 이메일 본문은 마크다운 기호 없이 작성합니다.
- 이메일 요청이 있으면 반드시 send_email 도구를 사용해 실제로 보냅니다. 도구를 쓰지 않고 보냈다고 말하지 않습니다.
- 받는 사람 주소는 [김 대리 이메일 주소]에 적힌 값만 사용합니다.
"""

# ── 이메일 발송 함수 (미션 1) ──
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    # Secrets에 값이 없으면 연결 전에 바로 안내
    if not sender or not password:
        return {"status": "failed",
                "error": "GMAIL_ADDRESS 또는 GMAIL_APP_PASSWORD가 설정되지 않았습니다."}

    msg = MIMEText(body, "plain", "utf-8")   # 한글 본문 깨짐 방지
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        # 465 포트: 처음부터 SSL로 연결 (클라우드 서버에서 더 안정적)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# ── 도구 정의 (미션 7)  ──
# description: LLM이 도구를 언제·어떻게 쓸지 판단하는 기준
tools = [{
    "type": "function",
    "name": "send_email",
    "description": "이메일을 발송합니다. 요청받은 내용은 한 통의 이메일로 모아서 보냅니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string",
                   "description": "이메일 수신자. 예: 'test@gmail.com'"},
            "subject": {"type": "string",
                        "description": "이메일 제목. 예: '마케팅 전략 초안'"},
            "body": {"type": "string",
                     "description": "이메일 본문. 회의 요약과 김 대리 담당 업무를 모두 포함하고, 마크다운 기호(#, *, -) 없이 일반 텍스트로 작성. 예: '안녕하세요, 김 대리님. 비서 에이블입니다.'"}
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}]

# ── 업무 우선순위 분석용 출력 스키마 (05 실습 JSON 구조화) ──
priority_schema = {
    "type": "json_schema",
    "name": "priority",
    "schema": {
        "type": "object",
        "properties": {
            "업무목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "업무명": {"type": "string"},
                        "기한": {"type": ["string", "null"]},
                        "긴급도": {"type": "string", "enum": ["높음", "낮음"]},
                        "중요도": {"type": "string", "enum": ["높음", "낮음"]},
                        "근거": {"type": "string"}
                    },
                    "required": ["업무명", "기한", "긴급도", "중요도", "근거"],
                    "additionalProperties": False
                }
            },
            "참석자별업무": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "이름": {"type": "string"},
                        "업무수": {"type": "integer"}
                    },
                    "required": ["이름", "업무수"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["업무목록", "참석자별업무"],
        "additionalProperties": False
    }
}

# 우선순위 분석 프롬프트 템플릿
priority_template = """
다음 회의록을 분석하세요.

1. 김 대리에게 배정된 업무를 모두 추출하세요.
   - 기한은 회의록에서 해당 업무에 명시된 표현 그대로 적고, 없으면 null로 하세요.
   - 긴급도: 기한이 가깝거나 회의에서 서둘러야 한다고 언급되면 '높음', 아니면 '낮음'
   - 중요도: 신제품 출시와 매출에 직접 영향을 주면 '높음', 보조 업무면 '낮음'
   - 근거: 판단 이유를 회의록 내용에 근거해 한 문장으로 쓰세요.
2. 회의 참석자(김 대리 포함)별로 새로 배정된 업무 수를 세세요.
   업무를 전달하거나 회의록을 정리하는 일은 업무로 세지 않습니다.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# 긴급도 × 중요도 → 분류 (파이썬으로 정해서 결과 일관되게)
def classify(urgent, important):
    if urgent == "높음" and important == "높음":
        return "즉시 착수"
    elif important == "높음":
        return "순차 진행"
    else:
        return "위임 검토"

# ── 이메일 요청인지 판단 ('메일' + '보내다/전송/발송'이 함께 있어야 요청으로 봄) ──
def wants_email(question):
    q = question.replace(" ", "")                        # '이 메일로'처럼 띄어 써도 인식
    has_mail = "메일" in q or "email" in q.lower()
    has_send = re.search(r"보내|전송|발송|송부", q) is not None
    return has_mail and has_send

# ── 에이블에게 질문하기 (음성·텍스트 대화 공통) ──
def ask_able(question, current_meeting):
    active_doc_name = st.session_state.active_doc
    history = st.session_state.messages_dict.get(active_doc_name, [])
    """질문 → LLM 호출(이메일 도구 포함) → (답변, 이메일 발송 결과) 반환"""
    # 프롬프트: 질문 + 고정 이메일 주소 + 회의록
    prompt = f"""
{question}

[김 대리 이메일 주소]
{MY_EMAIL}

[미팅 내용: {active_doc_name}]
{current_meeting}
"""
    # 1단계: 역할 → 이전 대화(멀티턴) → 이번 질문
    request = {
        "model": "gpt-4o-mini",
        "input": [{"role": "system", "content": sys_role}]
                 + history
                 + [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    # 이메일 도구는 '메일을 보내 달라'는 요청이 확실할 때만 전달 (파이썬이 판단)
    #  - 요청이 없으면 도구 자체를 주지 않아서, 이전 대화에 메일 요청이 있어도 다시 보내지 않음
    if wants_email(question):
        request["tools"] = tools
        request["tool_choice"] = {"type": "function", "name": "send_email"}
        request["parallel_tool_calls"] = False     # 메일 한 통만
    response = client.responses.create(**request)

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return response.output_text, None

    # 2단계: LLM이 정한 인자(to, subject, body)로 이메일 함수 실행
    tool_outputs = []
    email = None
    for item in tool_calls:
        if item.name == "send_email":
            args = json.loads(item.arguments)
            result = send_email(**args)
            email = {"to": args["to"], **result}
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result, ensure_ascii=False)
            })

    # 3단계: 실행 결과를 LLM에 돌려주고 최종 답변 받기
    final_response = client.responses.create(
        model="gpt-4o-mini",
        previous_response_id=response.id,
        input=tool_outputs,
        temperature=0
    )
    return final_response.output_text, email

# ══════════════════════════════════════════════════
# 화면 공통: 제목 · 소개 · 세션 상태 · 사이드바 메뉴
# ══════════════════════════════════════════════════
st.set_page_config(page_title="AI 개인비서 에이블", page_icon="🤖", layout="wide")
st.title("🤖 AI 개인비서 에이블")
st.caption("다중 회의록/업무자료 분석, 스마트 캘린더, 실시간 대화 및 지출/목표 대시보드를 제공합니다.")

# 소개 영역: 왼쪽 이미지 / 오른쪽 역할·핵심 기능
left, right = st.columns([1, 5])
with left:
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=180)
    else:
        st.info("🖼️ 이미지 준비 중")
with right:
    st.markdown("""
#### 김 대리의 AI 개인비서, 에이블
- 📝 회의록을 분석해 **요약하고 내 업무를 찾아** 드립니다.
- 🎙️ **음성이나 대화로 질문**하면 글과 음성으로 답하고, 요청 시 **이메일**로 보냅니다.
- 📅 업무 **우선순위**를 정리하고 마감일을 **캘린더**에 표시합니다.
""")
st.divider()

# 세션 상태 초기화 (S04: 처음 한 번만)
if "messages" not in st.session_state:
    st.session_state.messages = []        # 대화 히스토리 (음성·텍스트 대화 공통)
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""    # 업로드한 회의록 (메뉴를 옮겨도 유지)
if "notice" not in st.session_state:
    st.session_state.notice = None        # 최근 처리 결과 (인식된 질문, 이메일 발송 결과)
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None    # 최근 답변 음성
    st.session_state.play_audio = False   # 새 답변일 때만 자동 재생

# 사이드바 메뉴: 누른 메뉴의 기능만 화면에 표시
with st.sidebar:
    st.header("🤖 에이블 메뉴")
    page = st.radio("기능 선택", PAGES, label_visibility="collapsed")
    st.divider()

    # 회의록 업로드 (사이드바에 있어서 어느 메뉴에서든 올리고 바꿀 수 있음)
    st.subheader("📄 문서 업로드")
    uploaded_files = st.file_uploader(
        "파일 업로드 (여러 개 가능: txt)", 
        type=["txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_docs:
                content = uf.read().decode("utf-8", errors="ignore")
                st.session_state.uploaded_docs[uf.name] = content
                st.session_state.active_doc = uf.name
                # 아카이브에 날짜/시간별로 누적 저장 (삭제해도 아카이브에는 계속 남음!)
                st.session_state.archived_meetings.append({
                    "time": kst_now_str(),
                    "title": uf.name,
                    "content": content
                })
        st.success(f"{len(uploaded_files)}개 파일 로드 완료!")

    st.markdown("#### 📑 등록된 문서 목록")
    doc_keys = list(st.session_state.uploaded_docs.keys())
    if not doc_keys:
        st.caption("등록된 문서가 없습니다.")
    else:
        for dname in doc_keys:
            c_doc, c_del = st.columns([4, 1])
            is_active = (dname == st.session_state.active_doc)
            prefix = "👉 " if is_active else "📄 "
            c_doc.write(f"{prefix}**{dname}**")
            # 개별 파일 취소/삭제 기능
            if c_del.button("❌", key=f"del_{dname}", help="이 문서를 작업 목록에서 제거합니다 (아카이브에는 보존됨)"):
                del st.session_state.uploaded_docs[dname]
                if st.session_state.active_doc == dname:
                    st.session_state.active_doc = list(st.session_state.uploaded_docs.keys())[0] if st.session_state.uploaded_docs else ""
                st.rerun()

    if st.session_state.uploaded_docs:
        st.divider()
        st.session_state.active_doc = st.selectbox(
            "🎯 현재 작업 대상 문서 선택", 
            list(st.session_state.uploaded_docs.keys()),
            index=list(st.session_state.uploaded_docs.keys()).index(st.session_state.active_doc) if st.session_state.active_doc in st.session_state.uploaded_docs else 0
        )

    st.divider()
    active_history = st.session_state.messages_dict.get(st.session_state.active_doc, [])
    st.subheader("📋 히스토리 요약")
    st.markdown(f"**현재 문서:** `{st.session_state.active_doc or '선택 없음'}`")
    st.markdown(f"**대화 기록:** {len(active_history) // 2}건 (최대 {MAX_HISTORY//2}회)")

current_meeting = st.session_state.uploaded_docs.get(st.session_state.active_doc, "")


# ══════════════════════════════════════════════════
# 메뉴 1: AI 비서 (회의록 · 음성 요청 · 실시간 대화)
# ══════════════════════════════════════════════════
if page == PAGES[0]:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── 왼쪽: 회의록 업로드 + 음성 요청 ──
    with col_left:
        st.subheader("📄 문서 업로드")
        if not current_meeting:
            st.info("사이드바에서 문서 파일을 업로드하면 여기서 내용을 볼 수 있습니다.")
        else:
            with st.expander("문서 원문 미리보기", expanded=True):     # 눌러야 내용이 펼쳐짐
                st.text_area("문서 내용", current_meeting, height=230, label_visibility="collapsed")

        st.divider()
        st.subheader("🎙️ 음성으로 요청하기")
        audio_value = st.audio_input("무엇을 도와드릴까요?")
        voice_on = st.toggle("🔊 답변을 음성으로 듣기", value=True)
        run_voice = st.button("실행", type="primary", use_container_width=True)"

        # 최근 처리 결과: 인식된 질문(STT 정확성 확인), 이메일 발송 결과
        notice = st.session_state.notice
        if notice:
            if notice["source"] == "voice":
                st.markdown("#### 🎙️ 인식된 질문")
                st.info(notice["question"])
            if notice["email"]:
                if notice["email"]["status"] == "success":
                    st.success(f"📧 {notice['email']['to']}(으)로 이메일을 보냈습니다.")
                else:
                    st.error(f"이메일 발송 실패: {notice['email'].get('error')}")

    # ── 오른쪽: 에이블과 실시간 대화 (음성 질문도 함께 표시) ──
    with col_right:
        st.subheader(f"💬 '{st.session_state.active_doc}'전 실시간 대화")
        current_history = st.session_state.messages_dict.setdefault(st.session_state.active_doc, [])
        with st.container(height=480, border=True):
            if not current_history:
                st.caption("(f"'{st.session_state.active_doc}'에 대한 첫 대화를 시작해 보세요.")
            for message in current_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 최근 답변 음성 (새 답변일 때만 자동 재생)
        if st.session_state.last_audio:
            st.audio(st.session_state.last_audio, format="audio/mp3",
                     autoplay=st.session_state.play_audio)
            st.session_state.play_audio = False

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("에이블에게 메시지 보내기",
                                      placeholder="어떤 업무를 도와드릴까요?")
            submitted = st.form_submit_button("💬 에이블에게 보내기",
                                              type="primary", use_container_width=True)

        if st.button("🗑️ 대화 기록 지우기", use_container_width=True"):
            st.session_state.messages_dict[st.session_state.active_doc] = []
            st.session_state.notice = None
            st.session_state.last_audio = None
            st.rerun()

    # ── 질문 처리: 음성(실행 버튼) 또는 텍스트(보내기 버튼) ──
    question, source = None, None
    if run_voice:
        if audio_value is None:
            col_left.warning("먼저 마이크 버튼을 눌러 질문을 녹음해 주세요.")
        else:
            try:
                with col_left, st.spinner("음성을 인식하고 있습니다.."):
                    # STT: 녹음은 파일이 아니라 메모리 데이터 → (파일명, 바이트)로 전달
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("recording.wav", audio_value.getvalue())
                    )
                question, source = transcript.text, "voice"
            except Exception as e:
                col_left.error(f"음성 인식 오류: {e}")
    elif submitted and user_input.strip():
        question, source = user_input.strip(), "text"

    if question:
        try:
            with col_right, st.spinner("답변을 준비하고 있습니다.."):
                answer, email = ask_able(question, current_meeting)

                # 대화 히스토리 저장 (질문·답변 한 쌍씩, 최근 MAX_HISTORY개만 유지)
                current_history.append({"role": "user", "content": question})
                current_history.append({"role": "assistant", "content": answer})
                st.session_state.messages_dict[st.session_state.active_doc] = current_history[-MAX_HISTORY:]
                st.session_state.notice = {"source": source, "question": question, "email": email}

                # LLM 최종 응답을 음성으로 즉시 재생 (미션 5·6 TTS)
                if voice_on:
                    speech = client.audio.speech.create(
                        model="tts-1",
                        voice="nova",
                        input=to_speech_text(answer),
                        instructions="비서가 담당자에게 얘기하는 정중한 말투로 빠뜨리지 말고 또박또박 읽어주세요."
                    )
                    st.session_state.last_audio = speech.content
                    st.session_state.play_audio = True
            st.rerun()
        except Exception as e:
            col_right.error(f"오류: {e}")


# ══════════════════════════════════════════════════
# 메뉴 2: 업무 우선순위 & 목 분석 (아이젠하워 매트릭스)
# ══════════════════════════════════════════════════
elif page == PAGES[1]:
    st.subheader("📊 업무 우선순위 & 목표 분석")
    st.caption("선택된 회의록에서 김 대리 업무를 긴급도 × 중요도로 분류하고, 신제품 마케팅 지출 내역 및 주요 목표 진행 상황 대시보드를 시각화합니다.")

    priority_schema = {
        "type": "json_schema",
        "name": "priority",
        "schema": {
            "type": "object",
            "properties": {
                "업무목록": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "업무명": {"type": "string"},
                            "기한": {"type": ["string", "null"]},
                            "긴급도": {"type": "string", "enum": ["높음", "낮음"]},
                            "중요도": {"type": "string", "enum": ["높음", "낮음"]},
                            "근거": {"type": "string"}
                        },
                        "required": ["업무명", "기한", "긴급도", "중요도", "근거"],
                        "additionalProperties": False
                    }
                },
                "참석자별업무": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "이름": {"type": "string"},
                            "업무수": {"type": "integer"}
                        },
                        "required": ["이름", "업무수"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["업무목록", "참석자별업무"],
            "additionalProperties": False
        }
    }

    priority_template = """
다음 회의록을 분석하세요.
1. 김 대리에게 배정된 업무를 모두 추출하세요 (기한, 긴급도, 중요도, 상세 근거).
2. 참석자별 배정 업무 수를 세세요.

[회의록]
{meeting}
"""
    def classify(urgent, important):
        if urgent == "높음" and important == "높음": return "즉시 착수"
        elif important == "높음": return "순차 진행"
        else: return "위임 검토"

    if st.button("우선순위 분석", type="primary"):
        if not current_meeting:
            st.warning("먼저 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("업무를 분석하고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": priority_template.format(meeting=current_meeting)}],
                        text={"format": priority_schema},   # 정해진 JSON 형식으로만 답변
                        temperature=0
                    )
                    data = json.loads(res.output_text)   # JSON 문자열 → 딕셔너리
                    st.session_state[f"priority_data_{st.session_state.active_doc}"] = data # 주석 추가가ㅏㅏ
            except Exception as e:
                st.error(f"분석 오류: {e}")

                # ── ① 업무 집중도 (R&R 밸런스) ──────────
            pdata = st.session_state.get(f"priority_data_{st.session_state.active_doc}")
            if pdata:
                df_people = pd.DataFrame(data["참석자별업무"])
                kim = df_people[df_people["이름"].str.contains("김")]["업무수"].sum() if not df_people.empty else 0
                total = df_people["업무수"].sum() if not df_people.empty else 1

                st.markdown("#### ⚖️ 업무 집중도")
                col1, col2 = st.columns(2)
                col1.metric("김 대리 배정 업무", f"{kim}건")
                col2.metric("전체 배정 업무 중 비중",
                            f"{kim / total * 100:.0f}%" if total else "-"# if total else 삭제할까 말까...?)

                # 김 대리 업무가 3건 이상이거나 절반 이상이면 경보
                if kim >= 3 or (total and kim / total >= 0.5):
                    st.error("🚨 김 대리에게 업무가 집중되어 있습니다. 우선순위와 위임을 검토하세요.")
                else:
                    st.success("업무 배분이 적정한 수준입니다.")

                st.bar_chart(df_people.set_index("이름")["업무수"])

                # ── ② 우선순위 매트릭스 ──────────────────
                df_task = pd.DataFrame(data["업무목록"])
                df_task["기한"] = df_task["기한"].fillna("기한 미정").replace("null", "기한 미정")
                df_task["분류"] = df_task.apply(
                    lambda r: classify(r["긴급도"], r["중요도"]), axis=1)

                st.markdown("#### 🗂️ 우선순위 가이드")
                cols = st.columns(3)
                for col, group, icon in zip(cols,
                                            ["즉시 착수", "순차 진행", "위임 검토"],
                                            ["🔴", "🟡", "🔵"]):
                    col.markdown(f"**{icon} {group}**")
                    tasks = df_task[df_task["분류"] == group]
                    if tasks.empty:
                        col.caption("해당 업무 없음")
                    for _, t in tasks.iterrows():
                        col.info(f"**{t['업무명']}**\n\n⏰ {t['기한']}")

                # ── ③ 상세 분석 표 ───────────────────────
                st.markdown("#### 📋 상세 분석 표")
                st.dataframe(df_task[["분류", "업무명", "기한", "긴급도", "중요도", "근거"]],
                             column_config={
                                 "근거": st.column_config.TextColumn("분석 근거 및 배경", width="large"),
                                 "업무명": st.column_config.TextColumn("배정 업무명", width="medium")
                             },
                            use_container_width=True,
                            hide_index=True
                )
                st.caption("※ 중요도는 회의 내용을 바탕으로 한 에이블의 판단입니다.")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

                            
    # ── ④ 시각화 대시보드: 지출 내역 & 목표 진행 상황] ──
    st.divider()
    st.markdown("### 📈 마케팅 지출 내역 및 목표 진행 상황 대시보드")
    
    col_dash1, col_dash2 = st.columns([1, 1], gap="large")

    with col_dash1:
        st.markdown("#### 💰 마케팅 예산 지출 내역 (총 5,000만 원)")
        st.caption("회의록에 명시된 5천만 원 예산 배분 구성")
        
        # 지출 데이터프레임
        df_expense = pd.DataFrame({
            "항목": ["SNS 광고 (인스타/유튜브)", "인플루언서 협업", "숏폼/카드뉴스 제작", "출시 프로모션 이벤트", "예비비"],
            "금액(백만원)": [25.0, 10.0, 8.0, 5.0, 2.0]
        })
        
        # Streamlit 내장 도넛/파이 대용 시각화 및 지표
        c_tot1, c_tot2 = st.columns([1, 2])
        c_tot1.metric("총 집행 예산", "₩50M", delta="전체 100%")
        c_tot2.dataframe(df_expense, hide_index=True, use_container_width=True)
        st.bar_chart(df_expense.set_index("항목")["금액(백만원)"])

    with col_dash2:
        st.markdown("#### 🎯 김 대리 목표 진행 상황 (목표 달성률 %)")
        st.caption("배정된 3대 핵심 과업 및 출시 일정 기준 진행 현황")
        
        df_goals = pd.DataFrame({
            "핵심 목표 및 마일스톤": [
                "경쟁사 A사 온라인 마케팅 사례 분석",
                "신제품 기능 강조 숏폼/후기 초안 작성",
                "중급 인플루언서 3인 후보군 섭외 조사",
                "출시 2주 전 티저 광고 개시 준비",
                "출시 기념 할인 & SNS 공유 프로모션"
            ],
            "진행률(%)": [85, 60, 45, 30, 20]
        })
        
        # 수평형 바 차트 형태로 진행률 시각화
        st.dataframe(
            df_goals,
            column_config={
                "진행률(%)": st.column_config.ProgressColumn(
                    "진행 상황 (%)",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                ),
                "핵심 목표 및 마일스톤": st.column_config.TextColumn("추진 과제", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# ── 사이드바 아래쪽: 회의록 상태 · 저장된 대화 ──
with st.sidebar:
    status = "✅" if st.session_state.meeting_text else "⬜"
    st.subheader(f"📋 히스토리 관리 {status}")   # ✅: 회의록 준비됨 / ⬜: 회의록 없음
    # 저장된 대화 수 (st.metric보다 한 단계 작은 글자)
    st.markdown(
        f"<p style='font-size:0.875rem; margin:0;'>저장된 대화</p>"
        f"<p style='font-size:1.75rem; margin:0;'>{len(st.session_state.messages) // 2}건</p>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# 메뉴 3: 마감 캘린더 (앱 안 달력 + .ics 다운로드)
# 회의록 / 문서별 일정이 누적 병합되고, 다른 회의록을 추가해도 일정 중복을 방지합니다.
#  - 이 셀은 app.py 끝에 "이어 붙이기"(-a) 합니다.
#  - ⚠️ app.py 셀(%%writefile app.py)을 실행한 직후, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import calendar as pycal

# 오늘 날짜는 한국 시간 기준 (Streamlit Cloud 서버는 UTC라 date.today()를 쓰면 오전 9시 전엔 하루 전 날짜가 됨)
def kst_today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()

# ── ① 일정 추출용 출력 스키마 (05 실습 JSON 구조화) ──
    schedule_schema = {
        "type": "json_schema",
        "name": "schedule",
        "schema": {
            "type": "object",
            "properties": {
                "김대리업무": {                      # 김 대리 업무 목록 (기한 없어도 포함)
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "일정명": {"type": "string"},
                            "기한표현": {"type": ["string", "null"]},
                            "기한일수": {"type": ["integer", "null"]}
                        },
                        "required": ["일정명", "기한표현", "기한일수"],
                        "additionalProperties": False
                    }
                },
                "주요일정": {                        # 출시일 같은 전체 일정
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "일정명": {"type": "string"},
                            "기한표현": {"type": ["string", "null"]},
                            "기한일수": {"type": ["integer", "null"]}
                        },
                        "required": ["일정명", "기한표현", "기한일수"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["김대리업무", "주요일정"],
            "additionalProperties": False
        }
    }

    schedule_template = """
회의록에서 김대리업무와 주요일정을 추출하세요.

[김대리업무]
- 김 대리에게 배정된 업무를 업무 하나당 한 항목씩 빠짐없이 모두 적으세요.
- 기한이 없는 업무도 반드시 포함하세요. (기한표현, 기한일수는 null)
- 일정명은 회의록의 구체적인 업무명으로 쓰세요.
- 보고서 제출처럼 업무의 결과물 기한은 별도 항목으로 만들지 말고, 그 업무의 기한으로 적으세요.

[주요일정]
- 신제품 출시일처럼 회의에서 시점이 정해진 전체 일정만 적으세요.
- 다음 회의, 진행 상황 점검처럼 회의 운영 일정은 제외하세요.
- 김대리업무에 적은 업무는 다시 적지 마세요.

[기한 규칙]
- 기한표현: 회의록에 나온 시한 표현을 그대로 적고, 없으면 null로 하세요.
- 기한일수: 회의 날로부터 며칠 뒤인지 정수로 적으세요.
  '다음 주까지'는 7, '2주 안에'는 14처럼 회의 날 기준으로 정할 수 있을 때만 적습니다.
  '다음 달 초'처럼 정하기 어렵거나, '출시 2주 전'처럼 다른 일정을 기준으로 한 시점이거나,
  기한이 없으면 null로 하세요.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# ── ② 날짜 계산 (계산은 파이썬이 담당 → 결과가 항상 일관됨) ──
def calc_due(base, expr, days):
    """기준일 + 기한일수로 마감일 계산. '다음 달'이면 다음 달 1일. 못 정하면 None"""
    if days is not None:
        return base + timedelta(days=days)
    if expr and "다음 달" in expr:
        return (base.replace(day=1) + timedelta(days=32)).replace(day=1)
    return None

def make_milestones(name, due, base):
    """마감일 기준 D-14, D-7, D-1, D-day 생성 (기준일보다 이전 날짜는 제외)"""
    rows = []
    for label, before in [("D-14", 14), ("D-7", 7), ("D-1", 1), ("D-day", 0)]:
        day = due - timedelta(days=before)
        if day >= base:
            rows.append({"날짜": day, "단계": label, "일정명": name,
                         "제목": f"[{label}] {name}"})
    return rows

# ── ③ 캘린더 파일(.ics) 만들기 → 어떤 캘린더에서도 열면 바로 등록 ──
def make_ics(rows):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Able//Action Scheduler//KO"]
    for i, r in enumerate(rows):
        lines += ["BEGIN:VEVENT",
                  f"UID:able-{r['날짜']:%Y%m%d}-{i}@able",
                  f"DTSTAMP:{date.today():%Y%m%d}T000000Z",
                  f"DTSTART;VALUE=DATE:{r['날짜']:%Y%m%d}",
                  f"DTEND;VALUE=DATE:{r['날짜'] + timedelta(days=1):%Y%m%d}",
                  f"SUMMARY:{r['제목']}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)

# ── ④ 업무 상태 · 남은 기간 표시 (달력 색상과 정렬용) # 동글뱅이 4~5번. 코드 수정 안해도되는지 검토ㄱ) ──
def task_status(due, done, today):
    """(아이콘, 상태, 정렬 순서) 반환: 완료 → 기한 경과 → 마감 임박 → 예정"""
    if done:
        return "✅", "완료", 3
    if due < today:
        return "🔴", "기한 경과", 0
    if (due - today).days <= 3:
        return "🟠", "마감 임박", 1
    return "🔵", "예정", 2

def deadline_label(due, today):
    """D-3, 오늘 마감, 2일 경과처럼 남은 기간을 글로 표시"""
    days_left = (due - today).days
    if days_left < 0:
        return f"{-days_left}일 경과"
    return "오늘 마감" if days_left == 0 else f"D-{days_left}"

def sort_tasks(items, today):
    """미완료·급한 업무가 먼저 오도록 정렬"""
    return sorted(items, key=lambda it: (task_status(it["due"], it["done"], today)[2], it["name"]))

# ── 달력 버튼 동작 (on_click 콜백: 화면을 다시 그리기 전에 실행되어 한 번에 반영) ──
def move_month(offset):
    view = st.session_state.cal_view
    year, month0 = divmod(view.year * 12 + view.month - 1 + offset, 12)   # 연도가 바뀌어도 정확
    st.session_state.cal_view = date(year, month0 + 1, 1)

def go_today():
    st.session_state.cal_view = kst_today().replace(day=1)
    st.session_state.cal_selected = kst_today()

def select_day(day):
    st.session_state.cal_selected = day

# ── ⑤ 세션 상태 초기화 (메뉴를 옮겨도 일정·달력 위치 유지) ── (각각 주석 달기?!!!!)
if "messages_dict" not in st.session_state:
    st.session_state.messages_dict = {}       # {문서명: [메시지들]} 형태로 문서별 대화 분리
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}       # {문서명: 본문내용}
if "archived_meetings" not in st.session_state:
    st.session_state.archived_meetings = []   # [{"시간": ..., "제목": ..., "내용": ...}] 삭제되어도 계속 유지
if "active_doc" not in st.session_state:
    st.session_state.active_doc = ""
if "notice" not in st.session_state:
    st.session_state.notice = None
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
    st.session_state.play_audio = False
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "sched_ver" not in st.session_state:
    st.session_state.sched_ver = 0
if "cal_base" not in st.session_state:
    st.session_state.cal_base = kst_today()
if "cal_view" not in st.session_state:
    st.session_state.cal_view = kst_today().replace(day=1)
if "cal_selected" not in st.session_state:
    st.session_state.cal_selected = kst_today()
if "followups" not in st.session_state:
    st.session_state.followups = []
if "fu_ver" not in st.session_state:
    st.session_state.fu_ver = 0
if "briefing" not in st.session_state:
    st.session_state.briefing = None

# ── ⑥ 화면 ───────────────────────────────────────
elif page == PAGES[2]:
    st.subheader("📅 업무 캘린더")
    st.caption("회의록을 바꿔가며 일정을 추출해도, 기존 일정을 날리지 않고 중복 없이 누적하여 캘린더에 병합합니다.")

    # ① 일정 추출
    c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
    st.session_state.cal_base = c1.date_input("회의 날짜 (기한 계산 기준일)",
                                              value=st.session_state.cal_base)
    base_day = st.session_state.cal_base

    if c2.button("f"🔍 '{st.session_state.active_doc}'에서 일정 추출", type="primary", width="stretch"):
        if not current_meeting:
            st.warning("먼저 사이드바에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("신규 일정 추출 및 기존 캘린더와 병합 중.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": schedule_template.format(meeting=current_meeting)}],
                        text={"format": schedule_schema},
                        temperature=0
                    )
                    data = json.loads(response.output_text)

                existing_keys = {(it["name"], it["due"]) for it in st.session_state.schedule}
                added_count = 0

                for kind, items in [("김 대리 업무", data["김대리업무"]), ("주요 일정", data["주요일정"])]:
                    for item in items:
                        # 구분 이름만 있는 빈 항목은 건너뛰기 (AI 출력 검증)
                        if item["일정명"] in ("김 대리 업무", "주요 일정"):
                            continue
                        # "null" 글자도 기한 없음으로 처리
                        expr = item["기한표현"] if item["기한표현"] not in (None, "", "null") else None
                        days = item["기한일수"]
                        # '출시 2주 전'처럼 다른 일정 기준 시점은 AI가 계산한 일수를 쓰지 않음
                        if expr and re.search(r"\d+\s*(주|일)\s*전", expr):
                            days = None
                        due = calc_due(base_day, expr, days) or (base_day + timedelta(days=7))
                        schedule.append({"name": item["일정명"], "kind": kind, "expr": expr,
                                         "due": due or base_day + timedelta(days=7),
                        
                        if (name, due) not in existing_keys:
                            st.session_state.schedule.append({
                                "name": name, "kind": kind, "expr": expr, "due": due,
                                "use": True, "done": False, "source": st.session_state.active_doc
                            })
                            existing_keys.add((name, due))
                            added_count += 1

                st.session_state.sched_ver += 1
                st.success(f"새로운 일정 {added_count}건이 중복 없이 캘린더에 누적 추가되었습니다!")
            except Exception as e:
                st.error(f"오류: {e}")

    schedule = st.session_state.schedule
    ver = st.session_state.sched_ver


    if not schedule:
        st.info("회의록을 업로드한 뒤 '회의록에서 일정 추출'을 눌러 주세요.")
    else:
        # ② 일정 확인 및 수정 (고친 값은 일정 목록에 바로 저장 → 메뉴를 옮겨도 유지)
        with st.expander("✏️ 일정 확인 및 수정 (날짜 정하기 · 완료 표시)"):
            st.caption("기한이 없거나 애매한 일정은 날짜를 정한 뒤 '달력 표시'를 체크하세요.")
            for i, it in enumerate(schedule):
                a, b, c, d = st.columns([1, 3, 2, 1], vertical_alignment="center")
                it["use"] = a.checkbox("달력 표시", value=it["use"], key=f"use_{ver}_{i}")
                b.markdown(f"**{it['name']}** ({it['kind']})  \n<small>출처: {it.get('source', '알 수 없음')}</small>", unsafe_allow_html=True)
                it["due"] = c.date_input("마감일", value=it["due"], key=f"due_{ver}_{i}")
                it["done"] = d.checkbox("완료", value=it["done"], key=f"done_{ver}_{i}")

        today = kst_today()
        active = [it for it in schedule if it["use"]]

        # 날짜별 마감 업무 / 마일스톤 모으기
        due_map, ms_map, rows = {}, {}, []
        for it in active:
            due_map.setdefault(it["due"], []).append(it)
            for r in make_milestones(it["name"], it["due"], base_day):
                rows.append(r)
                if r["단계"] != "D-day":
                    ms_map.setdefault(r["날짜"], []).append(r)

        # ③ 달력 위쪽: 달 이동
        st.divider()
        view = st.session_state.cal_view
        n1, n2, n3, n4 = st.columns([1, 3, 1, 1], vertical_alignment="center")
        if n1.button("◀ 이전 달", use_container_width=True):
            y, m = (view.year, view.month - 1) if view.month > 1 else (view.year - 1, 12)
            st.session_state.cal_view = date(y, m, 1)
            st.rerun()
        n2.markdown(f"### {view.year}년 {view.month}월")
        if n3.button("다음 달 ▶", use_container_width=True):
            y, m = (view.year, view.month + 1) if view.month < 12 else (view.year + 1, 1)
            st.session_state.cal_view = date(y, m, 1)
            st.rerun()
        if n4.button("오늘", use_container_width=True):
            st.session_state.cal_view = today.replace(day=1)
            st.session_state.cal_selected = today
            st.rerun()
        st.caption("🔴 기한 경과 · 🟠 마감 임박(3일 이내) · 🔵 예정 · ✅ 완료 · ⚪ 기한 미정 · ⏰ 마일스톤(D-14, D-7, D-1)")

        # ④ 달력: 요일 머리글 + 날짜 칸 (월요일 시작)
        head = st.columns(7)
        for col, w in zip(head, ["월", "화", "수", "목", "금", "토", "일"]):
            col.markdown(f"**{w}**")

        for week in pycal.Calendar(firstweekday=0).monthdayscalendar(view.year, view.month):
            cols = st.columns(7)
            for col, d in zip(cols, week):
                if d == 0:           # 이번 달이 아닌 빈 칸
                    continue
                day = date(view.year, view.month, d)
                with col.container(border=True):
                    label = f"{d}일 · 오늘" if day == today else f"{d}일"
                    selected = day == st.session_state.cal_selected
                    if st.button(label, key=f"cal_{day}", , use_container_width=True, type="primary" if selected else "secondary"):
                        st.session_state.cal_selected = day
                        st.rerun()
                    for it in due_map.get(day, [])[:2]:
                        s_icon = "✅" if it["done"] else "🔴"
                        st.caption(f"{s_icon} {it['name'][:9]}")

                    # 칸이 너무 길어지지 않게 마감 업무는 2건까지, 제목은 14자까지만 #(수정 : 삭제할지 말지ㅠ)
                    items = sort_tasks(due_map.get(day, []), today)
                    if items:
                        st.caption(f"마감 업무 {len(items)}건")
                        for it in items[:2]:
                            icon = task_status(it["due"], it["done"], today)[0]
                            name = it["name"] if len(it["name"]) <= 14 else it["name"][:14] + "…"
                            st.caption(f"{icon} {name}")
                        if len(items) > 2:
                            st.caption(f"외 {len(items) - 2}건")
                    ms = ms_map.get(day, [])
                    if ms:
                        st.caption(f"⏰ 알림 {len(ms)}건 ({', '.join(r['단계'] for r in ms)})")

        # ⑤ 선택한 날짜의 마감 업무
        sel = st.session_state.cal_selected
        st.markdown(f"#### 📌 {sel.month}월 {sel.day}일 마감 업무")
        items = sort_tasks(due_map.get(sel, []), today) ### items = due_map.get(sel, []) (이걸로 수정 안해도 될지)
        if not items:
            st.info("선택한 날짜에 마감하는 업무가 없습니다.")
        for it in items:
            icon, status, _ = task_status(it["due"], it["done"], today)
            with st.container(border=True):
                st.markdown(f"**{it['name']}**")
                st.caption(f"{icon} {status} · {it['kind']} · {deadline_label(it['due'], today)}")
                st.caption(f"기한 근거(회의 표현): {it['expr'] or '회의록에 기한 표현 없음 · 직접 지정'}")
        for r in ms_map.get(sel, []):
            st.caption(f"⏰ {r['단계']} 알림: {r['일정명']} (마감 {r['날짜'] + timedelta(days=int(r['단계'][2:]))})")
        if items:
            st.caption("완료 여부는 위의 '일정 확인 및 수정'에서 변경할 수 있습니다.")

        # ⑥ 기한 미정 업무
        undecided = [it for it in schedule if not it["use"]]
        with st.expander(f"⚪ 기한 미정 업무 {len(undecided)}건"):
            if not undecided:
                st.write("기한 미정 업무가 없습니다.")
            for it in undecided:
                st.markdown(f"**{'✅ 완료' if it['done'] else '⬜ 미완료'} · {it['name']}**")
                st.caption(f"{it['kind']} · 기한 근거: {it['expr'] or '회의록에 기록 없음'} · "
                           "'일정 확인 및 수정'에서 날짜를 정하고 '달력 표시'를 체크하면 달력에 나타납니다.")

        # ⑦ 캘린더 파일 다운로드 (Google·Outlook·휴대폰 캘린더에서 열면 등록됨)
        st.divider()
        if rows:
            rows.sort(key=lambda r: r["날짜"])
            st.download_button("📥 캘린더 파일(.ics) 다운로드", data=make_ics(rows),
                               file_name="able_schedule.ics", mime="text/calendar",
                               type="primary", use_container_width="True")
            st.caption(f"마일스톤 {len(rows)}개(D-14, D-7, D-1, D-day)가 담깁니다. "
                       "파일을 열면 Google·Outlook·휴대폰 캘린더에 바로 등록됩니다.")
        else:
            st.info("달력에 표시할 일정을 체크하면 캘린더 파일을 받을 수 있습니다.")

# ══════════════════════════════════════════════════════════════
# 메뉴 4: 팔로업 메일 초안 자동 생성 & 원클릭 승인
#  - ⚠️ app.py 셀 → 캘린더 셀 다음에, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════

# ── ① 팔로업 요청 추출 스키마 (05 실습 JSON 구조화) ──
followup_schema = {
    "type": "json_schema",
    "name": "followup",
    "schema": {
        "type": "object",
        "properties": {
            "요청목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "담당자": {"type": "string"},      # 메일 받을 회의 참석자
                        "요청자료": {"type": "string"},    # 요청할 자료·정보
                        "관련업무": {"type": "string"},    # 김 대리의 어떤 업무에 필요한지
                        "근거": {"type": "string"},        # 회의록 속 근거 발언
                        "메일제목": {"type": "string"},
                        "메일본문": {"type": "string"}
                    },
                    "required": ["담당자", "요청자료", "관련업무", "근거", "메일제목", "메일본문"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["요청목록"],
        "additionalProperties": False
    }
}

followup_template = """
당신은 김 대리의 비서 에이블입니다. 회의록을 읽고, 김 대리가 배정된 업무를 시작하기 위해
다른 참석자에게 받아야 할 선행 자료나 정보를 찾아 요청 메일 초안을 작성하세요.

[찾는 기준]
- 참석자가 김 대리에게 넘겨주기로 한 자료, 또는 회의에서 그 참석자가 언급해 가장 잘 알고 있는 정보
- 김 대리 업무 수행에 실제로 필요한 것만, 최대 3건
- 근거에는 회의록에서 그 참석자가 한 발언을 그대로 적으세요.
- 회의록에 근거가 없는 요청은 만들지 마세요.

[메일 작성 규칙]
- 김 대리가 해당 참석자에게 보내는 메일입니다. 받는 사람 직함에 '님'을 붙여 정중하게 쓰세요.
- 무엇이, 왜(어떤 업무에 필요한지), 언제까지 필요한지를 포함하세요. 기한이 회의록에 없으면 '가능한 일정'을 여쭙는 표현으로 쓰세요.
- 마크다운 기호 없이 일반 텍스트로, 5~8문장 이내로 쓰세요.
- 마지막은 '김 대리 드림'으로 끝내세요.

[회의록]
{meeting}
"""

# ── ② 화면 ──
if "followups" not in st.session_state:
    st.session_state.followups = []      # 생성한 초안 목록 (메뉴를 옮겨도 유지)
    st.session_state.fu_ver = 0          # 새로 만들 때마다 +1 (입력칸 초기화용)

if page == PAGES[3]:
    st.subheader("📨 팔로업 메일 초안")
    st.caption("회의에서 동료에게 받아야 할 자료를 찾아 요청 메일 초안을 만들고, 승인하면 발송합니다. 발송 완료 후에도 내용을 수정하여 같은 담당자나 다른 수신자에게 자유롭게 추가 발송할 수 있습니다.")

    if st.button("초안 만들기", key="fu_make", type="primary"):
        if not current_meeting:
            st.warning("먼저 사이드바에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("필요한 자료를 찾고 메일 초안을 쓰고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": followup_template.format(meeting=current_meeting)}],
                        text={"format": followup_schema},
                        temperature=0
                    )
                    data = json.loads(response.output_text)

                # 초안마다 받는 사람(시연용 기본값: 김 대리 주소)과 발송 여부 추가
                st.session_state.followups = [
                    {**d, "to": MY_EMAIL, "sent": None} for d in data["요청목록"]
                ]
                st.session_state.fu_ver += 1
                if not data["요청목록"]:
                    st.info("회의록에서 요청할 선행 자료를 찾지 못했습니다.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 초안마다: 근거 확인 → 수정 → 승인 버튼 (고친 내용은 초안 목록에 바로 저장)
    ver = st.session_state.fu_ver
    for i, d in enumerate(st.session_state.followups):
        st.divider()
        st.markdown(f"**📌 {d['담당자']}님께 '{d['요청자료']}' 요청 메일 초안을 작성했습니다. 발송할까요?**")
        st.caption(f"관련 업무: {d['관련업무']}  \n회의 근거: \"{d['근거']}\"")

        d["to"] = st.text_input("받는 사람", value=d["to"], key=f"fu_to_{ver}_{i}",
                                help="시연용으로 김 대리 주소가 들어 있습니다. 실제 동료 주소로 바꿔 보내세요.")
        d["메일제목"] = st.text_input("제목", value=d["메일제목"], key=f"fu_sub_{ver}_{i}")
        d["메일본문"] = st.text_area("본문 (수정 가능)", value=d["메일본문"], height=200,
                                  key=f"fu_body_{ver}_{i}")
        
        col_s1, col_s2 = st.columns([1, 4])
        with col_s1:
            send_now = st.button("🚀 이메일 전송", key=f"fu_send_{ver}_{i}", type="primary")

        if d["sent"]: #수정? : 이 바로 3줄 삭제하고, if send_now:를 넣을지 고민
            st.success(f"📧 {d['sent']}(으)로 발송했습니다.")
        elif st.button("✅ 승인하고 발송", key=f"fu_send_{ver}_{i}"):
            if "@" not in d["to"]:
                st.warning("받는 사람 이메일 주소를 확인해 주세요.")
            else:
                with st.spinner("메일을 보내고 있습니다.."):
                    result = send_email(d["to"], d["메일제목"], d["메일본문"])   # 미션 1 함수 재사용
                if result["status"] == "success":
                    t_str = datetime.now().strftime("%H:%M:%S")
                    d.setdefault("logs", []).append(f"[{t_str}] 수신처: {d['to']} (발송 성공)")
                    st.success("✅ 메일이 성공적으로 발송되었습니다! (필요시 내용을 수정해 다시 전송할 수 있습니다)")
                else:
                    st.error(f"발송 실패: {res.get('error')}")

        if d.get("logs"):
            with st.expander("이 메일의 발송 이력"):
                for log in d["logs"]:
                    st.caption(log)

# ══════════════════════════════════════════════════════════════
# 메뉴 5: 1분 출근길 오디오 브리핑
#  - ⚠️ 팔로업 메일 셀 다음에, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════

briefing_role = """
당신은 김 대리의 비서 에이블이며, 출근길에 이어폰으로 듣는 1분 라디오 브리핑을 진행합니다.

[형식]
- 뉴스 앵커처럼 또렷하고 경쾌한 말투로, 말하듯 자연스러운 문장으로 씁니다.
- 마크다운 기호, 목록, 제목, 괄호를 쓰지 않습니다.
- 전체 250~300자 이내로 씁니다. (읽으면 약 1분)

[순서]
1. 짧은 인사: "김 대리님, 출근길 브리핑입니다."
2. 회의 결론 한두 문장: 무엇을 하기로 했는지
3. 먼저 챙길 일: 김 대리 업무를 기한이 가까운 순서로, 업무 개수를 먼저 말하고 '첫째, 둘째'로
4. 짧은 마무리 인사

[주의]
- 회의록에 없는 내용은 지어내지 않습니다.
- 기한은 회의에서 명시된 표현 그대로 말하고, 없으면 '기한은 정해지지 않았습니다'라고 말합니다.
"""

# ── 화면 ──
if "briefing" not in st.session_state:
    st.session_state.briefing = None      # {"script": 원고, "audio": 음성} 저장 (메뉴를 옮겨도 유지)

elif page == PAGES[4]:
    st.subheader(f"🎧 1분 출근길 브리핑 ({st.session_state.active_doc})")
    st.caption("회의 결론과 먼저 챙길 일만 1분 안팎의 음성으로 들려 드립니다.")

    voice = st.selectbox("목소리", ["nova", "alloy", "shimmer", "onyx"], key="br_voice")

    if st.button("브리핑 생성", type="primary"): # 수정 : key="br_make" 삭제해도 되겠지...?
        if not current_meeting:
            st.warning("먼저 사이드바에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("브리핑 원고를 쓰고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "system", "content": 김 대리의 비서 에이블로서 출근길 1분 라디오 뉴스 브리핑을 350자 내외로 정중하고 또렷하게 작성하세요."},
                               {"role": "user", "content": f"[회의록]\n{meeting_text}"}],
                        temperature=0
                    )
                    script = response.output_text

                with st.spinner("음성으로 변환하고 있습니다.."):
                    speech = client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=script,
                        instructions="라디오 뉴스 앵커처럼 또렷하고 경쾌하게, 조금 빠른 속도로 읽어 주세요."
                    )

                st.session_state.briefing = {"script": script, "audio": speech.content}
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 만든 브리핑이 있으면 재생 + 다운로드 + 원고
    if st.session_state.briefing:
        b = st.session_state.briefing
        st.audio(b["audio"], format="audio/mp3")
        sec = round(len(b["script"]) / 5.5)   # 한국어 약 5~6자/초 기준 예상 길이
        st.caption(f"원고 {len(b['script'])}자 · 예상 재생 시간 약 {sec}초")

        st.download_button("📥 MP3로 저장 (이어폰으로 듣기)", data=b["audio"],
                           file_name="able_briefing.mp3", mime="audio/mpeg")
        with st.expander("📄 브리핑 원고 보기"):
            st.write(b["script"])

# ══════════════════════════════════════════════════════════════
# 메뉴 6: 🍃 오늘의 할 일 (김 대리 업무 체크리스트)
#  - ⚠️ 출근길 브리핑 셀 다음에, 이 셀을 한 번만 실행하세요.
#  - 업무 목록은 '📅 마감 캘린더'에서 추출한 일정(st.session_state.schedule)을 그대로 사용
#  - 여기서 완료를 체크하면 캘린더의 완료 표시(✅)에도 바로 반영됨
# ══════════════════════════════════════════════════════════════

def today_group(item, today):
    """업무를 마감 상황별로 묶기"""
    if not item.get("use") or not item.get("due"):
        return "🗓️ 기한 확인 필요"
    if item["due"] < today:
        return "⚠️ 기한 경과"
    if item["due"] == today:
        return "🔥 오늘 마감"
    return "📋 앞으로 할 일"

def save_today_done(index, key):
    """체크박스를 바꾸면 일정 목록의 완료 여부를 저장"""
    st.session_state.schedule[index]["done"] = st.session_state[key]
    st.session_state.sched_ver += 1    # 캘린더의 완료 체크칸도 새 값으로 다시 그리기

elif page == PAGES[5]:
    st.subheader("🍃 오늘의 할 일")
    today = kst_today()                 # 캘린더 셀의 한국 시간 기준 오늘
    st.caption(f"{today:%Y.%m.%d} · 김 대리님의 업무")

    # 김 대리 업무만 골라내기 (번호 i는 일정 목록에서의 위치 → 완료 저장에 사용)
    tasks = [(i, it) for i, it in enumerate(st.session_state.schedule)
             if it["kind"] == "김 대리 업무"]

    if not tasks:
        st.info("'📅 마감 캘린더' 메뉴에서 먼저 일정을 추출해 주세요.")
    else:
        pending = [(i, it) for i, it in tasks if not it["done"]]
        done = [(i, it) for i, it in tasks if it["done"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("남은 업무", f"{len(pending)}건")
        c2.metric("오늘 마감", f"{sum(today_group(it, today) == '🔥 오늘 마감' for _, it in pending)}건")
        c3.metric("완료", f"{len(done)}건")

        ver = st.session_state.sched_ver
        for group in ["⚠️ 기한 경과", "🔥 오늘 마감", "📋 앞으로 할 일", "🗓️ 기한 확인 필요"]:
            items = [(i, it) for i, it in pending if today_group(it, today) == group]
            if not items:
                continue
            st.markdown(f"#### {group}")
            items.sort(key=lambda pair: pair[1]["due"] if pair[1]["use"] else date.max)

            for i, it in items:
                key = f"today_{ver}_{i}"
                with st.container(border=True):
                    st.checkbox(it["name"], value=False, key=key,
                                on_change=save_today_done, args=(i, key))
                    if it["use"]:
                        # deadline_label: 캘린더 셀의 'D-3 / 오늘 마감 / 2일 경과' 함수 재사용
                        st.caption(f"📅 마감일: {it['due']:%Y.%m.%d} · {deadline_label(it['due'], today)}")
                        if it["expr"]:
                            st.caption(f"회의록 표현: {it['expr']}")
                    elif it["expr"]:
                        st.caption(f"🗓️ 기한: {it['expr']} · 정확한 날짜 확인 필요 (캘린더에서 지정)")
                    else:
                        st.caption("🗓️ 기한 미정 · 캘린더의 '일정 확인 및 수정'에서 날짜를 정할 수 있어요")

        if not pending:
            st.success("모든 담당 업무를 완료했어요! 🎉")

        with st.expander(f"✅ 완료한 업무 {len(done)}건"):
            for i, it in done:
                key = f"today_done_{ver}_{i}"
                st.checkbox(it["name"], value=True, key=key,
                            on_change=save_today_done, args=(i, key))
            st.caption("체크를 해제하면 미완료 목록으로 돌아가요.")

# ══════════════════════════════════════════════════
# 메뉴 7: 🗄️ 회의록 아카이브 (날짜/시간별 보존 및 조회)
# ══════════════════════════════════════════════════
elif page == PAGES[6]:
    st.subheader("🗄️ 회의록 영구 아카이브 (날짜/시간별 기록실)")
    st.caption("사이드바 작업 목록에서 회의록을 삭제하거나 교체하더라도, 업로드되었던 모든 회의록 원본과 등록 일시가 이곳에 안전하게 영구 보존됩니다.")

    archives = st.session_state.archived_meetings
    if not archives:
        st.info("아직 저장된 회의록 아카이브가 없습니다.")
    else:
        st.markdown(f"**총 보존 회의록:** `{len(archives)}건`")
        for idx, item in enumerate(reversed(archives), 1):
            with st.expander(f"🕒 [{item['time']}] {item['title']}", expanded=(idx == 1)):
                st.caption(f"등록 시각: {item['time']}")
                st.text_area("회의록 원문", item["content"], height=200, key=f"arc_view_{idx}")
                if st.button("📥 이 문서를 현재 작업 문서로 다시 불러오기", key=f"restore_{idx}"):
                    st.session_state.uploaded_docs[item["title"]] = item["content"]
                    st.session_state.active_doc = item["title"]
                    st.success(f"'{item['title']}' 문서가 활성 작업 목록으로 복원되었습니다!")
                    st.rerun()
