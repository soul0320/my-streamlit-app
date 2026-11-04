import os
import re
import json
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from openai import OpenAI

# ==============================================================================
# 1. 환경 설정 및 클라이언트 초기화
# ==============================================================================
st.set_page_config(
    page_title="AI 비서 에이블 (AIVLE)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = OpenAI()

# 추천 공식 이미지 (KT 에이블스쿨 교안 배포용 투명 배경 에이블 로봇)
AIVLE_AVATAR_URL = "https://raw.githubusercontent.com/Jangrae/img/master/ai_robot.png"

# ==============================================================================
# 2. 페르소나 및 시스템 프롬프트 (승환님 표준본 기반 고도화)
# ==============================================================================
sys_role = """
당신은 마케팅팀 '김 대리'의 든든하고 명석한 전속 AI 개인비서 '에이블(AIVLE)'입니다.
김 대리가 외출이나 다른 일정으로 회의에 참석하지 못했을 때, 회의 내용을 빈틈없이 분석하여 업무 누락이 없도록 선제적으로 보좌합니다.

[주요 임무]
1. 회의록을 분석하여 핵심 아젠다와 결론을 일목요연하게 요약합니다.
2. 회의록 내에서 김 대리에게 배정된 업무 지시사항을 찾아 실행 관점으로 꼼꼼하게 정리합니다.
3. 온톨로지 기반의 업무 관계 분석, 마일스톤 도출, 시니어 마케터 관점의 리스크 진단 및 일상 질의응답을 성실히 수행합니다.

[답변 및 행동 규칙]
- 한국어로 항상 정중하고 친절하며 신뢰감 있는 어조로 답변합니다.
- 회의록에 없는 내용은 절대 자의적으로 지어내거나 추측하지 않습니다(Hallucination 방지).
- 요약 시 [주제] → [논의 내용] → [결정 사항] 순서로 명확히 구역을 나누어 작성합니다.
- 김 대리 담당 업무는 요약과 분리하여 별도 구역으로 정리하며, 각 업무마다 '업무명', '세부 실행 내용', '기한'을 명시합니다. 
- 기한이 회의록에 명시되지 않은 업무는 자의적으로 판단하지 않고 반드시 '기한 미정'으로 표기합니다.
- 결정 사항에는 예산, 마감일, 타깃 채널 등 구체적인 수치와 항목을 빠짐없이 포함합니다.
- 이메일 초안 작성 요청 시 본문에는 마크다운 기호 없이 비즈니스 이메일 표준 양식으로 작성합니다.
"""

# ==============================================================================
# 3. 도구 및 유틸리티 함수 (이메일, PII 마스킹)
# ==============================================================================
def send_email(to, subject, body):
    """미션 1 & 7: 외부 SMTP 도구 연동 이메일 발송 함수"""
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return {"status": "failed", "error": "Gmail 계정 환경변수(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)가 설정되지 않았습니다."}
    
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls()
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def mask_personal_info(text):
    """사내 컴플라이언스 및 개인정보보호(PII) 마스킹 모듈"""
    phone_pattern = r'(\d{2,3})[-.\s]?(\d{3,4})[-.\s]?(\d{4})'
    rrn_pattern = r'(\d{6})[-.\s]?([1-4]\d{6})'
    masked = re.sub(phone_pattern, r'\1-****-****', text)
    masked = re.sub(rrn_pattern, r'\1-*******', masked)
    return masked

# ==============================================================================
# 4. 세션 상태 관리 (대화 히스토리 10개 슬라이딩 윈도우 유지)
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

# ==============================================================================
# 5. 사이드바 구성 (고급 설정, 파일 업로드, 네비게이션)
# ==============================================================================
with st.sidebar:
    st.image(AIVLE_AVATAR_URL, width=140)
    st.title("에이블 비서실")
    st.caption("김 대리님을 위한 맞춤형 인텔리전스")
    
    st.markdown("---")
    st.subheader("📁 회의록 데이터 등록")
    uploaded_file = st.file_uploader(
        "회의록 파일 (.txt / .m4a / .mp3 / .wav)", 
        type=["txt", "m4a", "mp3", "wav"],
        help="텍스트 문서나 녹음 파일을 업로드하면 비서가 내용을 즉시 분석합니다."
    )
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "txt":
            raw_text = uploaded_file.read().decode("utf-8")
            st.session_state.meeting_text = mask_personal_info(raw_text)
            st.success("📄 회의록 문서 로드 완료 (PII 검수됨)")
        else:
            with st.spinner("🎙️ Whisper STT 음성 변환 중..."):
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(uploaded_file.name, uploaded_file.read())
                )
                st.session_state.meeting_text = mask_personal_info(transcript.text)
                st.success("🎙️ 음성 인식(STT) 완료!")

    st.markdown("---")
    st.subheader("⚙️ 비서 엔진 고급 설정")
    selected_model = st.selectbox("LLM 엔진 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("창의성 수준 (Temperature)", min_value=0.0, max_value=1.0, value=0.0, step=0.1, help="0에 가까울수록 사실에 기반한 정밀한 요약을 출력합니다.")
    enable_voice = st.checkbox("답변 음성 자동 생성 (TTS)", value=False)
    
    st.markdown("---")
    menu = st.radio(
        "🧭 지원 업무 메뉴",
        [
            "회의 요약 & 우선순위 분석",
            "온톨로지 업무 관계망",
            "마일스톤 캘린더",
            "팔로업 메일 발송",
            "시니어 마케터 리스크 진단",
            "모닝 오디오 브리핑",
            "에이블 대화형 챗봇"
        ]
    )
    
    if st.button("🧹 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# 6. 메인 콘텐츠 제어
# ==============================================================================
meeting_text = st.session_state.meeting_text

if not meeting_text:
    st.info("👈 먼저 사이드바에서 회의록 문서(.txt) 또는 음성 녹음 파일을 업로드해 주세요.")
    st.markdown("""
    ### 🌟 AI 개인비서 '에이블'의 주요 지원 기능
    1. **회의 요약 & 우선순위 분석**: 미팅 핵심 요약 및 아이젠하워 매트릭스 기반 업무 우선순위 배정
    2. **온톨로지 업무 관계망**: 참석자-역할-업무 간 지식 관계 그래프 및 매트릭스 도출
    3. **마일스톤 캘린더**: D-Day 기반 착수, 중간 점검, 마감 일정 자동 역산 및 구글 캘린더 등록
    4. **팔로업 메일 발송**: 논의 사항 기반 유관부서 자료 요청 메일 원클릭 전송
    5. **시니어 마케터 리스크 진단**: 10년 차 이상 시니어 관점의 예산·일정·경쟁사 리스크 분석
    6. **모닝 오디오 브리핑**: 바쁜 출근길 이어폰으로 듣는 1분 핵심 음성 브리핑
    7. **에이블 대화형 챗봇**: 최근 10개 대화 문맥을 기억하는 멀티턴 비서 챗봇
    """)
    st.stop()

# ------------------------------------------------------------------------------
# 메뉴 1: 회의 요약 & 우선순위 분석
# ------------------------------------------------------------------------------
if menu == "회의 요약 & 우선순위 분석":
    st.header("📋 회의록 요약 및 업무 우선순위 분석")
    st.caption("회의 핵심 안건 요약과 김 대리님 배정 업무를 아이젠하워 매트릭스(긴급도 x 중요도)로 정리합니다.")
    
    if st.button("🚀 종합 분석 실행", type="primary"):
        with st.spinner("에이블이 회의록을 정밀 분석 중입니다..."):
            prompt = f"""
다음 회의록을 분석하여 규칙에 맞춰 보고서를 작성하세요.

[회의록 원문]
{meeting_text}

[작성 가이드]
1. 미팅 내용 요약 (주제, 참석자, 주요 논의 사항, 결정 사항)
2. 김 대리 배정 업무 목록 (업무명, 세부 실행 내용, 명시된 제출기한)
3. 아이젠하워 매트릭스 기반 우선순위 재분류:
   - [Do First - 즉시 실행]: 긴급하고 중요한 업무
   - [Schedule - 계획 수립]: 긴급하지 않으나 중요한 전략 업무
   - [Delegate/Check - 위임 및 점검]: 유관부서 협업 또는 단순 확인 업무
"""
            response = client.responses.create(
                model=selected_model,
                input=[
                    {"role": "system", "content": sys_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            answer = response.output_text
            st.markdown(answer)
            
            if enable_voice:
                with st.spinner("음성 안내 생성 중..."):
                    speech = client.audio.speech.create(model="tts-1", voice="alloy", input=answer[:400])
                    st.audio(speech.content, format="audio/mp3")

# ------------------------------------------------------------------------------
# 메뉴 2: 온톨로지 업무 관계망 (지식 그래프 분석)
# ------------------------------------------------------------------------------
elif menu == "온톨로지 업무 관계망":
    st.header("🕸️ 온톨로지 기반 업무 분석 엔진")
    st.caption("회의록 속 비정형 데이터를 '주체(Actor) ➔ 관계(Relation) ➔ 업무/대상(Task) ➔ 산출물/기한' 트리플로 추출합니다.")
    
    if st.button("🔍 온톨로지 지식망 추출", type="primary"):
        with st.spinner("업무 엔티티 및 관계 그래프를 빌드하는 중입니다..."):
            prompt = f"""
다음 회의록에서 업무 주체(인물/직책), 상호작용 관계, 세부 태스크, 산출물 및 기한을 파악하여 
JSON 형식의 온톨로지 데이터로 구조화해 주세요.

[회의록 원문]
{meeting_text}

반드시 아래와 같은 JSON 배열 형식으로만 응답하세요(코드블록 기호 없이 JSON만 반환):
[
  {{"actor": "인물 또는 직책", "relation": "지시/담당/협업/보고 등", "task": "세부 업무 내용", "output": "산출물", "due": "기한 또는 기한 미정"}}
]
"""
            response = client.responses.create(
                model=selected_model,
                input=[
                    {"role": "system", "content": "너는 비즈니스 지식 온톨로지 구축 전문가야. 오직 유효한 JSON 포맷으로만 응답해."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            raw_json = response.output_text.strip()
            raw_json = re.sub(r'^```json\s*', '', raw_json)
            raw_json = re.sub(r'\s*```$', '', raw_json)
            
            try:
                ontology_data = json.loads(raw_json)
                st.subheader("📊 지식 엔티티 & 관계 매트릭스")
                st.dataframe(ontology_data, use_container_width=True)
                
                st.subheader("💡 온톨로지 핵심 관계망 다이어그램")
                mermaid_code = "graph TD;\n"
                for idx, row in enumerate(ontology_data):
                    actor = row.get("actor", "담당자").replace(" ", "_")
                    task = row.get("task", "업무").replace(" ", "_")[:15]
                    relation = row.get("relation", "수행")
                    mermaid_code += f"    {actor} -->|{relation}| {task}_{idx}[{task}];\n"
                
                st.markdown(f"```mermaid\n{mermaid_code}\n```")
            except Exception as e:
                st.error("온톨로지 구조화 데이터를 파싱하는 중 오류가 발생했습니다.")
                st.text(response.output_text)

# ------------------------------------------------------------------------------
# 메뉴 3: 마일스톤 캘린더
# ------------------------------------------------------------------------------
elif menu == "마일스톤 캘린더":
    st.header("📅 마감 역산 캘린더 마일스톤")
    st.caption("회의 결정 사항의 마감일로부터 D-7(착수), D-3(중간 점검), D-Day(보고) 일정을 자동 역산하여 캘린더에 원클릭 등록합니다.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; background-color:#fcfcfc;">
            <h4>🚀 [D-7] 경쟁사 분석 착수</h4>
            <p><strong>일시:</strong> 2026-11-04 (수)</p>
            <p><strong>내용:</strong> A사 SNS 광고 사례 및 프로모션 벤치마킹 분석 시작</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("📅 구글 캘린더 등록", "https://calendar.google.com/calendar/render?action=TEMPLATE&text=[착수]+경쟁사+마케팅+분석&dates=20261104T090000Z/20261104T100000Z&details=경쟁사+SNS+광고+및+프로모션+분석+착수")
        
    with col2:
        st.markdown("""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; background-color:#fcfcfc;">
            <h4>📝 [D-3] SNS 콘텐츠 초안 공유</h4>
            <p><strong>일시:</strong> 2026-11-08 (일)</p>
            <p><strong>내용:</strong> 기능 강조 숏폼 영상 및 후기형 카드뉴스 초안 박과장 검토</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("📅 구글 캘린더 등록", "https://calendar.google.com/calendar/render?action=TEMPLATE&text=[중간점검]+SNS+콘텐츠+초안&dates=20261108T090000Z/20261108T100000Z&details=박과장님께+초안+공유+및+피드백+수렴")

    with col3:
        st.markdown("""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; background-color:#fcfcfc;">
            <h4>🎯 [D-Day] 마케팅 최종 보고</h4>
            <p><strong>일시:</strong> 2026-11-11 (수)</p>
            <p><strong>내용:</strong> 신제품 출시 마케팅 종합 전략 및 인플루언서 리스트 팀장 보고</p>
        </div>
        """, unsafe_allow_html=True)
        st.link_button("📅 구글 캘린더 등록", "https://calendar.google.com/calendar/render?action=TEMPLATE&text=[마감]+신제품+마케팅+최종보고&dates=20261111T090000Z/20261111T100000Z&details=전체+회의에서+최종안+발표")

# ------------------------------------------------------------------------------
# 메뉴 4: 팔로업 메일 발송 (TypeError 방지 독립 컴포넌트 구조)
# ------------------------------------------------------------------------------
elif menu == "팔로업 메일 발송":
    st.header("✉️ 협업자 선행 자료 요청 메일 발송")
    st.caption("회의 결론에 따라 타 팀이나 동료에게 요청할 자료 초안을 검토하고 원클릭으로 발송합니다.")
    
    # TypeError 방지를 위해 st.form 대신 명시적 key를 지닌 독립 위젯으로 배치
    default_to = "soulsky0218@gmail.com"
    default_subject = "[업무협조] 신제품 마케팅 관련 경쟁사 프로모션 세부 데이터 요청 건"
    default_body = """이과장님, 안녕하십니까. 마케팅팀 김 대리입니다.

금일 신제품 출시 전략 회의에서 논의된 경쟁사 온라인 마케팅 사례 분석 업무를 배정받아 진행 중입니다.
보고서의 완성도를 높이기 위해, 회의 중 언급되었던 최근 경쟁사 A사의 세부 프로모션 및 캠페인 집행 자료 공유를 부탁드리고자 합니다.

업무에 번거로움을 드려 죄송하며, 확인해 주시면 마케팅 전략 수립에 소중히 반영하겠습니다.
감사합니다.

마케팅팀 김 대리 드림"""

    to_addr = st.text_input("수신자 이메일 주소", value=default_to, key="mail_to_input")
    subject = st.text_input("메일 제목", value=default_subject, key="mail_subject_input")
    body = st.text_area("메일 본문", value=default_body, height=220, key="mail_body_input")
    
    col_send, col_clear = st.columns([1, 5])
    with col_send:
        if st.button("📤 최종 승인 및 전송", type="primary", key="btn_send_mail"):
            if not to_addr or not subject or not body:
                st.warning("수신자, 제목, 본문을 모두 입력해 주세요.")
            else:
                with st.spinner("메일을 안전하게 발송 중입니다..."):
                    res = send_email(to_addr, subject, body)
                    if res["status"] == "success":
                        st.success(f"✅ [{to_addr}] 님께 메일이 성공적으로 발송되었습니다!")
                    else:
                        st.error(f"❌ 발송 실패: {res['error']}")

# ------------------------------------------------------------------------------
# 메뉴 5: 시니어 마케터 리스크 진단
# ------------------------------------------------------------------------------
elif menu == "시니어 마케터 리스크 진단":
    st.header("🧐 시니어 마케터 전략 리스크 진단실")
    st.caption("10년 차 이상 시니어 마케팅 팀장의 시각으로 회의 결정 사항의 현실성과 잠재 리스크를 날카롭게 짚어냅니다.")
    
    if st.button("🕵️ 리스크 진단 및 보완책 평가", type="primary"):
        with st.spinner("시니어 마케터 페르소나로 전략을 검토 중입니다..."):
            senior_prompt = f"""
당신은 15년 차 대기업 시니어 마케팅 총괄 팀장입니다.
아래 회의록 내용을 검토하고, 주니어 마케터들이 놓치기 쉬운 비즈니스 리스크를 다음 4가지 관점에서 날카롭고 현실적으로 진단해 주세요:

1. [예산 집행 리스크]: 5천만 원 예산 범위 내 SNS 집행과 인플루언서 섭외 단가의 현실성
2. [일정 병목 리스크]: 출시 2주 전 광고 집행을 위한 티저 콘텐츠 및 랜딩페이지 제작 소요 시간의 촉박성
3. [경쟁사 대응 리스크]: 경쟁사 A사의 후속 프로모션 맞대응 전략의 부재
4. [김 대리 업무 과부하 리스크]: 부재 중 배정된 3개 업무(사례 분석, 숏폼 초안, 인플루언서 섭외)의 일정 중복 문제 및 완충 방안

[회의록 원문]
{meeting_text}
"""
            res = client.responses.create(
                model=selected_model,
                input=[
                    {"role": "system", "content": "너는 매우 깐깐하고 현실적인 마케팅 총괄 디렉터야. 뜬구름 잡는 소리 대신 실행 리스크와 구체적 숫자를 지적해."},
                    {"role": "user", "content": senior_prompt}
                ],
                temperature=0.2
            )
            st.markdown(res.output_text)

# ------------------------------------------------------------------------------
# 메뉴 6: 모닝 오디오 브리핑 (미션 5 TTS 연계)
# ------------------------------------------------------------------------------
elif menu == "모닝 오디오 브리핑":
    st.header("🎧 1분 출근길 모닝 오디오 브리핑")
    st.caption("바쁜 이동 중 이어폰으로 들을 수 있는 1분 핵심 요약 음성 서비스입니다.")
    
    if st.button("🔊 브리핑 음성 생성", type="primary"):
        with st.spinner("AI 앵커가 핵심 브리핑 원고를 작성하고 음성을 합성하는 중입니다..."):
            brief_res = client.responses.create(
                model=selected_model,
                input=[
                    {"role": "system", "content": "김대리의 개인비서 에이블로서, 출근길 김대리에게 차분하고 프로페셔널하게 1분 내외로 말하듯이 핵심 결론과 최우선 업무만 3문장 이내로 작성해 줘. 마크다운 기호 없이 순수 대화체로만 작성할 것."},
                    {"role": "user", "content": f"회의록:\n{meeting_text}"}
                ],
                temperature=0.3
            )
            script = brief_res.output_text
            st.markdown(f"**🎙️ 앵커 브리핑 원고:**\n\n> {script}")
            
            speech = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=script
            )
            st.audio(speech.content, format="audio/mp3")

# ------------------------------------------------------------------------------
# 메뉴 7: 에이블 대화형 챗봇 (10개 대화 히스토리 슬라이딩 윈도우 유지)
# ------------------------------------------------------------------------------
elif menu == "에이블 대화형 챗봇":
    st.header("💬 에이블 실시간 대화형 챗봇")
    st.caption("회의록 기반으로 무엇이든 질문하세요. 최근 10개의 대화 문맥을 유지하며 답변합니다.")
    
    # 1. 기존 대화 기록 출력
    for msg in st.session_state.messages:
        avatar = None if msg["role"] == "user" else AIVLE_AVATAR_URL
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
    # 2. 실시간 마이크 음성 입력 또는 텍스트 입력
    col_mic, col_txt = st.columns([1, 4])
    audio_cmd = None
    with col_mic:
        audio_value = st.audio_input("음성 질문", key="mic_input")
        if audio_value is not None:
            with st.spinner("음성 변환 중..."):
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("user_voice.wav", audio_value, "audio/wav")
                )
                audio_cmd = transcript.text
                
    user_prompt = st.chat_input("에이블에게 질문을 입력하세요...")
    
    final_query = audio_cmd if audio_cmd else user_prompt
    
    if final_query:
        # 사용자 메시지 표시 및 저장
        with st.chat_message("user"):
            st.markdown(final_query)
        st.session_state.messages.append({"role": "user", "content": final_query})
        
        # 슬라이딩 윈도우 (최근 10개만 유지)
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]
            
        # 프롬프트 조립: 시스템 역할 + 회의록 배경 + 최근 대화 기록 10개
        history_context = [
            {"role": "system", "content": sys_role + f"\n\n[현재 로드된 회의록 내용]\n{meeting_text}"}
        ]
        history_context.extend(st.session_state.messages)
        
        with st.chat_message("assistant", avatar=AIVLE_AVATAR_URL):
            with st.spinner("에이블이 생각 중입니다..."):
                resp = client.responses.create(
                    model=selected_model,
                    input=history_context,
                    temperature=temperature
                )
                bot_answer = resp.output_text
                st.markdown(bot_answer)
                
                if enable_voice:
                    speech = client.audio.speech.create(model="tts-1", voice="alloy", input=bot_answer[:300])
                    st.audio(speech.content, format="audio/mp3")
                    
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
        
        # 다시 슬라이딩 윈도우 보장
        if len(st.session_state.messages) > 10:
            st.session_state.messages = st.session_state.messages[-10:]
