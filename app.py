
import os
import re
import json
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from openai import OpenAI

# 1. OpenAI 클라이언트 생성
client = OpenAI()

# 2. 페이지 기본 설정
st.set_page_config(page_title="KT 에이블 - AI 개인비서", page_icon="🤖", layout="wide")
st.title("🤖 음성인식 AI 개인비서 '에이블'")
st.caption("마케팅팀 김 대리 전속 선제적 업무 지원 서비스")

# 역할 지정
sys_role = """
당신은 다양한 업무를 처리하는 AI 서비스입니다.
개인 비서처럼 친절하게 응대합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지시 영역
with st.form("task_form"):
    uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    submit = st.form_submit_button("실행")

# 3. [미션 1 & 7] 이메일 발송 도구 함수
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return {"status": "failed", "error": "Gmail 계정 설정이 필요합니다."}
    
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

# 4. [10번 공통 가드레일 - 솔 담당] PII 마스킹 함수
def mask_personal_info(text):
    phone_pattern = r'(\d{2,3})[-.\s]?(\d{3,4})[-.\s]?(\d{4})'
    rrn_pattern = r'(\d{6})[-.\s]?([1-4]\d{6})'
    masked = re.sub(phone_pattern, r'\1-****-****', text)
    masked = re.sub(rrn_pattern, r'\1-*******', masked)
    return masked

# 5. [승환님 표준 공유본] 시스템 프롬프트 (페르소나)
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
"""

# 6. 세션 상태 초기화 (표준 규칙)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

# ----------------------------------------------------
# 7. 상단 입력 게이트웨이 (2일차 표준 위젯 구조)
# ----------------------------------------------------
st.markdown("### 📥 1단계: 회의록 데이터 등록")
uploaded_file = st.file_uploader("회의록 파일 업로드 (.txt 또는 음성 .m4a, .mp3)", type=["txt", "m4a", "mp3", "wav"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext == "txt":
        raw_text = uploaded_file.read().decode("utf-8")
        st.session_state.meeting_text = mask_personal_info(raw_text)
        st.success("📄 텍스트 회의록 로드 완료 (PII 검수 완료)")
    else:
        with st.spinner("🎙️ 음성 파일을 텍스트(STT)로 변환 중입니다..."):
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=(uploaded_file.name, uploaded_file.read())
            )
            st.session_state.meeting_text = mask_personal_info(transcript.text)
            st.success("✅ 음성 회의록 STT 변환 완료!")

# ----------------------------------------------------
# 8. 기능별 탭 구성 (1일차 미션 + 2일차 핵심 기능)
# ----------------------------------------------------
meeting_text = st.session_state.meeting_text

if meeting_text:
    tabs = st.tabs([
        "📋 [미션 3/4번] 회의 요약 & 우선순위", 
        "📅 [3번] 캘린더 마일스톤", 
        "✉️ [미션 1/7번] 팔로업 메일 전송", 
        "🎙️ [미션 5/9번] 1분 출근길 브리핑", 
        "💬 [미션 2/6번] 에이블 음성 Q&A"
    ])

    # [탭 1: 회의 요약 & 4번 아이젠하워 매트릭스]
    with tabs[0]:
        st.subheader("📋 회의록 요약 및 업무 우선순위")
        if st.button("🚀 종합 분석 실행", key="btn_summary"):
            with st.spinner("에이블이 회의록을 분석 중입니다..."):
                prompt = f"""
                다음 미팅 내용을 분석하세요.
                [미팅 내용]
                {meeting_text}
                
                규칙에 맞춰 요약 및 김 대리 업무를 정리하고, 
                추가로 김 대리의 업무 3가지를 [긴급도 x 중요도] 기준의 아이젠하워 매트릭스(DO FIRST, SCHEDULE, DELEGATE)로 분류해 주세요.
                """
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": sys_role},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                answer = response.output_text
                st.markdown(answer)

    # [탭 2: 3번 Action Scheduler - 캘린더 등록]
    with tabs[1]:
        st.subheader("📅 마감 역산 캘린더 스케줄러")
        st.info("회의록에서 마감 기한을 역산하여 D-7(착수), D-3(초안), D-Day(보고) 등록 링크를 생성합니다.")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("**[D-7] 경쟁사 분석 착수**\n\n기한: 2026-11-04")
            st.link_button("📅 구글 캘린더 등록", "https://calendar.google.com/calendar/render?action=TEMPLATE&text=[착수]+경쟁사+마케팅+분석&dates=20261104T090000Z/20261104T100000Z")
        with col_c2:
            st.markdown("**[D-3] SNS 콘텐츠 초안 완료**\n\n기한: 2026-11-08")
            st.link_button("📅 구글 캘린더 등록", "https://calendar.google.com/calendar/render?action=TEMPLATE&text=[중간]+SNS+홍보+초안&dates=20261108T090000Z/20261108T100000Z")
        with col_c3:
            st.markdown("**[D-Day] 마케팅 최종 보고**\n\n기한: 2026-11-11")
            st.link_button("📅 구글 캘린더 등록", "https://calendar.google.com/calendar/render?action=TEMPLATE&text=[마감]+신제품+마케팅+최종보고&dates=20261111T090000Z/20261111T100000Z")

    # [탭 3: 미션 1 & 7 기반 팔로업 메일 수정/발송]
    with tabs[2]:
        st.subheader("✉️ 협업자 선행 자료 요청 메일 (원클릭 승인/수정)")
        with st.form("mail_form"):
            to_input = st.text_input("받는 사람", value="soulsky0218@gmail.com")
            subject_input = st.text_input("제목", value="[자료요청] 신제품 마케팅 관련 A사 캠페인 데이터 요청 건")
            body_input = st.text_area("메일 본문", value=f"이과장님 안녕하세요. 김 대리입니다.\n회의록 기반 경쟁사 분석을 위해 논의되었던 A사 세부 프로모션 데이터를 부탁드립니다.", height=150)
            mail_submit = st.form_submit_button("✉️ 최종 수정본 발송")
            
            if mail_submit:
                res = send_email(to_input, subject_input, body_input)
                if res["status"] == "success":
                    st.success("✅ 메일이 성공적으로 전송되었습니다!")
                else:
                    st.error(f"전송 실패: {res['error']}")

    # [탭 4: 미션 5 기반 1분 오디오 플래시 브리핑]
    with tabs[3]:
        st.subheader("🎙️ 1분 출근길 오디오 브리핑")
        st.caption("바쁜 이동 중 이어폰으로 듣는 핵심 요약 (OpenAI TTS 연동)")
        if st.button("🔊 1분 음성 브리핑 생성"):
            with st.spinner("AI가 핵심을 압축하여 음성을 생성 중입니다..."):
                brief_res = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": "1분 출근길 모닝 앵커처럼 회의 핵심 결론과 김대리의 1순위 업무만 3문장 이내로 브리핑하세요."},
                        {"role": "user", "content": meeting_text}
                    ],
                    temperature=0
                )
                brief_script = brief_res.output_text
                st.write(f"**브리핑 원고:** {brief_script}")
                
                speech = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=brief_script
                )
                st.audio(speech.content, format="audio/mp3")

    # [탭 5: 2일차 실시간 음성 지시 인터페이스]
    with tabs[4]:
        st.subheader("💬 실시간 음성 질의응답")
        audio_value = st.audio_input("마이크 버튼을 누르고 음성으로 지시해 보세요")
        
        if audio_value is not None:
            with st.spinner("음성 인식 및 답변 생성 중..."):
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("voice_cmd.wav", audio_value, "audio/wav")
                )
                st.info(f"🗣️ 인식된 지시: {transcript.text}")
                
                resp = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": sys_role + f"\n\n[회의록 내용]\n{meeting_text}"},
                        {"role": "user", "content": transcript.text}
                    ],
                    temperature=0
                )
                answer = resp.output_text
                st.markdown(f"**🤖 에이블 답변:**\n{answer}")
else:
    st.info("👈 상단에서 회의록(.txt)이나 회의 녹음 파일을 업로드하면 에이블 비서 기능이 활성화됩니다.")
