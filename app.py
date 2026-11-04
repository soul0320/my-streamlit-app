import os
import re
import json
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from openai import OpenAI

# ==============================================================================
# 1. 기본 페이지 설정 및 OpenAI 클라이언트 초기화
# ==============================================================================
st.set_page_config(
    page_title="마케팅팀 김 대리 전속 AI 비서 - 에이블",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

client = OpenAI()
AIVLE_AVATAR_URL = "https://raw.githubusercontent.com/Jangrae/img/master/ai_robot.png"

# ==============================================================================
# 2. 기본 탑재 회의록 (업로드 없이 즉시 시연 가능하도록 기본 세팅)
# ==============================================================================
DEFAULT_MEETING_TEXT = """팀장: 오늘 미팅을 시작하겠습니다. 오늘 주제는 신제품 출시 마케팅 전략입니다.
박과장: 김대리가 외출 중이라 오늘 회의에는 참석하지 못했습니다.
최대리: 제가 나중에 김대리에게 미팅 기록을 전달하겠습니다.
팀장: 네, 그렇게 해주세요. 김대리도 내용은 꼭 알아야 합니다.
이과장: 이번 신제품은 다음 달 초 출시 예정이라 마케팅 준비를 서둘러야 합니다.
팀장: 맞습니다. 우선 전체적인 마케팅 방향부터 정해보죠.
박과장: 저는 온라인 중심으로 마케팅을 진행하는 것이 좋다고 생각합니다.
이과장: 요즘은 SNS 광고가 가장 효과적이니까요.
최대리: 특히 인스타그램과 유튜브가 좋을 것 같습니다.
팀장: 네, SNS 마케팅은 기본으로 가져가야겠네요.
박과장: 그리고 경쟁사 사례를 분석해 보면 좋겠습니다.
팀장: 좋은 의견입니다. 경쟁사들이 어떤 마케팅을 하는지 파악해야 합니다.
이과장: 최근 경쟁사 A사가 SNS 캠페인을 크게 진행했습니다.
최대리: 맞습니다. 인플루언서를 활용한 캠페인이었습니다.
팀장: 그렇다면 우리도 경쟁사 마케팅을 분석할 필요가 있겠네요. 이 업무는 김대리가 맡아주면 좋겠습니다.
박과장: 김대리가 자료 정리와 분석을 잘하니까 적합할 것 같습니다.
팀장: 경쟁사 SNS 광고와 프로모션 사례를 정리해 달라고 전달해 주세요. 보고서는 다음 주까지 작성해서 공유하면 좋겠습니다.
최대리: 네, 제가 김대리에게 전달하겠습니다.
이과장: 그리고 SNS 홍보 콘텐츠도 준비해야 합니다.
박과장: 카드뉴스나 짧은 영상 콘텐츠가 좋겠습니다.
최대리: 요즘은 숏폼 영상 반응이 좋습니다.
팀장: 그럼 SNS 홍보 콘텐츠 초안 작성도 김대리에게 맡기죠.
박과장: 제품 기능을 강조하는 방향이 좋을 것 같습니다.
이과장: 사용 후기 형식의 콘텐츠도 효과적입니다.
팀장: 그런 방향까지 포함해서 김대리에게 설명해 주세요.
최대리: 네, 알겠습니다.
박과장: 인플루언서 마케팅도 같이 검토하면 좋겠습니다.
이과장: 비용이 많이 들 수 있으니 중간 규모 인플루언서가 좋겠습니다.
팀장: 좋은 생각입니다. 중급 인플루언서 3명 정도 후보 리스트를 조사해 보죠. 이것도 김대리가 같이 조사해 주면 좋겠습니다.
최대리: 네, 함께 전달하겠습니다.
팀장: 전체 마케팅 예산은 약 5천만 원 정도로 생각하고 있습니다.
이과장: SNS 광고에 절반 정도 사용하는 것이 좋겠습니다.
박과장: 나머지는 콘텐츠 제작과 인플루언서 협업에 쓰면 되겠네요.
팀장: 광고는 출시 2주 전부터 시작하는 것이 좋겠습니다.
이과장: 초기 프로모션으로 출시 기념 할인 이벤트와 SNS 공유 이벤트를 진행하면 좋겠습니다.
팀장: 다음 주 회의에서 전체 진행 상황을 점검하겠습니다."""

# ==============================================================================
# 3. 유틸리티 함수 (이메일 발송, 개인정보 마스킹, TTS 음성 생성)
# ==============================================================================
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return {"status": "failed", "error": "Gmail 환경변수(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)가 등록되지 않았습니다."}
    
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
    phone_pattern = r'(\d{2,3})[-.\s]?(\d{3,4})[-.\s]?(\d{4})'
    rrn_pattern = r'(\d{6})[-.\s]?([1-4]\d{6})'
    masked = re.sub(phone_pattern, r'\1-****-****', text)
    masked = re.sub(rrn_pattern, r'\1-*******', masked)
    return masked

def play_tts(text, voice="nova"):
    try:
        clean_text = re.sub(r'[#*`>-]', '', text)[:450]
        speech = client.audio.speech.create(model="tts-1", voice=voice, input=clean_text)
        st.audio(speech.content, format="audio/mp3")
    except Exception as e:
        st.warning(f"음성 생성 중 일시적 오류: {e}")

# ==============================================================================
# 4. 세션 상태 초기화 (멀티턴 히스토리 10개 관리 및 자동 분석 캐시)
# ==============================================================================
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = DEFAULT_MEETING_TEXT
if "meeting_source" not in st.session_state:
    st.session_state.meeting_source = "기본 탑재 회의록 (신제품 출시 마케팅 전략)"
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요, 김 대리님! 외출 중 진행되었던 **'신제품 출시 마케팅 전략'** 회의록 분석과 WBS 일정 수립을 모두 마쳤습니다. 위쪽 대시보드에서 제가 미리 차려둔 업무 결과물을 확인하시거나, 아래 채팅창으로 무엇이든 지시해 주세요!"}
    ]
if "auto_wbs_report" not in st.session_state:
    st.session_state.auto_wbs_report = ""
if "last_analyzed_text" not in st.session_state:
    st.session_state.last_analyzed_text = ""

# ==============================================================================
# 5. 시니어 AI 비서 '에이블' 페르소나 프롬프트 (기한 미정 금지 & 선제적 가이드)
# ==============================================================================
def get_sys_role(target_opt, tone_opt, format_opt):
    return f"""
당신은 마케팅팀 '김 대리(Marketing Data Analyst)'를 보좌하는 10년 차 시니어 AI 비서 '에이블(AIVLE)'입니다.
김 대리의 일이 늘어나지 않도록, 김 대리가 지시하기 전에 먼저 일정을 짜고, 누구에게 무엇을 요청해야 하는지 챙기고, 실무 초안까지 완성해 두는 '선제적(Proactive) 비서'입니다.

[현재 적용된 마케팅 부가 설정]
- 타깃 시장/고객층: {target_opt}
- 보고서 및 카피 톤앤매너: {tone_opt}
- 산출물 포맷 기준: {format_opt}

[핵심 업무 및 답변 절대 규칙 (승환님 표준 공유본 + 시니어 비서 고도화)]
1. 한국어로 정중하고 명확하며, 유능한 비서답게 주도적으로 브리핑합니다.
2. 회의록 요약은 반드시 [주제] → [논의 내용] → [결정 사항(예산/일정/채널 수치 필수)] 순서로 정리합니다.
3. **[절대 규칙 - '기한 미정' 표기 금지]**:
   - 회의록에 기한이 명시된 업무는 해당 기한(예: '다음 주 수요일 11/11까지')을 적습니다.
   - 회의록에 기한이 명시되지 않은 업무라도 절대 '기한 미정'이라고 적지 마세요! 전체 출시 일정(다음 달 초)과 광고 시작일(출시 2주 전), 다음 점검 회의(다음 주)를 역산하여 **"2026-11-09(월)까지 [비서 추천 기한: 다음 주 점검 회의 및 출시 2주 전 광고 집행 역산]"** 형태로 구체적인 권장 마감일과 그 이유를 비서처럼 정확히 가이드하세요.
4. 김 대리의 모든 업무는 **WBS(Work Breakdown Structure)** 형태로 [대분류 과업 - 세부 실행 작업(Work Package) - 협업/제안 대상(누구에게 무엇을 제안/요청할지) - 목표 기한(명시/비서 추천) - 기대 산출물]로 쪼개어 표로 제시합니다.
5. 이메일 본문 작성 시 마크다운 기호(*, # 등) 없이 즉시 발송 가능한 깔끔한 비즈니스 텍스트로 작성합니다.
"""

# ==============================================================================
# 6. 사이드바 구성 (TypeError 유발 위젯 제거 -> 안전한 기본 메뉴로 통합)
# ==============================================================================
with st.sidebar:
    st.image(AIVLE_AVATAR_URL, width=130)
    st.title("🤖 에이블 비서실")
    st.caption("김 대리님, 타이핑은 제가 할 테니 클릭으로 승인만 하세요!")
    
    # 1) 워크스페이스 메인 메뉴 (기본 사이드바 메뉴에 통합)
    st.markdown("### 📌 비서 업무 워크스페이스")
    menu = st.radio(
        "실행할 비서 업무를 선택하세요",
        [
            "📊 회의 요약 & WBS 업무 설계",
            "✍️ 마케팅 실무 초안 자동 생성기",
            "✉️ 원클릭 협업 메일 & 일정 등록",
            "🧐 시니어 마케터 리스크 진단",
            "🎧 1분 출근길 오디오 브리핑"
        ]
    )
    
    st.markdown("---")
    
    # 2) 회의록 파일 & 음성 즉시 업로드 (올리자마자 자동 분석)
    st.markdown("### 📥 새 회의록 / 녹음 즉시 분석")
    uploaded_file = st.file_uploader(
        "문서(.txt) 또는 음성(.m4a/.mp3/.wav)",
        type=["txt", "m4a", "mp3", "wav"]
    )
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "txt":
            raw_txt = uploaded_file.read().decode("utf-8")
            st.session_state.meeting_text = mask_personal_info(raw_txt)
            st.session_state.meeting_source = f"업로드 문서: {uploaded_file.name}"
            st.success("✅ 텍스트 회의록 즉시 반영 완료!")
        else:
            with st.spinner("🎙️ 음성 파일을 텍스트로 변환(STT)하여 분석 중..."):
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(uploaded_file.name, uploaded_file.read())
                )
                st.session_state.meeting_text = mask_personal_info(transcript.text)
                st.session_state.meeting_source = f"음성 파일(STT): {uploaded_file.name}"
                st.success(f"✅ 음성 변환 완료! ({transcript.text[:30]}...)")

    # 현장 음성 녹음 모드 (필요시 체크박스로 안전하게 활성화)
    use_live_mic = st.checkbox("🎙️ 마이크 실시간 음성 지시/녹음 켜기", value=False)
    if use_live_mic:
        audio_val = st.audio_input("말씀하시면 즉시 텍스트로 변환해 반영합니다")
        if audio_val is not None:
            with st.spinner("음성 인식(STT) 중..."):
                tr = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=("live.wav", audio_val, "audio/wav")
                )
                st.info(f"🗣️ 인식된 음성: {tr.text}")
                if "요약" in tr.text or
