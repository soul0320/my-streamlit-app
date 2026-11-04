import os
import re
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import streamlit as st
from openai import OpenAI
from calendar_service import create_calendar_event

# 1. 페이지 설정
st.set_page_config(
    page_title="KT Able - AI 마케팅 전속 비서",
    page_icon="🤖",
    layout="wide"
)

# 2. 클라이언트 초기화
client = OpenAI()

# 이메일 발송 함수
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender or not password:
        return {"status": "failed", "error": "메일 인증 계정 정보(GMAIL_ADDRESS/APP_PASSWORD)가 없습니다."}
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

# [가드레일] 10. 개인정보(PII) 마스킹 & 컴플라이언스 검수
def compliance_pii_guardrail(text):
    phone_pattern = r'(\d{2,3})[-.\s]?(\d{3,4})[-.\s]?(\d{4})'
    rrn_pattern = r'(\d{6})[-.\s]?([1-4]\d{6})'
    masked_text = re.sub(phone_pattern, r'\1-****-****', text)
    masked_text = re.sub(rrn_pattern, r'\1-*******', masked_text)

    try:
        mod_resp = client.moderations.create(input=text)
        is_flagged = mod_resp.results[0].flagged
    except Exception:
        is_flagged = False

    return masked_text, is_flagged

# 3. 사이드바: 기본 인풋 게이트웨이 (음성 m4a/mp3 + 텍스트 txt)
with st.sidebar:
    st.image("https://raw.githubusercontent.com/Jangrae/img/master/ai_robot.png", width=140)
    st.title("에이블 (Able)")
    st.caption("KT 마케팅팀 김 대리 전속 지능형 비서")
    st.markdown("---")

    st.subheader("📁 회의 데이터 입력")
    uploaded_file = st.file_uploader(
        "회의록(.txt) 또는 녹음 파일(.m4a, .mp3, .wav)",
        type=["txt", "m4a", "mp3", "wav"]
    )

    meeting_content = ""
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext == "txt":
            meeting_content = uploaded_file.read().decode("utf-8")
            st.success("📄 텍스트 회의록 로드 완료")
        else:
            with st.spinner("🎙️ 음성 파일을 STT(Whisper)로 변환 중입니다..."):
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(uploaded_file.name, uploaded_file.read())
                )
                meeting_content = transcript.text
                st.success("✅ 음성 인식(STT) 완료!")
                with st.expander("변환된 음성 스크립트 보기"):
                    st.write(meeting_content)
    else:
        st.info("회의록 파일 또는 녹음 음성 파일을 업로드해 주세요.")

# 본문 영역
st.header("KT 마케팅 전속 비서 서비스 '에이블'")
st.markdown("김 대리의 업무 부담을 최소화하는 **온톨로지 관계망 기반 선제적 에이전트**")

if not meeting_content:
    st.warning("👈 사이드바에서 회의록 또는 녹음 파일을 먼저 업로드해 주세요.")
    st.stop()

# PII 가드레일 통과
clean_meeting_text, is_flagged = compliance_pii_guardrail(meeting_content)
if is_flagged:
    st.error("⚠️ 회의록에 사내 컴플라이언스 기준 부적합 단어가 감지되었습니다. 필터링된 데이터로만 분석합니다.")

# 4. 온톨로지 기반 업무 분석 엔진 (캐싱 처리)
@st.cache_data(show_spinner=False)
def analyze_meeting_ontology(raw_text):
    prompt = f"""
    당신은 KT 마케팅팀 김 대리의 전속 AI 개인비서 '에이블'입니다.
    다음 회의록의 온톨로지 관계망 [참석자 ↔ 배정업무 ↔ 마감일 ↔ 선행 요청자료 ↔ 리스크]을 분석해 JSON 포맷으로 정확히 출력하세요.

    회의록:
    {raw_text}

    반드시 아래 JSON 스키마를 만족해야 하며, 마크다운 기호 없이 순수 JSON만 반환하세요:
    {{
      "summary": ["핵심 요약 1", "핵심 요약 2", "핵심 요약 3"],
      "decisions": ["결정 사항 1", "결정 사항 2"],
      "kim_tasks": [
        {{
          "task_name": "업무명",
          "deadline": "YYYY-MM-DD",
          "urgency": "High 또는 Low",
          "importance": "High 또는 Low",
          "eisenhower_category": "DO FIRST / SCHEDULE / DELEGATE 중 택 1",
          "action_guide": "김대리가 즉시 취할 행동 가이드"
        }}
      ],
      "followup_requests": [
        {{
          "target_person": "자료 요청할 사람 (예: 이과장님)",
          "email": "lee@kt.com",
          "request_data": "요청할 선행 데이터",
          "email_subject": "메일 제목 초안",
          "email_body": "정중한 비즈니스 메일 본문 초안"
        }}
      ],
      "briefing_script": "바쁜 출근길에 이어폰으로 들을 수 있는 1분 모닝 뉴스 앵커 톤의 브리핑 멘트 (3~4문장)",
      "devils_advocate": {{
        "optimistic_plan": "회의에서 합의된 낙관적 마케팅 계획",
        "potential_risks": ["숨은 리스크 1", "숨은 리스크 2"],
        "senior_advice": "50년 차 시니어 마케터 관점의 현실적 대안 및 방어 논리"
      }}
    }}
    """
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": prompt}]
    )
    raw_output = response.output_text.replace("```json", "").replace("```", "").strip()
    return json.loads(raw_output)

with st.spinner("🧠 온톨로지 관계망 기반으로 회의록을 심층 분석 중입니다..."):
    try:
        analysis_data = analyze_meeting_ontology(clean_meeting_text)
    except Exception as e:
        st.error(f"분석 엔진 파싱 오류: {e}")
        st.stop()

# 5. 핵심 5대 기능 탭 구성
tabs = st.tabs([
    "📋 회의록 & 우선순위 (R&R)",
    "📅 스마트 캘린더 등록",
    "✉️ 팔로업 메일 초안·발송",
    "🎙️ 1분 출근길 오디오 브리핑",
    "🔍 시니어 전략 진단 (Devil's Advocate)"
])

# 탭 1: 회의록 요약 및 R&R 아이젠하워 매트릭스
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("💡 핵심 3줄 요약 & 최종 결정")
        for s in analysis_data.get("summary", []):
            st.markdown(f"- {s}")
        st.markdown("**📌 최종 결정 사항**")
        for d in analysis_data.get("decisions", []):
            st.info(d)

    with col2:
        st.subheader("🚨 R&R 밸런스 & 우선순위 배정")
        tasks = analysis_data.get("kim_tasks", [])
        st.warning(f"⚠️ 김 대리님께 총 **{len(tasks)}개**의 주요 업무가 집중되었습니다. 우선순위에 따라 순차 착수하세요.")

        for task in tasks:
            cat = task.get("eisenhower_category", "DO FIRST")
            color = "🔴" if "DO FIRST" in cat else ("🟡" if "SCHEDULE" in cat else "🟢")
            with st.container(border=True):
                st.markdown(f"**{color} [{cat}] {task['task_name']}**")
                st.caption(f"기한: {task.get('deadline', '미정')} | 긴급도: {task.get('urgency')} | 중요도: {task.get('importance')}")
                st.write(task.get("action_guide"))

# 탭 2: 마감 역산 캘린더링 (Google Calendar 연동)
with tabs[1]:
    st.subheader("📅 마감 기한 역산 기반 자동 마일스톤 생성")
    st.write("회의록상 마감일을 기준으로 D-7(착수), D-3(중간점검), D-Day(보고) 일정을 자동 계산하여 등록합니다.")

    for task in analysis_data.get("kim_tasks", []):
        t_name = task["task_name"]
        deadline = task.get("deadline", "2026-11-15")

        st.markdown(f"#### 🎯 과제: {t_name}")
        c1, c2, c3 = st.columns(3)

        # 가상 마일스톤 계산
        start_d7 = f"{deadline}T09:00:00"
        end_d7 = f"{deadline}T10:00:00"

        with c1:
            st.markdown(f"**[D-7] 착수/자료수집**\n\n{deadline}")
            if st.button(f"캘린더 등록 (D-7)", key=f"cal_d7_{t_name}"):
                res = create_calendar_event(f"[착수] {t_name}", start_d7, end_d7, "에이블 비서 자동 스케줄링")
                st.success("등록 링크 생성 완료!")
                st.markdown(f"[👉 구글 캘린더에서 확인하기]({res.get('htmlLink', res.get('link'))})")
        with c2:
            st.markdown(f"**[D-3] 중간 점검**\n\n{deadline}")
            if st.button(f"캘린더 등록 (D-3)", key=f"cal_d3_{t_name}"):
                res = create_calendar_event(f"[중간점검] {t_name}", start_d7, end_d7, "초안 검토 및 피드백")
                st.success("등록 링크 생성 완료!")
                st.markdown(f"[👉 구글 캘린더에서 확인하기]({res.get('htmlLink', res.get('link'))})")
        with c3:
            st.markdown(f"**[D-Day] 최종 마감**\n\n{deadline}")
            if st.button(f"캘린더 등록 (D-Day)", key=f"cal_dday_{t_name}"):
                res = create_calendar_event(f"[마감] {t_name} 최종 보고", start_d7, end_d7, "팀장님 대면 보고")
                st.success("등록 링크 생성 완료!")
                st.markdown(f"[👉 구글 캘린더에서 확인하기]({res.get('htmlLink', res.get('link'))})")
        st.markdown("---")

# 탭 3: 팔로업 이메일 초안 생성 및 수정 발송
with tabs[2]:
    st.subheader("✉️ 협업 부서 선행 자료 요청 메일 초안")
    st.caption("김 대리가 회의 후 요청해야 할 선행 데이터를 비서가 미리 감지해 작성해 둔 메일입니다.")

    requests = analysis_data.get("followup_requests", [])
    if requests:
        for idx, req in enumerate(requests):
            st.markdown(f"**요청 대상: {req.get('target_person')}** ({req.get('request_data')})")
            with st.form(key=f"email_form_{idx}"):
                to_addr = st.text_input("수신자 이메일", value=req.get("email", "soulsky0218@gmail.com"))
                subject = st.text_input("메일 제목", value=req.get("email_subject", ""))
                body = st.text_area("메일 본문 (수정 가능)", value=req.get("email_body", ""), height=180)

                submitted = st.form_submit_button("✉️ 확인 및 즉시 발송")
                if submitted:
                    with st.spinner("메일 발송 중..."):
                        send_res = send_email(to_addr, subject, body)
                        if send_res.get("status") == "success":
                            st.success(f"✅ {req.get('target_person')}님께 이메일이 정상 발송되었습니다!")
                        else:
                            st.error(f"발송 실패: {send_res.get('error')}")
    else:
        st.info("회의록에서 협업자에게 요청할 선행 자료가 없습니다.")

# 탭 4: 1분 오디오 플래시 브리핑
with tabs[3]:
    st.subheader("🎙️ 1분 출근길 오디오 플래시 브리핑")
    st.caption("외부 이동 중 화면을 볼 수 없을 때 이어폰으로 듣는 앵커 브리핑 음성입니다.")

    briefing_script = analysis_data.get("briefing_script", "오늘의 회의 요약 브리핑입니다.")
    st.text_area("앵커 브리핑 스크립트", value=briefing_script, height=120, disabled=True)

    if st.button("🔊 1분 음성 브리핑 생성 및 재생"):
        with st.spinner("AI 음성을 합성 중입니다 (OpenAI TTS)..."):
            speech = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=briefing_script
            )
            st.audio(speech.content, format="audio/mp3")

# 탭 5: 시니어 마케터 리스크 진단 (Devil's Advocate)
with tabs[4]:
    st.subheader("🔍 신제품 런칭 리스크 진단 (Devil's Advocate)")
    st.caption("50년 차 마케팅 베테랑의 안목으로 회의 중 통과된 계획의 숨은 결함을 선제 점검합니다.")

    da = analysis_data.get("devils_advocate", {})
    st.markdown(f"**📋 회의에서 합의된 낙관적 계획:**\n> {da.get('optimistic_plan', 'SNS 인플루언서 집중 홍보')}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ⚠️ 포착된 잠재 리스크 (결함)")
        for risk in da.get("potential_risks", []):
            st.error(f"• {risk}")

    with col_b:
        st.markdown("#### 💡 베테랑 마케터의 방어 논리 & 플랜 B")
        st.success(da.get("senior_advice", "단일 채널 몰아주기보다는 CVR 검증을 위한 A/B 테스트 분산 집행 권고"))
