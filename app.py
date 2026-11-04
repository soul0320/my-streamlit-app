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
    # ============================================================
# 웹 브라우저 실시간 음성 녹음 및 STT
# ============================================================
# 기능:
# 1. 웹 브라우저에서 마이크로 직접 녹음
# 2. 녹음 종료 후 OpenAI Whisper API 호출
# 3. 음성을 텍스트로 자동 변환
# 4. 변환 결과 확인 및 회의록 분석에 활용
# ============================================================

st.markdown("---")

st.subheader("🎙️ 실시간 음성 녹음")

st.caption(
    "마이크 버튼을 누르고 녹음하세요. "
    "녹음이 끝나면 자동으로 텍스트로 변환합니다."
)

# 1. Streamlit 내장 마이크 녹음 기능
recorded_audio = st.audio_input(
    "회의 내용 또는 음성 명령 녹음",
    sample_rate=16000
)

# 2. 녹음 파일이 있는 경우
if recorded_audio is not None:

    audio_bytes = recorded_audio.getvalue()

    # 동일한 녹음을 반복 변환하지 않도록 상태 관리
    if st.session_state.get("last_audio") != audio_bytes:

        with st.spinner("🎙️ 음성을 텍스트로 변환하고 있습니다..."):

            try:

                # 3. OpenAI STT API 호출
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(
                        "recording.wav",
                        audio_bytes,
                        "audio/wav"
                    )
                )

                # 4. 변환 결과 저장
                st.session_state["live_transcript"] = transcript.text
                st.session_state["last_audio"] = audio_bytes

            except Exception as e:

                st.error(f"음성 인식 오류: {e}")

    # 5. 변환된 텍스트 화면 표시
    if st.session_state.get("live_transcript"):

        st.success("✅ 음성 인식 완료!")

        st.text_area(
            "음성 인식 결과",
            value=st.session_state["live_transcript"],
            height=150
        )

        # 6. TXT 파일 다운로드
        st.download_button(
            "📄 변환된 TXT 다운로드",
            data=st.session_state["live_transcript"],
            file_name="meeting_transcript.txt",
            mime="text/plain"
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

# ============================================================
# 실시간 녹음 결과를 회의록 분석 데이터에 연결
# ============================================================
# 업로드한 파일이 없을 경우,
# 웹에서 녹음한 텍스트를 회의록으로 사용합니다.
# ============================================================

if not meeting_content:

    meeting_content = st.session_state.get(
        "live_transcript",
        ""
    )
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
    [마감일 추출 원칙]
    1. 회의록에 명시된 마감일만 확정 마감일로 추출합니다.
    2. 회의록에 마감일이 없으면 deadline을 null로 반환합니다.
    3. 회의 날짜를 마감일로 잘못 인식하지 않습니다.
    4. AI가 임의로 마감일을 생성하지 않습니다.

    회의록:
    {raw_text}

    반드시 아래 JSON 스키마를 만족해야 하며, 마크다운 기호 없이 순수 JSON만 반환하세요:
    {{
      "summary": ["핵심 요약 1", "핵심 요약 2", "핵심 요약 3"],
      "decisions": ["결정 사항 1", "결정 사항 2"],
      "kim_tasks": [
        {{
          "task_name": "업무명",
          "deadline": "YYYY-MM-DD 또는 null",
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
    "🔍 마케팅 전략 보고서"
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

# ============================================================
# 탭 2. 마감 기한 역산 기반 자동 마일스톤 생성
# ============================================================
# 기능:
# 1. 회의록에서 추출한 김 대리의 담당 업무 확인
# 2. 실제 마감일을 기준으로 D-7, D-3, D-Day 계산
# 3. 마감일이 없는 경우 임의 날짜 생성 방지
# 4. 필요 시 사용자가 마감일 직접 수정
# 5. Google Calendar 일정 등록 링크 생성
# ============================================================

with tabs[1]:

    st.subheader("📅 마감 기한 역산 기반 자동 마일스톤 생성")

    st.write(
        "회의록상 마감일을 기준으로 "
        "D-7(착수), D-3(중간점검), D-Day(보고) "
        "일정을 자동 계산하여 등록합니다."
    )

    # 김 대리에게 배정된 업무 목록
    tasks = analysis_data.get("kim_tasks", [])

    for idx, task in enumerate(tasks):

        # 1. 업무명 및 기존 마감일 확인
        t_name = task.get("task_name", "업무명 미정")
        deadline_str = task.get("deadline")

        st.markdown(f"#### 🎯 과제: {t_name}")

        # 2. AI가 추출한 마감일의 유효성 검증
        try:
            deadline = datetime.strptime(
                deadline_str,
                "%Y-%m-%d"
            ).date()

        except (ValueError, TypeError):
            deadline = None

        # 3. 마감일이 없는 경우 사용자 입력 요청
        if deadline is None:

            st.warning(
                "⚠️ 회의록에서 유효한 마감일을 "
                "확인하지 못했습니다."
            )

            deadline = st.date_input(
                "업무 마감일을 직접 설정하세요.",
                value=None,
                key=f"deadline_{idx}"
            )

        else:

            # AI가 추출한 마감일도 사용자가 수정할 수 있음
            deadline = st.date_input(
                "업무 마감일 확인 및 수정",
                value=deadline,
                key=f"deadline_{idx}"
            )

        # 마감일이 없으면 캘린더 생성 중단
        if deadline is None:
            st.info("마감일을 설정하면 일정이 자동 계산됩니다.")
            st.divider()
            continue

        # 4. 실제 마감일 기준 역산
        d7 = deadline - timedelta(days=7)
        d3 = deadline - timedelta(days=3)
        dday = deadline

        # 5. 캘린더에 전달할 시작 및 종료 시간 생성
        # 기본 일정 시간은 오전 9시~10시로 설정
        def make_calendar_time(date_value):

            start = f"{date_value}T09:00:00"
            end = f"{date_value}T10:00:00"

            return start, end

        # 6. 마일스톤별 화면 구성
        milestones = [
            ("D-7", "착수/자료수집", d7),
            ("D-3", "중간 점검", d3),
            ("D-Day", "최종 마감", dday)
        ]

        cols = st.columns(3)

        for col, milestone in zip(cols, milestones):

            label, title, date_value = milestone

            with col:

                st.markdown(f"**[{label}] {title}**")

                # 정확하게 계산된 날짜 표시
                st.write(date_value.strftime("%Y-%m-%d"))

                # 캘린더 일정 시작 및 종료 시간
                start_time, end_time = make_calendar_time(
                    date_value
                )

                # 캘린더 등록 버튼
                if st.button(
                    f"캘린더 등록 ({label})",
                    key=f"calendar_{idx}_{label}"
                ):

                    result = create_calendar_event(
                        f"[{title}] {t_name}",
                        start_time,
                        end_time,
                        f"에이블 AI 비서 자동 생성 일정: {t_name}"
                    )

                    # 생성된 캘린더 링크 확인
                    calendar_link = result.get(
                        "htmlLink",
                        result.get("link")
                    )

                    if calendar_link:

                        st.success("✅ 일정 등록 링크 생성 완료")

                        st.markdown(
                            f"[👉 Google Calendar에서 등록하기]"
                            f"({calendar_link})"
                        )

                    else:

                        st.error(
                            "캘린더 링크를 생성하지 못했습니다."
                        )

        st.divider()

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

# ============================================================
# 탭 5. AI 마케팅 전략 보고서 작성 및 이메일 발송
# ============================================================
# 기능:
# 1. 회의록 기반 마케팅 전략 보고서 초안 생성
# 2. 핵심 결정사항 및 담당 업무 포함
# 3. 리스크와 대응방안 분석
# 4. 사용자가 보고서 내용을 직접 수정
# 5. 최종 검토 후 이메일 발송
# ============================================================

with tabs[4]:

    st.subheader("📑 AI 마케팅 전략 보고서 작성")

    st.caption(
        "회의록을 기반으로 보고서 초안을 생성하고, "
        "내용을 수정한 뒤 이메일로 발송할 수 있습니다."
    )

    # 1. 보고서 초안 저장 공간
    if "marketing_report" not in st.session_state:

        st.session_state.marketing_report = ""

    # 회의록이 변경되면 기존 보고서와 구분
    import hashlib

    meeting_id = hashlib.sha256(
        clean_meeting_text.encode("utf-8")
    ).hexdigest()

    if st.session_state.get("report_meeting_id") != meeting_id:

        st.session_state.report_meeting_id = meeting_id
        st.session_state.marketing_report = ""

    # 2. 보고서 작성 버튼
    if st.button(
        "📝 마케팅 전략 보고서 초안 생성",
        type="primary"
    ):

        with st.spinner(
            "에이블이 마케팅 전략 보고서를 작성하고 있습니다..."
        ):

            try:

                report_prompt = f"""
                당신은 마케팅팀의 시니어 전략 기획 비서입니다.

                아래 회의록을 분석하여 실무에서 활용할 수 있는
                마케팅 전략 보고서 초안을 작성하세요.

                [보고서 구성]

                1. 보고서 제목

                2. Executive Summary
                - 핵심 내용 요약
                - 주요 결정 사항

                3. 시장 및 경쟁 환경 분석
                - 경쟁사 현황
                - 주요 시사점

                4. 마케팅 실행 전략
                - 타깃 고객
                - 콘텐츠 전략
                - 채널 운영 방향

                5. 업무 추진 계획
                - 담당자
                - 수행 업무
                - 마감일
                - 선행 협업사항

                6. 주요 리스크 및 대응방안
                - 예상 위험요인
                - 대응방안
                - 추가 검토 필요사항

                7. 향후 실행 계획 및 의사결정 요청사항

                [작성 원칙]
                - 전문적인 비즈니스 보고서 문체를 사용합니다.
                - 실제 회의록에 나온 정보만 확정 사실로 작성합니다.
                - 회의록에 없는 수치나 일정은 만들어내지 않습니다.
                - AI가 제안하는 전략은 'AI 제안'으로 표시합니다.
                - 사용자가 수정하기 쉬운 문단 형태로 작성합니다.
                - 작성된 보고서는 최종 승인 전 초안입니다.

                [회의록]
                {clean_meeting_text[:30000]}
                """

                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=report_prompt
                )

                # 생성한 보고서 저장
                st.session_state.marketing_report = (
                    response.output_text
                )

                st.success("✅ 보고서 초안 생성 완료!")

            except Exception as e:

                st.error(
                    f"보고서 생성 오류: {e}"
                )

    # 3. 보고서 수정 및 이메일 전송
    if st.session_state.marketing_report:

        st.markdown("---")

        st.subheader("✏️ 보고서 검토 및 수정")

        # 사용자가 직접 보고서 내용 수정
        edited_report = st.text_area(
            "보고서 본문",
            value=st.session_state.marketing_report,
            height=500,
            key=f"edited_marketing_report_{meeting_id}"
        )

        st.caption(
            "AI가 작성한 초안입니다. "
            "내용을 검토하고 수정한 뒤 발송하세요."
        )

        # 4. 보고서 TXT 다운로드
        st.download_button(
            "📄 보고서 TXT 다운로드",
            data=edited_report,
            file_name="marketing_strategy_report.txt",
            mime="text/plain"
        )

        st.markdown("---")

        # 5. 이메일 전송 기능
        st.subheader("✉️ 보고서 이메일 발송")

        with st.form("report_email_form"):

            recipient = st.text_input(
                "수신자 이메일 주소"
            )

            subject = st.text_input(
                "이메일 제목",
                value="[마케팅팀] 전략 보고서 초안 공유"
            )

            st.info(
                "위에서 수정한 보고서가 이메일 본문으로 전송됩니다."
            )

            # 사용자 확인 후에만 전송
            confirm = st.checkbox(
                "보고서 내용과 수신자 주소를 확인했습니다."
            )

            submitted = st.form_submit_button(
                "✉️ 최종 보고서 이메일 발송"
            )

            if submitted:

                if not recipient:

                    st.warning(
                        "수신자 이메일 주소를 입력해 주세요."
                    )

                elif not confirm:

                    st.warning(
                        "이메일 발송 전 확인란에 체크해 주세요."
                    )

                else:

                    # 기존 이메일 발송 함수 재사용
                    result = send_email(
                        recipient,
                        subject,
                        edited_report
                    )

                    if result.get("status") == "success":

                        st.success(
                            "✅ 보고서 이메일 발송 완료!"
                        )

                    else:

                        st.error(
                            f"이메일 발송 실패: "
                            f"{result.get('error')}"
                        )
