# ============================================================
# 캘린더 일정 생성 모듈 (calendar_service.py)
# ============================================================
# 목적:
# AI 비서가 회의록에서 추출한 업무 일정을
# Google Calendar, Outlook 등의 캘린더에 등록할 수 있도록
# ICS 형식의 일정 데이터를 생성합니다.
#
# 사용 방법:
# from calendar_service import create_calendar_event
# ============================================================

from datetime import datetime, timedelta
from uuid import uuid4


def create_calendar_event(
    title,
    start_date,
    description="",
    duration_minutes=60
):
    """
    캘린더 일정 생성 함수

    Parameters
    ----------
    title : str
        일정 제목

    start_date : str
        일정 시작 날짜 및 시간
        예: "2026-11-04 14:00"

    description : str
        일정 상세 설명

    duration_minutes : int
        일정 소요 시간 (기본 60분)

    Returns
    -------
    str
        ICS 형식의 캘린더 데이터
    """

    # 1. 입력된 날짜를 datetime 객체로 변환
    start = datetime.strptime(
        start_date,
        "%Y-%m-%d %H:%M"
    )

    # 2. 일정 종료 시간 계산
    end = start + timedelta(minutes=duration_minutes)

    # 3. 캘린더 일정 데이터 생성
    event = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ABLE AI Secretary//KR
BEGIN:VEVENT
UID:{uuid4()}@able.ai
DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{start.strftime('%Y%m%dT%H%M%S')}
DTEND:{end.strftime('%Y%m%dT%H%M%S')}
SUMMARY:{title}
DESCRIPTION:{description}
END:VEVENT
END:VCALENDAR
"""

    # 4. 생성된 일정 데이터 반환
    return event
