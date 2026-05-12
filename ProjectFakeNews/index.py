import re
import json
import time
import streamlit as st
import trafilatura
from google import genai

st.set_page_config(page_title="가짜뉴스 판별기", page_icon="✅", layout="wide")

st.title("🧾 Gemini기반 가짜뉴스 판별기")
st.write("뉴스 기사 링크 또는 직접 입력한 글을 논문에 정리된 가짜뉴스 경향 기준으로 판별합니다.")

# ⚠️ API 키를 공개 저장소에 올리지 마세요. 학교 프로젝트용으로 직접 넣을 경우 아래 문자열 안에 새 키를 넣으세요.
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
GEMINI_MODEL_OPTIONS = {
    "라이트": "gemini-2.5-flash-lite",
    "일반": "gemini-2.5-flash",
}
MAX_GEMINI_RETRIES = 2
AI_LIMIT_NOTICE = "API 사정으로 AI분석은 일 20회로 제한되어 있습니다. 만약 AI분석이 안된다면 다른 모델로 시도해보세요"
AI_SCORE_KEYS = [
    "ai_emotion_score",
    "ai_exaggeration_score",
    "ai_source_transparency_score",
    "ai_risk_score",
    "ai_frame_score",
    "ai_authority_borrow_score",
    "ai_headline_score",
    "ai_causal_score",
]
AI_TO_RULE_SCORE_KEY = {
    "ai_emotion_score": "emotion_score",
    "ai_exaggeration_score": "exaggeration_score",
    "ai_source_transparency_score": "source_transparency_score",
    "ai_risk_score": "risk_score",
    "ai_frame_score": "frame_score",
    "ai_authority_borrow_score": "authority_borrow_score",
    "ai_headline_score": "headline_score",
    "ai_causal_score": "causal_score",
}

# -----------------------------
# 1. 한국형 가짜뉴스 위험 신호 사전 일반
# -----------------------------
# 국내 논문 요약에서 반복된 기준:
# 감정 자극, 과장·왜곡, 출처 불투명성, 권위 차용, 의혹·고발 프레임,
# 제목-본문 불일치, 질문형 제목, 직접 인용형 제목, 허위 인과/인과 단순화

emotion_words = [
    "충격", "분노", "공포", "소름", "경악", "위험", "끔찍", "난리", "불안", "패닉",
    "공포감", "위협", "비상", "혼란", "분개", "격분", "참사", "재앙", "대혼란",
    "두려움", "충격적", "경악할", "위태", "무섭", "불길"
]

exaggeration_words = [
    "무조건", "절대", "100%", "완벽", "역대급", "레전드", "긴급", "단독", "최초",
    "최악", "반드시", "믿기 힘든", "상상 초월", "충격적인", "소름 돋는", "초유의",
    "대박", "난리났다", "끝장", "폭망", "폭발적", "전부", "모두", "완전히", "무너졌다",
    "발칵", "초토화", "충격 고백", "결정적", "숨겨진 진실"
]

source_transparency_words = [
    "출처", "원문", "통계", "논문", "보고서", "연구", "공식", "기관", "정부", "자료",
    "조사", "분석", "발표", "인용", "근거", "데이터", "원자료", "공문", "보도자료",
    "학회", "대학", "연구진", "doi"
]

official_org_words = [
    "통계청", "질병관리청", "보건복지부", "교육부", "과학기술정보통신부", "한국은행",
    "대법원", "헌법재판소", "국회", "UN", "WHO", "OECD", "IMF", "World Bank", "세계은행",
    "KDI", "한국개발연구원", "식약처", "환경부", "고용노동부", "경찰청", "검찰청"
]

vague_source_words = [
    "관계자", "한 관계자", "익명", "소식통", "내부자", "제보자", "일부 전문가", "전문가들은",
    "업계에 따르면", "온라인 커뮤니티", "커뮤니티에 따르면", "카더라", "알려졌다", "전해졌다",
    "~라는 말이 나온다", "누리꾼", "네티즌", "모 온라인", "일각"
]

authority_words = [
    "전문가", "교수", "박사", "연구원", "연구진", "언론인", "기자", "관계자", "고위 관계자",
    "당국자", "내부자", "소식통", "의사", "변호사", "경제학자", "분석가", "평론가"
]

suspicion_frame_words = [
    "의혹", "고발", "폭로", "논란", "은폐", "조작", "배후", "음모", "카르텔", "수상하다",
    "수상한", "몰랐던 진실", "감춰진", "숨겨진", "비밀", "정체", "실체", "진실", "검은", "게이트",
    "사기", "배신", "충격 고발", "폭로됐다"
]

certainty_words = [
    "명백", "확실", "분명", "단정", "결정적", "드러났다", "밝혀졌다", "사실로 확인", "판명",
    "증거", "빼박", "틀림없", "부정할 수 없", "확정", "완전히 밝혀"
]

counter_view_words = [
    "반박", "해명", "반론", "다른 입장", "반면", "한편", "그러나", "다만", "반대 의견",
    "공식 입장", "부인했다", "논란에 대해", "설명했다", "재확인 필요"
]

causal_words = [
    "때문", "탓", "원인", "결과", "그래서", "따라서", "유발", "초래", "불러왔다", "이어졌다",
    "영향", "관련", "연관", "증명", "입증", "밝혀졌다", "드러났다"
]

weak_causal_words = [
    "가능성", "추정", "의심", "보인다", "추측", "정황", "아마", "것 같다", "듯하다", "로 추정"
]

subjective_words = [
    "어이없", "말도 안", "황당", "기가 막", "충격적", "끔찍한", "심각한", "터무니없",
    "당연히", "상식적으로", "누가 봐도", "말문이 막히", "분노할"
]

headline_question_words = ["왜", "진짜", "정말", "누가", "어떻게", "무슨 일", "알고 보니", "혹시"]

STOPWORDS = {
    "그리고", "그러나", "하지만", "이번", "오늘", "지난", "관련", "대한", "위해", "통해", "있는",
    "없는", "한다", "했다", "됐다", "된다", "것으로", "뉴스", "기사", "단독", "속보", "기자"
}


# -----------------------------
# 2. 기본 유틸 함수
# -----------------------------
def clamp_score(value):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def normalize_ai_scores(result):
    for score_key in AI_SCORE_KEYS:
        result[score_key] = clamp_score(result.get(score_key))
    return result


def ai_scores_are_same_as_rule(ai_result, rule_result):
    return all(
        clamp_score(ai_result.get(ai_key)) == clamp_score(rule_result.get(rule_key))
        for ai_key, rule_key in AI_TO_RULE_SCORE_KEY.items()
    )


def format_prompt_items(items, limit=10):
    if not items:
        return "없음"
    visible_items = items[:limit]
    suffix = " 외" if len(items) > limit else ""
    return ", ".join(visible_items) + suffix


def count_keywords(text, word_list):
    count = 0
    found_words = []
    for word in word_list:
        word_count = text.count(word)
        if word_count > 0:
            count += word_count
            found_words.append(word)
    return count, found_words


def density_score(count, text, multiplier=18):
    """텍스트가 길수록 단순 빈도 점수가 과대평가되지 않도록 1,000자당 빈도로 보정한다."""
    normalized_count = count / max(1, len(text) / 1000)
    return clamp_score(normalized_count * multiplier)


def split_title_body(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", text

    first_line = lines[0]
    # 첫 줄이 너무 길면 제목이 아니라 본문일 가능성이 크므로 제목 분석을 약하게 처리한다.
    if len(first_line) <= 120 and len(lines) >= 2:
        return first_line, "\n".join(lines[1:])
    return "", text


def extract_korean_keywords(text):
    words = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    return [word for word in words if word not in STOPWORDS]


def get_risk_level(score):
    if score >= 75:
        return "위험"
    if score >= 45:
        return "주의"
    return "안전"


def get_risk_color(level):
    colors = {
        "안전": "#2563eb",
        "주의": "#f59e0b",
        "위험": "#dc2626",
    }
    return colors.get(level, "#6b7280")


def render_risk_bar(score):
    score = clamp_score(score)
    level = get_risk_level(score)
    color = get_risk_color(level)
    st.markdown(
        f"""
        <div style="width: 100%; height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 0.35rem 0 0.75rem 0;">
            <div style="width: {score}%; height: 100%; background: {color}; border-radius: 999px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if hasattr(st, "dialog"):
    @st.dialog("AI 분석 사용 안내")
    def show_ai_limit_notice():
        st.write(AI_LIMIT_NOTICE)
        if st.button("확인"):
            st.session_state["ai_limit_notice_seen"] = True
            st.rerun()
else:
    def show_ai_limit_notice():
        st.toast(AI_LIMIT_NOTICE)
        st.session_state["ai_limit_notice_seen"] = True


# -----------------------------한국형
# 3. 기사 본문 추출
# -----------------------------
def extract_article_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None, "기사 내용을 가져오지 못했습니다. 사이트가 자동 접근을 차단했을 수 있습니다."

        title = ""
        text = ""

        # 메타데이터 포함 추출 시도
        extracted_json = trafilatura.extract(
            downloaded,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
        )

        if extracted_json:
            try:
                data = json.loads(extracted_json)
                title = (data.get("title") or "").strip()
                text = (data.get("text") or "").strip()
            except json.JSONDecodeError:
                pass

        # 실패하면 일반 텍스트 추출
        if not text:
            text = trafilatura.extract(downloaded) or ""
            text = text.strip()

        if not text or len(text) < 100:
            return None, "본문을 충분히 추출하지 못했습니다. 기사 본문을 직접 붙여넣어 주세요."

        if title:
            return f"{title}\n\n{text}", None
        return text, None

    except Exception as e:
        return None, f"오류가 발생했습니다: {e}"


# -----------------------------
# 4. 한국형 정보 영양 점수 계산
# -----------------------------
def rule_based_analysis(text):
    title, body = split_title_body(text)

    emotion_count, found_emotion = count_keywords(text, emotion_words)
    exaggeration_count, found_exaggeration = count_keywords(text, exaggeration_words)
    explicit_source_count, found_source = count_keywords(text, source_transparency_words)
    official_org_count, found_org = count_keywords(text, official_org_words)
    vague_source_count, found_vague_source = count_keywords(text, vague_source_words)
    authority_count, found_authority = count_keywords(text, authority_words)
    suspicion_count, found_suspicion = count_keywords(text, suspicion_frame_words)
    certainty_count, found_certainty = count_keywords(text, certainty_words)
    counter_view_count, found_counter_view = count_keywords(text, counter_view_words)
    causal_count, found_causal = count_keywords(text, causal_words)
    weak_causal_count, found_weak_causal = count_keywords(text, weak_causal_words)
    subjective_count, found_subjective = count_keywords(text, subjective_words)

    number_count = len(re.findall(r"\d+", text))
    url_count = len(re.findall(r"https?://|www\.", text))
    quote_count = text.count("\"") + text.count("'") + text.count("“") + text.count("”") + text.count("‘") + text.count("’")

    # 1) 감정 자극도: 부정 정서 + 주관적 표현
    emotion_score = clamp_score(
        density_score(emotion_count, text, 20) + density_score(subjective_count, text, 12)
    )

    # 2) 과장·선정성: 과장 표현 + 문장부호 남용
    exclamation_count = text.count("!")
    question_count = text.count("?")
    punctuation_score = clamp_score((exclamation_count + question_count) * 6)
    exaggeration_score = clamp_score(
        density_score(exaggeration_count, text, 20) + punctuation_score
    )

    # 3) 출처 투명도: 명확한 출처·기관·원문·링크·수치가 있으면 증가, 익명/불명확 출처는 감소
    source_transparency_score = clamp_score(
        explicit_source_count * 13
        + official_org_count * 12
        + url_count * 25
        + min(number_count, 10) * 4
        - vague_source_count * 15
    )
    source_opacity_score = 100 - source_transparency_score

    # 4) 의혹·고발 프레임: 의혹·폭로·은폐 프레임 + 단정적 표현 + 반론 부족
    single_view_bonus = 20 if suspicion_count > 0 and counter_view_count == 0 else 0
    frame_score = clamp_score(
        density_score(suspicion_count, text, 22)
        + density_score(certainty_count, text, 10)
        + single_view_bonus
    )

    # 5) 권위 차용 위험: 전문가·관계자 등을 언급하지만 실제 출처·링크가 부족할 때 증가
    authority_borrow_score = clamp_score(
        density_score(authority_count, text, 16)
        + vague_source_count * 18
        - explicit_source_count * 5
        - url_count * 10
    )

    # 6) 허위 인과/인과 단순화 위험: 인과 표현이 많은데 근거가 약하거나 추정 표현과 결합될 때 증가
    causal_score = clamp_score(
        density_score(causal_count, text, 14)
        + density_score(weak_causal_count, text, 10)
        + (20 if causal_count > 0 and source_transparency_score < 40 else 0)
    )

    # 7) 제목 위험도: 제목의 질문형·직접인용·과장·본문 핵심어 겹침 부족
    headline_score = 0
    title_overlap_ratio = None
    found_headline_signals = []
    if title:
        title_exag_count, found_title_exag = count_keywords(title, exaggeration_words)
        title_emotion_count, found_title_emotion = count_keywords(title, emotion_words)
        title_question_count, found_title_question = count_keywords(title, headline_question_words)

        title_keywords = extract_korean_keywords(title)
        body_keywords = set(extract_korean_keywords(body))
        if title_keywords:
            overlap = sum(1 for word in title_keywords if word in body_keywords)
            title_overlap_ratio = overlap / len(title_keywords)
        else:
            title_overlap_ratio = 1

        mismatch_score = 0
        if len(title_keywords) >= 3 and title_overlap_ratio < 0.35:
            mismatch_score = 30
            found_headline_signals.append("제목 핵심어가 본문과 충분히 연결되지 않을 가능성")

        if "?" in title or title_question_count > 0:
            found_headline_signals.append("질문형 제목")
        if quote_count > 0 and any(mark in title for mark in ["\"", "'", "“", "”", "‘", "’"]):
            found_headline_signals.append("직접 인용형 제목")
        if found_title_exag or found_title_emotion:
            found_headline_signals.append("제목의 감정·과장 표현")

        headline_score = clamp_score(
            title_exag_count * 18
            + title_emotion_count * 16
            + title_question_count * 12
            + (12 if "?" in title else 0)
            + (8 if "!" in title else 0)
            + (10 if any(mark in title for mark in ["\"", "'", "“", "”", "‘", "’"]) else 0)
            + mismatch_score
        )

    # 8) 종합 위험도: 한국 학술논문에서 강조된 항목을 반영한 가중합
    risk_score = clamp_score(
        emotion_score * 0.18
        + exaggeration_score * 0.15
        + source_opacity_score * 0.18
        + frame_score * 0.17
        + headline_score * 0.12
        + causal_score * 0.12
        + authority_borrow_score * 0.08
    )
    risk_level = get_risk_level(risk_score)

    # 발견된 위험 신호 설명
    risk_notes = []
    if found_emotion or found_subjective:
        risk_notes.append("불안·분노 등 감정 자극 또는 주관적 표현이 발견되었습니다.")
    if found_exaggeration:
        risk_notes.append("과장·선정적 표현이 발견되었습니다.")
    if source_transparency_score < 40:
        risk_notes.append("명확한 출처·원문·기관·링크가 부족합니다.")
    if found_vague_source:
        risk_notes.append("관계자·소식통 등 불명확한 정보원이 사용되었습니다.")
    if found_suspicion:
        risk_notes.append("의혹·고발·은폐 프레임이 발견되었습니다.")
    if causal_score >= 45:
        risk_notes.append("인과관계가 단순화되었거나 근거가 약한 설명 구조가 의심됩니다.")
    if found_headline_signals:
        risk_notes.append("제목에서 질문형·인용형·과장형 유인 장치가 발견되었습니다.")
    if not risk_notes:
        risk_notes.append("규칙 기반 분석에서 강한 위험 신호는 적게 발견되었습니다.")

    return {
        "title": title,
        "emotion_score": emotion_score,
        "exaggeration_score": exaggeration_score,
        "source_transparency_score": source_transparency_score,
        "source_opacity_score": source_opacity_score,
        "frame_score": frame_score,
        "authority_borrow_score": authority_borrow_score,
        "headline_score": headline_score,
        "causal_score": causal_score,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "found_emotion": found_emotion,
        "found_exaggeration": found_exaggeration,
        "found_source": found_source,
        "found_org": found_org,
        "found_vague_source": found_vague_source,
        "found_authority": found_authority,
        "found_suspicion": found_suspicion,
        "found_certainty": found_certainty,
        "found_counter_view": found_counter_view,
        "found_causal": found_causal,
        "found_weak_causal": found_weak_causal,
        "found_subjective": found_subjective,
        "found_headline_signals": found_headline_signals,
        "title_overlap_ratio": title_overlap_ratio,
        "number_count": number_count,
        "url_count": url_count,
        "risk_notes": risk_notes,
    }


# -----------------------------
# 5. Gemini 실패 시 대체 분석
# -----------------------------
def fallback_deep_analysis(text, rule_result):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if sentence.strip()
    ]
    preview_sentences = sentences[:3] if sentences else [text[:120].strip()]

    signal_words = set(
        rule_result["found_emotion"]
        + rule_result["found_exaggeration"]
        + rule_result["found_suspicion"]
        + rule_result["found_vague_source"]
        + rule_result["found_causal"]
        + ["%", "명", "원", "배", "증가", "감소", "최초", "최악", "긴급", "의혹", "관계자"]
    )

    check_needed_sentences = []
    for sentence in sentences:
        if any(word and word in sentence for word in signal_words):
            check_needed_sentences.append(sentence[:180])
        if len(check_needed_sentences) >= 3:
            break

    if not check_needed_sentences:
        check_needed_sentences = [sentence[:180] for sentence in preview_sentences[:2]]

    neutral_summary = " ".join(preview_sentences)[:260]
    if len(" ".join(preview_sentences)) > 260:
        neutral_summary += "..."

    return {
        "ai_emotion_score": rule_result["emotion_score"],
        "ai_exaggeration_score": rule_result["exaggeration_score"],
        "ai_source_transparency_score": rule_result["source_transparency_score"],
        "ai_risk_score": rule_result["risk_score"],
        "ai_frame_score": rule_result["frame_score"],
        "ai_authority_borrow_score": rule_result["authority_borrow_score"],
        "ai_headline_score": rule_result["headline_score"],
        "ai_causal_score": rule_result["causal_score"],
        "core_claims": [sentence[:140] for sentence in preview_sentences],
        "check_needed_sentences": check_needed_sentences,
        "risk_signals": rule_result["risk_notes"][:5],
        "source_check_questions": [
            "이 기사에서 주장하는 핵심 내용의 원문 출처가 명확히 제시되어 있는가?",
            "수치·인용·전문가 발언의 발표 기관, 날짜, 원자료를 확인할 수 있는가?",
            "의혹·고발·인과관계 주장이 다른 신뢰 가능한 자료에서도 확인되는가?",
            "제목의 표현이 본문 내용보다 과장되어 있지는 않은가?",
        ],
        "neutral_summary": neutral_summary,
        "final_advice": (
            "Gemini API가 응답하지 않아 대체 분석을 표시했습니다. "
            "이 결과는 진위 판정이 아니라 출처, 프레임, 제목, 인과관계의 위험 신호를 점검하는 용도입니다."
        ),
    }


# -----------------------------
# 6. Gemini AI 심화 판단
# -----------------------------
def gemini_analysis(text, rule_result, gemini_model):
    if not GEMINI_API_KEY:
        return None, "Gemini API 키가 비어 있습니다. Streamlit Secrets에 GEMINI_API_KEY를 설정하세요."

    def is_quota_error(error):
        error_text = str(error).lower()
        quota_signals = [
            "quota exceeded",
            "exceeded your current quota",
            "generate_content_free_tier",
            "billing details",
            "resource_exhausted",
            "permission_denied",
            "api key",
            "leaked",
            "403",
            "429",
        ]
        return any(signal in error_text for signal in quota_signals)

    def is_retryable_error(error):
        error_text = str(error).lower()
        retryable_signals = [
            "503",
            "unavailable",
            "high demand",
            "temporarily",
            "deadline",
            "timeout",
        ]
        return any(signal in error_text for signal in retryable_signals)

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        shortened_text = text[:2500]

        rule_signal_summary = f"""
일반 분석에서 발견된 참고 신호(점수 아님):
- 감정·주관 표현: {format_prompt_items(rule_result["found_emotion"] + rule_result["found_subjective"])}
- 과장·선정 표현: {format_prompt_items(rule_result["found_exaggeration"])}
- 명확한 출처 표현: {format_prompt_items(rule_result["found_source"])}
- 공식 기관 표현: {format_prompt_items(rule_result["found_org"])}
- 불명확한 출처 표현: {format_prompt_items(rule_result["found_vague_source"])}
- 의혹·고발 프레임 표현: {format_prompt_items(rule_result["found_suspicion"])}
- 권위 차용 표현: {format_prompt_items(rule_result["found_authority"])}
- 인과 표현: {format_prompt_items(rule_result["found_causal"] + rule_result["found_weak_causal"])}
- 제목 위험 신호: {format_prompt_items(rule_result["found_headline_signals"])}
- 숫자 포함 개수: {rule_result["number_count"]}개
- 링크 포함 개수: {rule_result["url_count"]}개
"""

        json_schema = """
{
  "core_claims": ["핵심 주장 1", "핵심 주장 2", "핵심 주장 3"],
  "check_needed_sentences": ["검증이 필요한 문장 1", "검증이 필요한 문장 2", "검증이 필요한 문장 3"],
  "risk_signals": ["위험 신호 1", "위험 신호 2", "위험 신호 3", "위험 신호 4"],
  "source_check_questions": ["확인 질문 1", "확인 질문 2", "확인 질문 3", "확인 질문 4"],
  "ai_emotion_score": 0,
  "ai_exaggeration_score": 0,
  "ai_source_transparency_score": 0,
  "ai_risk_score": 0,
  "ai_frame_score": 0,
  "ai_authority_borrow_score": 0,
  "ai_headline_score": 0,
  "ai_causal_score": 0,
  "neutral_summary": "자극적 표현을 줄인 중립적 요약",
  "final_advice": "사용자가 이 정보를 읽을 때 주의해야 할 점"
}
"""

        def request_json(current_prompt):
            response = None
            last_error = None

            for attempt in range(MAX_GEMINI_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=gemini_model,
                        contents=current_prompt,
                        config={
                            "response_mime_type": "application/json",
                            "temperature": 0.35,
                        },
                    )
                    break

                except Exception as e:
                    last_error = e

                    if is_quota_error(e):
                        return None, (
                            f"Gemini 사용량 한도, API 키, 또는 권한 문제로 분석하지 못했습니다. 실제 오류: {str(e)[:300]}"
                        )

                    if not is_retryable_error(e):
                        return None, (
                            f"Gemini 호출 오류로 분석하지 못했습니다. 실제 오류: {type(e).__name__}: {str(e)[:300]}"
                        )

                    if attempt < MAX_GEMINI_RETRIES - 1:
                        time.sleep(2 ** attempt)

            if response is None:
                return None, (
                    f"선택한 Gemini 모델이 응답하지 않았습니다. 마지막 오류: {str(last_error)[:300]}"
                )

            raw_text = getattr(response, "text", "")

            if not raw_text or not raw_text.strip():
                return None, "Gemini 응답이 비어 있어 AI 분석 결과를 표시하지 못했습니다."

            raw_text = raw_text.strip().replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError:
                return None, (
                    f"Gemini 응답이 JSON 형식이 아니어서 분석 결과를 표시하지 못했습니다. 응답 앞부분: {raw_text[:300]}"
                )

            return normalize_ai_scores(parsed), None

        prompt = f"""
너는 뉴스 기사와 SNS 글의 허위정보 위험 신호를 분석하는 보조 AI이다.
절대로 글을 "가짜뉴스다/진짜뉴스다"라고 단정하지 마라.
사용자가 비판적으로 정보를 판단하도록 돕는 것이 목적이다.

분석 기준은 한국 학술논문에서 정리된 가짜뉴스 경향을 따른다.
AI도 반드시 일반 분석과 같은 8개 카테고리로 각각 평가하라.
단, 일반 분석 점수를 따라 쓰지 말고 원문 맥락을 읽어 독립적으로 점수를 매겨라.

8개 카테고리:
1. 감정 자극도
2. 과장·선정성
3. 출처 투명도
4. 종합 위험도
5. 의혹·고발 프레임
6. 권위 차용 위험
7. 제목 위험도
8. 인과 왜곡 위험

채점 원칙:
- 참고 신호는 단어 사전이 찾은 목록일 뿐이며 정답이나 점수가 아니다.
- 단어가 없어도 맥락상 위험하면 점수를 올리고, 단어가 있어도 근거가 충분하면 점수를 낮춰라.
- ai_risk_score는 나머지 세부 점수와 글의 전체 신뢰 위험을 종합해 일관되게 매겨라.
- 일반 분석 점수와 일부 항목이 같을 수는 있지만, 모든 항목을 그대로 맞추려 하지 마라.

{rule_signal_summary}

분석할 글:
{shortened_text}

반드시 아래 JSON 형식으로만 답하라. JSON 밖의 설명은 쓰지 마라.

{json_schema}

점수는 모두 0부터 100 사이의 정수로 작성하라.
"""

        parsed, error = request_json(prompt)
        if error:
            return None, error

        if ai_scores_are_same_as_rule(parsed, rule_result):
            independent_prompt = f"""
첫 번째 AI 점수가 일반 분석 점수와 완전히 같았다.
이번에는 일반 분석 참고 신호도 보지 말고, 아래 원문만 기준으로 다시 독립 채점하라.

평가 기준:
1. 감정 자극도
2. 과장·선정성
3. 출처 투명도
4. 종합 위험도
5. 의혹·고발 프레임
6. 권위 차용 위험
7. 제목 위험도
8. 인과 왜곡 위험

분석할 글:
{shortened_text}

반드시 아래 JSON 형식으로만 답하라. JSON 밖의 설명은 쓰지 마라.

{json_schema}

점수는 모두 0부터 100 사이의 정수로 작성하라.
"""
            independent_result, independent_error = request_json(independent_prompt)
            if independent_result and not independent_error:
                parsed = independent_result

        return parsed, None

    except Exception as e:
        return None, (
            f"Gemini 분석 중 오류가 발생했습니다. 실제 오류: {type(e).__name__}: {str(e)[:300]}"
        )
# -----------------------------
# 7. Streamlit UI
# -----------------------------한국형
if not st.session_state.get("ai_limit_notice_seen", False):
    st.session_state["ai_limit_notice_seen"] = True
    show_ai_limit_notice()

st.sidebar.header("⚙ 설정")
use_gemini = st.sidebar.checkbox("Gemini 분석 사용", value=True)
selected_gemini_label = "라이트"
selected_gemini_model = GEMINI_MODEL_OPTIONS[selected_gemini_label]

if use_gemini:
    selected_gemini_label = st.sidebar.radio(
        "Gemini 모델 선택",
        list(GEMINI_MODEL_OPTIONS.keys()),
        horizontal=True,
        help="라이트 모델은 빠르고, 일반 모델은 더 자세한 분석에 유리합니다.",
    )
    selected_gemini_model = GEMINI_MODEL_OPTIONS[selected_gemini_label]
    st.sidebar.caption(f"현재 선택 모델: `{selected_gemini_model}`")

if not GEMINI_API_KEY:
    st.sidebar.warning("Gemini API 키가 비어 있습니다. 규칙 기반 대체 분석은 계속 작동합니다.")

st.subheader("1단계: 분석 방식 선택")
mode = st.radio("분석할 방식을 선택하세요.", ["뉴스 기사 링크 분석", "본문 직접 입력"], horizontal=True)

article_text = ""

if mode == "뉴스 기사 링크 분석":
    url = st.text_input("뉴스 기사 링크를 입력하세요", placeholder="예: https://example.com/news/article")

    if st.button("기사 가져오기"):
        if url.strip() == "":
            st.warning("뉴스 기사 링크를 입력하세요.")
        else:
            with st.spinner("기사 본문을 가져오는 중입니다..."):
                extracted_text, error = extract_article_text(url)

            if error:
                st.error(error)
                st.info("이 경우 기사 본문을 복사해서 '본문 직접 입력' 방식으로 분석하면 됩니다.")
            else:
                st.success("기사 본문을 가져왔습니다.")
                st.session_state["article_text"] = extracted_text

    article_text = st.session_state.get("article_text", "")

    if article_text:
        st.subheader("추출된 기사 본문 미리보기")
        st.text_area("본문", article_text, height=250)

else:
    article_text = st.text_area(
        "분석할 글을 직접 입력하세요",
        height=300,
        placeholder="첫 줄에 제목을 넣고, 다음 줄부터 본문을 넣으면 제목 위험도까지 분석됩니다.",
    )

st.subheader("2단계: 분석 실행")

if st.button("분석하기"):
    if article_text.strip() == "":
        st.warning("분석할 기사 본문이 없습니다.")
    else:
        rule_result = rule_based_analysis(article_text)

        st.subheader("🟢 정보 판별표")
        st.caption("점수가 높을수록 해당 위험 신호가 강합니다. 단, 출처 투명도는 높을수록 긍정적입니다.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("감정 자극도", f"{rule_result['emotion_score']}점")
        col2.metric("과장,선정성", f"{rule_result['exaggeration_score']}점")
        col3.metric("출처 투명도", f"{rule_result['source_transparency_score']}점")
        col4.metric("종합 위험도", f"{rule_result['risk_score']}점")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("의혹,고발 프레임", f"{rule_result['frame_score']}점")
        col6.metric("권위 차용 위험", f"{rule_result['authority_borrow_score']}점")
        col7.metric("제목 위험도", f"{rule_result['headline_score']}점")
        col8.metric("인과 왜곡 위험", f"{rule_result['causal_score']}점")

        st.subheader("✅ 규칙 기반 종합 판단")
        st.write(f"이 글의 허위정보 위험 신호 수준은 **{rule_result['risk_level']}**입니다.")
        render_risk_bar(rule_result["risk_score"])

        st.subheader("🔍 발견된 위험")
        for note in rule_result["risk_notes"]:
            st.write(f"- {note}")

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write("**감정 자극 표현:**", rule_result["found_emotion"] if rule_result["found_emotion"] else "발견되지 않음")
            st.write("**과장 표현:**", rule_result["found_exaggeration"] if rule_result["found_exaggeration"] else "발견되지 않음")
            st.write("**의혹·고발 프레임:**", rule_result["found_suspicion"] if rule_result["found_suspicion"] else "발견되지 않음")
            st.write("**인과 표현:**", rule_result["found_causal"] if rule_result["found_causal"] else "발견되지 않음")
        with detail_col2:
            st.write("**명확한 출처 표현:**", rule_result["found_source"] if rule_result["found_source"] else "발견되지 않음")
            st.write("**공식 기관 표현:**", rule_result["found_org"] if rule_result["found_org"] else "발견되지 않음")
            st.write("**불명확한 출처 표현:**", rule_result["found_vague_source"] if rule_result["found_vague_source"] else "발견되지 않음")
            st.write("**제목 위험 신호:**", rule_result["found_headline_signals"] if rule_result["found_headline_signals"] else "발견되지 않음")
            st.write("**숫자 포함 개수:**", rule_result["number_count"])
            st.write("**링크 포함 개수:**", rule_result["url_count"])

        if use_gemini:
            st.subheader("🔷 Gemini 판별")
            with st.spinner(f"Gemini {selected_gemini_label} 모델이 논문 기준으로 핵심 주장과 위험을 분석하는 중입니다..."):
                ai_result, ai_error = gemini_analysis(article_text, rule_result, selected_gemini_model)

            if ai_error:
                st.warning(ai_error)

            if ai_result:
                st.write("### AI 판별 점수")
                ai_col1, ai_col2, ai_col3, ai_col4 = st.columns(4)
                ai_col1.metric("감정 자극도", f"{ai_result.get('ai_emotion_score', 0)}점")
                ai_col2.metric("과장,선정성", f"{ai_result.get('ai_exaggeration_score', 0)}점")
                ai_col3.metric("출처 투명도", f"{ai_result.get('ai_source_transparency_score', 0)}점")
                ai_col4.metric("종합 위험도", f"{ai_result.get('ai_risk_score', 0)}점")

                ai_col5, ai_col6, ai_col7, ai_col8 = st.columns(4)
                ai_col5.metric("의혹,고발 프레임", f"{ai_result.get('ai_frame_score', 0)}점")
                ai_col6.metric("권위 차용 위험", f"{ai_result.get('ai_authority_borrow_score', 0)}점")
                ai_col7.metric("제목 위험도", f"{ai_result.get('ai_headline_score', 0)}점")
                ai_col8.metric("인과 왜곡 위험", f"{ai_result.get('ai_causal_score', 0)}점")

                ai_risk_score = clamp_score(ai_result.get("ai_risk_score", 0))
                ai_risk_level = get_risk_level(ai_risk_score)
                st.write(f"AI 분석 기준으로 이 글의 허위정보 위험 신호 수준은 **{ai_risk_level}**입니다.")
                render_risk_bar(ai_risk_score)

                st.write("### 핵심 주장")
                for claim in ai_result.get("core_claims", []):
                    st.write(f"- {claim}")

                st.write("### 검증이 필요한 문장")
                for sentence in ai_result.get("check_needed_sentences", []):
                    st.write(f"- {sentence}")

                st.write("### AI가 본 위험")
                for signal in ai_result.get("risk_signals", []):
                    st.write(f"- {signal}")

                st.write("### 출처 확인 질문")
                for question in ai_result.get("source_check_questions", []):
                    st.write(f"- {question}")

                st.write("### 중립적 요약")
                st.info(ai_result.get("neutral_summary", ""))

                st.write("### 최종 조언")
                st.warning(ai_result.get("final_advice", ""))
            else:
                st.error("심화 분석 결과를 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.")

        st.caption(
            "※ 이 시스템은 글의 진실 여부를 단정하지 않습니다. 국내 논문에서 보고된 가짜뉴스의 반복적 경향을 기준으로 위험 신호를 보여주는 보조 도구입니다."
        )
