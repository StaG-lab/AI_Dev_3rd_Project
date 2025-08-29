# %%
import json, re
from pathlib import Path
# from statistics import median
from collections import Counter
import numpy as np

ROOT_DIR = Path(r"D:\ex\최종프로젝트\AI_Dev_3rd_Project\CHOI\feellog-project_03\reports\analysis")

# %%
def scan(date_range: dict, root_dir: Path):
    """
    지정된 기간(date_range) 안의 analysis_result_*.json 파일들을 스캔
    return: dict[date] -> list[Path]
    """
    start = date_range.get("start")
    end   = date_range.get("end")
    files_by_day = {}

    for p in Path(root_dir).glob("analysis_result_*.json"):
        m = re.search(r"(\d{8})", p.name)
        if not m:
            continue
        d = m.group(1)
        day = f"{d[:4]}-{d[4:6]}-{d[6:]}"   # YYYY-MM-DD
        if start <= day <= end:
            files_by_day.setdefault(day, []).append(p)
    return files_by_day



# %%
EMOTIONS = ["기쁨","당황","분노","불안","상처","슬픔","중립"]

def load_and_normalize(path: Path):
    """
    파일에서 감정 결과를 읽어 7감정 분포로 확장/정규화
    return: dict {score:int, top:str, dist:dict}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    score = raw.get("sentiment_score")
    top = raw.get("dominant_overall_emotion")
    
    dist = {e:0.0 for e in EMOTIONS}
    for item in raw.get("emotion_distribution", []):
        emo = item["emotion"]
        val = float(item["percentage"].replace("%",""))/100
        if emo in dist:
            dist[emo] += val
    # 정규화
    s = sum(dist.values())
    if s>0:
        dist = {k:v/s for k,v in dist.items()}
    return {"score":score, "top":top, "dist":dist}


# %%
# def aggregate_daily(paths:list[Path]):
#     """
#     하루 여러 파일 → DailyCardJSON
#     """
#     recs = [load_and_normalize(p) for p in paths]
    
    
#     scores = [r["score"] for r in recs if r["score"] is not None]
#     score = round(np.mean(scores)) if scores else None
    
#     tops = [r["top"] for r in recs if r["top"]]
#     top = Counter(tops).most_common(1)[0][0] if tops else None
    
#     dist_sum = {e:0 for e in EMOTIONS}
#     for r in recs:
#         for e,v in r["dist"].items():
#             dist_sum[e]+=v
#     s = sum(dist_sum.values())
#     if s>0:
#         dist_sum = {k:v/s for k,v in dist_sum.items()}
    
#     return {"score":score,"top":top,"dist":dist_sum}


# %%
def aggregate_daily(paths: list[Path]):
    """
    하루 여러 파일 → DailyCardJSON
    로더(load_and_normalize) + 집계
    """
    recs = [load_and_normalize(p) for p in paths]

    # 점수: 단순 평균
    scores = [r["score"] for r in recs if r["score"] is not None]
    score = round(np.mean(scores)) if scores else None

    # 분포: 합산 → 정규화
    dist_sum = {e: 0.0 for e in EMOTIONS}
    for r in recs:
        for e, v in r["dist"].items():
            dist_sum[e] += v
    s = sum(dist_sum.values())
    if s > 0:
        dist_sum = {k: v / s for k, v in dist_sum.items()}

    # 주감정: 합산된 분포의 argmax
    top = max(dist_sum, key=dist_sum.get) if s > 0 else None

    return {"score": score, "top": top, "dist": dist_sum}


# %%
from datetime import date, timedelta
from collections import Counter

def _expand_days(start: str, end: str) -> list[str]:
    s = date.fromisoformat(start); e = date.fromisoformat(end)
    out = []
    cur = s
    while cur <= e:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out

# %%
def aggregate_period(daily_cards: dict, date_range=None) -> dict:
    """
    daily_cards: {"YYYY-MM-DD": {"score":..., "top":..., "dist": {...}}, ...}
    date_range:  {"start":"YYYY-MM-DD","end":"YYYY-MM-DD", ...} | None
    """
    # 0) trend 채울 날짜 리스트
    if date_range and date_range.get("start") and date_range.get("end"):
        all_days = _expand_days(date_range["start"], date_range["end"])
    else:
        all_days = sorted(daily_cards.keys())

    # 1) trend.by_day: 빈 날은 None/0.0으로 채움
    by_day = []
    covered = []
    for d in all_days:
        card = daily_cards.get(d)
        if card is None:
            by_day.append({
                "date": d,
                "score": None,
                "top": None
            })
        else:
            by_day.append({
                "date": d,
                "score": card.get("score"),
                "top": card.get("top")
            })
            covered.append(d)

    # 2) 평균 점수(실데이터만), 우세감정(실데이터만)
    valid_scores = [c.get("score") for c in daily_cards.values() if c.get("score") is not None]
    avg_score = round(sum(valid_scores) / len(valid_scores)) if valid_scores else None

    tops = [c.get("top") for c in daily_cards.values() if c.get("top")]
    cnt = Counter(tops)
    tot = sum(cnt.values()) or 1
    dominant = [{"emotion": k, "percentage": v / tot} for k, v in cnt.items()]

    # 3) notable_changes: Δ=10 고정(단순). 필요하면 나중에 조정
    NOTI_DELTA = 10
    notable = []
    for i in range(1, len(by_day)):
        prev, cur = by_day[i-1], by_day[i]
        ps, cs = prev["score"], cur["score"]
        if ps is not None and cs is not None:
            delta = cs - ps
            if abs(delta) >= NOTI_DELTA:
                notable.append({"date": cur["date"], "event": "점수 급변", "delta": delta})
        # 주감정 전환
        if prev["top"] and cur["top"] and prev["top"] != cur["top"]:
            notable.append({"date": cur["date"], "event": f"주감정 전환: {prev['top']}→{cur['top']}"})

    # 4) 결과
    out = {
        "average_sentiment_score": avg_score,
        "dominant_emotions": dominant,
        "trend": {
            "by_day": by_day,
            "notable_changes": notable
        },
        "meta": {
            "days_requested": (
                date_range.get("start") if date_range else None,
                date_range.get("end") if date_range else None
            ),
            "days_covered": covered
        }
    }
    return out

# %%
# def aggregate_period(cards:dict[str,dict]):
#     """
#     여러 일간 카드 → 기간 리포트 JSON
#     """
#     if not cards: return {}
    
#     scores = [c["score"] for c in cards.values() if c["score"]]
#     avg_score = round(np.mean(scores)) if scores else None
    
#     tops = [c["top"] for c in cards.values() if c["top"]]
#     top_counts = Counter(tops)
#     doms = [{"emotion":k,"percentage":v/len(tops)} for k,v in top_counts.items()]
    
#     trend = [{"date":d,"score":c["score"],"top":c["top"]} for d,c in sorted(cards.items())]
    
#     # 급변 탐지
#     notable=[]
#     for i in range(1,len(trend)):
#         delta = trend[i]["score"]-trend[i-1]["score"]
#         if abs(delta)>=10:
#             notable.append({"date":trend[i]["date"],"event":"점수 급변","delta":delta})
#         if trend[i]["top"]!=trend[i-1]["top"]:
#             notable.append({"date":trend[i]["date"],"event":f"주감정 전환: {trend[i-1]['top']}→{trend[i]['top']}"})
    
#     return {"average_sentiment_score":avg_score,
#             "dominant_emotions":doms,
#             "trend":{"by_day":trend,"notable_changes":notable}}


# %%
# 1. intent 모듈 결과 예시
date_range = {
    "mode": "weekly",
    "date": None,
    "start": "2025-08-18",
    "end": "2025-08-26",
    "source": "저번주인가 일이 많아서 우울했던거 같은데 어땠더라?"
}

# date_range={'mode': 'error', 'error': 'cannot_parse', 'source': 'ldsjfa;ljfpoivjlfnaweifhp', 'date': None, 'start': None, 'end': None}

# # 2. 스캔
# files = scan(date_range, ROOT_DIR)

# # 3. 일간 집계
# daily_cards = {d:aggregate_daily(ps) for d,ps in files.items()}

# # 4. 기간 집계
# period = aggregate_period(daily_cards)


# %%
def run_data_engine(date_range: dict, root_dir: Path):
    """A → B 연결용 단일 호출 엔트리포인트"""
    # intent 에러/잘못된 범위면 안전 종료
    if date_range.get("mode") == "error" or not date_range.get("start") or not date_range.get("end"):
        return {}, {"average_sentiment_score": None,
                    "dominant_emotions": [],
                    "trend": {"by_day": [], "notable_changes": []},
                    "meta": {
                        "status": "intent_error" if date_range.get("mode") == "error" else "invalid_range",
                        "days_requested": (date_range.get("start"), date_range.get("end")),
                        "days_covered": []
                    }}

    files = scan(date_range, root_dir)
    daily_cards = {d: aggregate_daily(ps) for d, ps in files.items()}
    period = aggregate_period(daily_cards,date_range)
    
    
    return period




