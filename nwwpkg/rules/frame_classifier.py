# nwwpkg/rules/frame_classifier.py
import json, re, sys
from typing import List, Dict, Any

# 룰 정의: 정규식(메타문자 포함)도 허용, 일반 단어도 허용
FRAME_RULES: Dict[str, List[str]] = {
    "military": [r"\b(troop|mobiliz|missile|artillery|drill)\b", r"[가-힣]{1,}부대|군사|훈련|동원"],
    "diplomacy": [r"\b(talks?|negotiat|summit|sanction)\b", r"회담|협상|제재|외교"],
    "economy":  [r"\b(gdp|inflation|tariff|export|currency)\b",  r"관세|수출|환율|물가|성장률"],
}

# 정규식 메타문자 존재 여부로 '그냥 문자열'인지 '정규식'인지 판별
_REGEX_META = set(r".^$*+?{}[]\|()")

def _compile_pattern(p: str) -> re.Pattern:
    p = (p or "").strip()
    if not p:
        # 절대 매칭 안 되는 더미 패턴
        return re.compile(r"(?!x)")
    # 메타문자가 들어 있으면 '정규식'으로 간주해 그대로 컴파일
    if any(ch in _REGEX_META for ch in p):
        src = p
    else:
        # 일반 문자열은 이스케이프하여 오탐 방지
        # 영어/숫자에는 \b 경계를 줄 수도 있지만, 한글은 \b가 잘 동작하지 않아 공통으로 부분매칭 채택
        src = re.escape(p)
    return re.compile(src, flags=re.I | re.S)

# 규칙을 사전 컴파일
_COMPILED_RULES: Dict[str, List[re.Pattern]] = {
    frame: [_compile_pattern(p) for p in patterns]
    for frame, patterns in FRAME_RULES.items()
}

def _extract_keywords(rec: Dict[str, Any]) -> List[str]:
    kws_raw = rec.get("keywords", []) or []
    out: List[str] = []
    for k in kws_raw:
        if isinstance(k, dict):
            out.append(str(k.get("text", "") or ""))
        else:
            out.append(str(k or ""))
    return out

def score_by_rules(txt: str, keywords: List[str]) -> Dict[str, float]:
    s = {k: 0.0 for k in _COMPILED_RULES.keys()}
    text = (txt or "")
    bag  = " ".join(keywords or [])
    for frame, pats in _COMPILED_RULES.items():
        for pat in pats:
            try:
                if pat.search(text) or pat.search(bag):
                    s[frame] += 1.0
            except re.error:
                # 방어: 혹시 런타임에 잘못된 패턴이 들어와도 전체 파이프라인은 계속
                continue
    return s

def run(fin: str, fout: str, on: float = 0.5) -> None:
    with open(fin, "r", encoding="utf-8") as f, open(fout, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            txt = r.get("clean_text") or r.get("text") or ""
            kws = _extract_keywords(r)
            sr  = score_by_rules(txt, kws)

            total = sum(sr.values()) or 1.0
            norm = {k: v/total for k,v in sr.items()}
            label = max(norm, key=norm.get)
            conf  = norm[label]

            item = {"label": label if conf>=on else "uncertain",
                    "conf": round(conf, 3),
                    "source": "rule"}

            # 🔧 둘 다 기록 (파이프라인은 frames를 사용)
            r["frame"] = item
            r["frames"] = [item]

            w.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = sys.argv
    run(
        fin=args[args.index("--in") + 1],
        fout=args[args.index("--out") + 1],
        on=float(args[args.index("--on") + 1]) if "--on" in args else 0.5,
    )
