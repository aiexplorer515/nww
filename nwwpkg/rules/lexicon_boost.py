# nwwpkg/rules/lexicon_boost.py
import json, re, sys, csv
from collections import defaultdict
from typing import List, Tuple, Dict, Any

# 정규식 메타문자 집합
_REGEX_META = set(r".^$*+?{}[]\|()")

def _compile_pattern(p: str) -> re.Pattern:
    """패턴 문자열을 안전하게 컴파일한다.
    - 메타문자가 있으면 정규식으로 간주
    - 없으면 literal 로 escape 하여 부분매칭
    - 비거나 잘못된 패턴은 절대매칭되지 않게 처리
    """
    p = (p or "").strip()
    if not p:
        return re.compile(r"(?!x)")
    try:
        if any(ch in _REGEX_META for ch in p):
            src = p
        else:
            src = re.escape(p)
        return re.compile(src, flags=re.I | re.S)
    except re.error:
        # 잘못된 정규식은 무시되도록
        return re.compile(r"(?!x)")

def load_rules(csv_path: str) -> List[Tuple[re.Pattern, float, str, str]]:
    """CSV의 각 row → (compiled_regex, weight, domain, inject_text)
    - 컬럼: pattern(필수), weight(선택, 기본 0.2), domain(선택), label(주입 텍스트 우선)
    - UTF-8 with BOM도 허용
    """
    rules: List[Tuple[re.Pattern, float, str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pat = (row.get("pattern") or "").strip()
            if not pat:
                continue
            rx = _compile_pattern(pat)
            try:
                weight = float(row.get("weight", 0.2))
            except Exception:
                weight = 0.2
            domain = (row.get("domain") or "generic").strip() or "generic"
            inject = (row.get("label") or pat).strip()  # label이 있으면 그걸 쓰고, 없으면 pattern
            rules.append((rx, weight, domain, inject))
    return rules

def _normalize_keywords(obj: Any) -> List[Dict[str, Any]]:
    """keywords 필드를 list[dict]로 표준화."""
    if obj is None:
        return []
    out: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                text = str(it.get("text", "") or "")
                try:
                    score = float(it.get("score", 0.0))
                except Exception:
                    score = 0.0
                src = str(it.get("src", "") or "")
            else:
                text = str(it)
                score, src = 0.0, ""
            if text:
                out.append({"text": text, "score": score, "src": src})
    else:
        out.append({"text": str(obj), "score": 0.0, "src": ""})
    return out

def run(fin: str, rules_csv: str, fout: str, bonus: float = 0.25, topk: int = 20) -> None:
    rules = load_rules(rules_csv)
    with open(fin, "r", encoding="utf-8") as f, open(fout, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            txt: str = (r.get("clean_text") or r.get("text") or "")
            kyw = _normalize_keywords(r.get("keywords"))

            # 빠른 조회용 집합
            seen = {k["text"] for k in kyw if "text" in k}

            # 룰 매칭 → 부스팅 및 후보 주입
            for (rx, wt, domain, inject_text) in rules:
                try:
                    hit = bool(rx.search(txt))
                except re.error:
                    hit = False
                if not hit:
                    continue

                # 1) 기존 키워드 점수 부스팅
                for item in kyw:
                    try:
                        item["score"] = float(item.get("score", 0.0)) * (1.0 + wt)
                    except Exception:
                        item["score"] = 0.0
                    src = item.get("src", "") or ""
                    if "lex" not in src:
                        item["src"] = (src + "+lex").lstrip("+")

                # 2) 신규 후보 주입(중복 방지)
                if inject_text and inject_text not in seen:
                    kyw.append({
                        "text": inject_text,
                        "score": float(bonus),
                        "src": "lex-inject",
                        "domain": domain
                    })
                    seen.add(inject_text)

            # 점수 기준 정렬 & 상위 유지
            kyw.sort(key=lambda x: (float(x.get("score", 0.0))), reverse=True)
            r["keywords"] = kyw[:topk]
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = sys.argv
    run(
        fin=args[args.index("--in") + 1],
        rules_csv=args[args.index("--rules") + 1],
        fout=args[args.index("--out") + 1],
        bonus=float(args[args.index("--bonus") + 1]) if "--bonus" in args else 0.25,
    )
