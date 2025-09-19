# tools/llm_frames.py
# -*- coding: utf-8 -*-
import os, sys, time, json, hashlib
from typing import List, Dict
import orjson as oj
import requests
import os, sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("ERROR: OPENAI_API_KEY 미설정 (.env 또는 환경변수 설정 필요)")

PROVIDER = os.getenv("NWW_LLM_PROVIDER","openai").lower()
MODEL    = os.getenv("NWW_LLM_MODEL","gpt-4o-mini")
headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY'].strip()}"}

LABELS = [
  ("군사충돌","Armed_Clash"),
  ("군사동원","Military_Mobilization"),
  ("국경충돌","Border_Skirmish"),
  ("핵/미사일","WMD_Missile"),
  ("외교협상","Diplomatic_Talks"),
  ("제재/보복","Sanctions_Retaliation"),
  ("경제위기","Economic_Crisis"),
  ("사회불안","Domestic_Unrest"),
  ("사이버공격","Cyber_Attack"),
  ("재난/재해","Disaster"),
  ("없음","None")
]
LABEL_LIST_STR = ", ".join([f"{k}({v})" for k,v in LABELS])

SYSTEM = """너는 위기 인지 프레임 분류기다. 한국어 기사 제목+본문을 보고
허용 라벨 집합에서 0~3개를 선택하고, 각 신뢰도(confidence 0~1)를 부여하라.
출력은 JSON ONLY. 형식:
{"frames":[{"label":"군사동원(Military_Mobilization)","conf":0.78}, ...],
 "evidence":["근거 문구 1","근거 문구 2"]}"""

USER_TMPL = """허용 라벨: {label_list}
기사:
제목: {title}
본문: {text}

요구사항:
1) frames[].label 은 한국어(영문) 병기 형식(예: 군사동원(Military_Mobilization))
2) conf 는 0~1 부동소수.
3) 최대 3개. 없으면 없음(None) 1개만 conf 0.0 로.
4) evidence 는 근거 문구 1~3개.
반드시 JSON만 출력."""

def _req_openai(prompt: str) -> str:
    import os, requests

    def _ascii_only(x: str) -> str:
        return "".join(ch for ch in x.strip() if ord(ch) < 128)

    api_key = _ascii_only(os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (or non-ASCII chars present)")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base  = os.getenv("OPENAI_API_BASE", "https://api.openai.com") or "https://api.openai.com"
    if not base.startswith("http"):
        base = "https://" + base
    base = base.rstrip("/")

    url = f"{base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json; charset=utf-8",
    }

    # 헤더 ASCII 검증(디버그/보호)
    for k, v in headers.items():
        try:
            str(v).encode("latin-1")
        except UnicodeEncodeError:
            raise ValueError(f"HTTP header {k} has non-ASCII chars: {repr(v)}")

    data = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Reply in compact JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(url, json=data, headers=headers, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

# --- add: robust extractor ---
def _extract_frames(obj_or_text):
    import json, re
    # 1) json dict일 때 후보 키 통합
    if isinstance(obj_or_text, dict):
        for k in ("frames","frame","labels","llm_frames","tags","categories","predictions"):
            if k in obj_or_text:
                v = obj_or_text[k]
                if isinstance(v, str): return [v.strip()]
                if isinstance(v, list): return [str(x).strip() for x in v if x]
    # 2) 문자열일 때 json 파싱 시도
    if isinstance(obj_or_text, str):
        try:
            j = json.loads(obj_or_text)
            return _extract_frames(j)
        except Exception:
            # 3) fallback: "frames: [ ... ]" 패턴 긁기
            m = re.search(r'frames?\s*[:=]\s*\[(.*?)\]', obj_or_text, re.I|re.S)
            if m:
                try: return json.loads('['+m.group(1)+']')
                except Exception: pass
    return []

# LLM 응답(content)을 받는 직후:
#   frames = _extract_frames(content_or_json)
#   rec["frames"] = frames


def _req_ollama(prompt:str)->str:
    url="http://localhost:11434/api/generate"
    data={"model":MODEL,"prompt":SYSTEM+"\n\n"+prompt,"stream":False,"options":{"temperature":0.1}}
    r=requests.post(url,json=data,timeout=120); r.raise_for_status()
    return r.json()["response"]

def _to_json(s:str)->Dict:
    # 코드블럭 제거 & JSON 추출
    s=s.strip()
    if "```" in s:
        s=s.split("```")[-2] if len(s.split("```"))>=2 else s
    s=s.strip()
    return oj.loads(s)

def call_llm(title:str,text:str)->Dict:
    prompt=USER_TMPL.format(label_list=LABEL_LIST_STR,title=title or "",text=(text or "")[:4000])
    raw = _req_ollama(prompt) if PROVIDER=="ollama" else _req_openai(prompt)
    try:
        return _to_json(raw)
    except Exception:
        # 실패시 안전 JSON
        return {"frames":[{"label":"없음(None)","conf":0.0}], "evidence":[]}

def read_jsonl(p): 
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s: yield oj.loads(s)

def main(fin, fout, limit=None):
    out=[]
    t0=time.time()
    for i,row in enumerate(read_jsonl(fin)):
        if limit and i>=limit: break
        title=row.get("title") or ""
        text = row.get("clean_text") or row.get("text") or row.get("body") or ""
        t1=time.time()
        ans=call_llm(title,text)
        dt=int((time.time()-t1)*1000)
        rec={"id":row.get("id"),"url":row.get("url"),"title":title,
             "frames":ans.get("frames",[]),"evidence":ans.get("evidence",[]),
             "provider":PROVIDER,"model":MODEL,"latency_ms":dt}
        out.append(rec)
    with open(fout,"w",encoding="utf-8") as w:
        for r in out: w.write(oj.dumps(r).decode()+"\n")
    print(f"LLM frames -> {fout}  n={len(out)}  total_ms={int((time.time()-t0)*1000)}")

if __name__=="__main__":
    fin=sys.argv[1]; fout=sys.argv[2]; limit=int(sys.argv[3]) if len(sys.argv)>3 else None
    main(fin,fout,limit)
