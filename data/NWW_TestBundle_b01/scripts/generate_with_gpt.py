# -*- coding: utf-8 -*-
"""
OpenAI(GPT)로 테스트 기사/라벨을 **자동 확장**하는 스크립트
사용 전 .env 또는 환경변수에 OPENAI_API_KEY 설정 필요.
"""
import os, json, uuid, random, time
from datetime import datetime, timedelta, timezone

# pip install openai>=1.0.0
from openai import OpenAI

CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM = "You are a news synthesizer that outputs realistic but fictional OSINT-style articles with clear metadata."
PROMPT_TMPL = """Create {n} short news records in JSONL. Fields: id(str), url(str), title(str), published_at(ISO8601 UTC), source(domain), text(str >= 600 chars), lang(ko|en), country(2-letter), frames(list[str] from {frames}).
Topic mix: military, diplomacy, economy, politics, info_ops, energy, cyber, sanctions.
Diversity: vary sources and countries. 60% ko, 40% en. Dates within last 60 days.
Return JSONL only, no extra text.
"""

def gpt_generate(n=50, frames=None):
    frames = frames or ["military","diplomacy","economy","politics","info_ops","energy","cyber","sanctions"]
    prompt = PROMPT_TMPL.format(n=n, frames=frames)
    rsp = CLIENT.chat.completions.create(
        model=os.getenv("MODEL_NAME","gpt-4o-mini"),
        temperature=0.7,
        messages=[{"role":"system","content":SYSTEM},
                  {"role":"user","content":prompt}]
    )
    text = rsp.choices[0].message.content.strip()
    # Save as JSONL
    ts = int(time.time())
    out = f"data/b01/clean.gpt_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + ("
" if not text.endswith("
") else ""))
    print("generated ->", out)

if __name__ == "__main__":
    os.makedirs("data/b01", exist_ok=True)
    gpt_generate(n=80)
