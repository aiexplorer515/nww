# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_scenario_with_llm(block: dict, top_events: list[str], model_name: str = "gpt-4o-mini") -> tuple[str, str]:
    """LLM 기반 시나리오 제목과 요약 생성"""
    prompt = f"""
당신은 국제 분쟁/정치 위기 분석가입니다.
다음 사건 블록을 기반으로 시나리오를 작성하세요.

- 기간: {block.get('start','')} ~ {block.get('end','')}
- 규모: {block.get('count',0)}건
- 키워드: {', '.join(block.get('top_keywords', []))}
- 대표 사건: {block.get('rep_title','')}
- 기사 요약: {' '.join(top_events)}

출력 형식:
1. 시나리오 제목
2. 시나리오 요약 (3~5문장, 맥락과 위험도 중심)
"""
    try:
        resp = client.chat.completions.create(
            model=model_name,   # ← 선택된 모델 반영
            messages=[
                {"role": "system", "content": "너는 위기 예측 전문가이다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
        )
        content = resp.choices[0].message.content.strip()
        lines = content.splitlines()
        title = lines[0].replace("1.", "").strip() if lines else "[Scenario]"
        summary = "\n".join(lines[1:]).replace("2.", "").strip() if len(lines) > 1 else ""
        return title, summary
    except Exception as e:
        return "[Scenario: Error]", f"LLM 생성 실패: {e}"
