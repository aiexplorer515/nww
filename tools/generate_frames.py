# tools/generate_frames.py
# -*- coding: utf-8 -*-
import json
import random
from pathlib import Path

LABELS = [
    ("군사충돌", "Armed_Clash"),
    ("군사동원", "Military_Mobilization"),
    ("국경충돌", "Border_Skirmish"),
    ("핵/미사일", "WMD_Missile"),
    ("외교협상", "Diplomatic_Talks"),
    ("제재/보복", "Sanctions_Retaliation"),
    ("경제위기", "Economic_Crisis"),
    ("사회불안", "Domestic_Unrest"),
    ("사이버공격", "Cyber_Attack"),
    ("재난/재해", "Disaster"),
    ("없음", "None")
]

def generate_frames_for_text(title, text):
    """간단한 키워드 기반 프레임 분류"""
    frames = []
    evidence = []
    
    # 키워드 매칭
    keywords = {
        "군사동원": ["troops", "mobilized", "military", "군사", "동원", "군대"],
        "외교협상": ["negotiation", "talks", "diplomatic", "협상", "회담", "외교"],
        "제재/보복": ["sanctions", "retaliation", "제재", "보복", "제재"],
        "경제위기": ["economic", "crisis", "경제", "위기", "금융"],
        "사회불안": ["unrest", "protest", "사회", "불안", "시위"],
        "재난/재해": ["disaster", "earthquake", "flood", "재난", "재해", "지진", "홍수"]
    }
    
    text_lower = (title + " " + text).lower()
    
    for frame_name, frame_en in LABELS[:-1]:  # "없음" 제외
        for keyword in keywords.get(frame_name, []):
            if keyword in text_lower:
                confidence = random.uniform(0.6, 0.9)
                frames.append({
                    "label": f"{frame_name}({frame_en})",
                    "conf": round(confidence, 2)
                })
                evidence.append(f"'{keyword}' 키워드 발견")
                break
    
    # 프레임이 없으면 "없음"
    if not frames:
        frames.append({
            "label": "없음(None)",
            "conf": 0.0
        })
        evidence.append("관련 키워드 없음")
    
    return frames[:3], evidence[:3]  # 최대 3개

def main(input_file, output_file, limit=None):
    """JSONL 파일을 읽어서 프레임 분류 결과 생성"""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"입력 파일이 존재하지 않습니다: {input_file}")
        return
    
    results = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                title = data.get("title", "")
                text = data.get("clean_text", "") or data.get("text", "")
                
                frames, evidence = generate_frames_for_text(title, text)
                
                result = {
                    "id": data.get("id", f"doc_{i}"),
                    "url": data.get("url", ""),
                    "title": title,
                    "frames": frames,
                    "evidence": evidence,
                    "provider": "dummy",
                    "model": "keyword_matcher",
                    "latency_ms": random.randint(10, 50)
                }
                results.append(result)
                
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {line}")
                continue
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"프레임 분류 완료: {output_file}")
    print(f"처리된 문서 수: {len(results)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("사용법: python generate_frames.py <입력파일> <출력파일> [제한수]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    main(input_file, output_file, limit)
