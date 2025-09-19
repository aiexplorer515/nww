# tools/generate_rule_frames.py
# -*- coding: utf-8 -*-
import json
import re
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

def generate_rule_frames_for_text(title, text):
    """규칙 기반 프레임 분류"""
    frames = []
    evidence = []
    
    # 더 정교한 키워드 매칭 규칙
    rules = {
        "군사동원": {
            "keywords": ["troops", "mobilized", "military", "군사", "동원", "군대", "병력", "배치"],
            "patterns": [r"군사.*동원", r"troops.*mobilized", r"military.*deployment"],
            "weight": 1.0
        },
        "외교협상": {
            "keywords": ["negotiation", "talks", "diplomatic", "협상", "회담", "외교", "대화"],
            "patterns": [r"협상.*결렬", r"diplomatic.*talks", r"외교.*회담"],
            "weight": 1.0
        },
        "제재/보복": {
            "keywords": ["sanctions", "retaliation", "제재", "보복", "제재", "경제제재"],
            "patterns": [r"제재.*검토", r"economic.*sanctions", r"보복.*조치"],
            "weight": 1.0
        },
        "경제위기": {
            "keywords": ["economic", "crisis", "경제", "위기", "금융", "시장"],
            "patterns": [r"경제.*위기", r"financial.*crisis", r"시장.*불안"],
            "weight": 0.8
        },
        "사회불안": {
            "keywords": ["unrest", "protest", "사회", "불안", "시위", "폭동"],
            "patterns": [r"사회.*불안", r"civil.*unrest", r"시위.*발생"],
            "weight": 0.8
        },
        "재난/재해": {
            "keywords": ["disaster", "earthquake", "flood", "재난", "재해", "지진", "홍수"],
            "patterns": [r"자연.*재해", r"natural.*disaster", r"지진.*발생"],
            "weight": 0.9
        },
        "군사충돌": {
            "keywords": ["conflict", "clash", "battle", "충돌", "전투", "교전"],
            "patterns": [r"군사.*충돌", r"armed.*conflict", r"전투.*발생"],
            "weight": 1.0
        },
        "국경충돌": {
            "keywords": ["border", "skirmish", "국경", "경계", "분쟁"],
            "patterns": [r"국경.*충돌", r"border.*skirmish", r"경계.*분쟁"],
            "weight": 0.9
        }
    }
    
    text_lower = (title + " " + text).lower()
    text_combined = title + " " + text
    
    for frame_name, frame_en in LABELS[:-1]:  # "없음" 제외
        if frame_name in rules:
            rule = rules[frame_name]
            confidence = 0.0
            found_evidence = []
            
            # 키워드 매칭
            for keyword in rule["keywords"]:
                if keyword in text_lower:
                    confidence += 0.3
                    found_evidence.append(f"'{keyword}' 키워드")
            
            # 패턴 매칭
            for pattern in rule["patterns"]:
                if re.search(pattern, text_combined, re.IGNORECASE):
                    confidence += 0.4
                    found_evidence.append(f"'{pattern}' 패턴")
            
            # 가중치 적용
            confidence *= rule["weight"]
            
            if confidence > 0.5:
                frames.append({
                    "label": f"{frame_name}({frame_en})",
                    "conf": min(round(confidence, 2), 1.0)
                })
                evidence.extend(found_evidence[:2])  # 최대 2개 증거
    
    # 프레임이 없으면 "없음"
    if not frames:
        frames.append({
            "label": "없음(None)",
            "conf": 0.0
        })
        evidence.append("관련 키워드/패턴 없음")
    
    return frames[:3], evidence[:3]  # 최대 3개

def main(input_file, output_file, limit=None):
    """JSONL 파일을 읽어서 규칙 기반 프레임 분류 결과 생성"""
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
                
                frames, evidence = generate_rule_frames_for_text(title, text)
                
                result = {
                    "id": data.get("id", f"doc_{i}"),
                    "url": data.get("url", ""),
                    "title": title,
                    "frames": frames,
                    "evidence": evidence,
                    "provider": "rule_based",
                    "model": "keyword_pattern_matcher",
                    "latency_ms": 5
                }
                results.append(result)
                
            except json.JSONDecodeError:
                print(f"JSON 파싱 오류: {line}")
                continue
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"규칙 기반 프레임 분류 완료: {output_file}")
    print(f"처리된 문서 수: {len(results)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("사용법: python generate_rule_frames.py <입력파일> <출력파일> [제한수]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    main(input_file, output_file, limit)
