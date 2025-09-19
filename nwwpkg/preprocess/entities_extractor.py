# nwwpkg/preprocess/entities_extractor.py
import spacy
import json
from pathlib import Path

def get_nlp(lang: str = "en"):
    if lang == "en":
        model = "en_core_web_sm"
    elif lang == "ko":
        model = "ko_core_news_sm"  # 한국어 모델 (설치 필요)
    else:
        raise ValueError(f"지원하지 않는 언어: {lang}")

    try:
        return spacy.load(model)
    except OSError:
        from spacy.cli import download
        download(model)
        return spacy.load(model)

nlp = get_nlp("en")
# en_core_web_sm 모델 로드

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # 모델이 없을 경우 자동 다운로드
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_entities_from_text(text: str):
    """단일 텍스트에서 엔티티 추출"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    return entities


def run(text: str):
    """호환용: 단일 텍스트 입력"""
    return extract_entities_from_text(text)


def run_file(fin: str | Path, fout: str | Path):
    """
    JSONL 파일 입력받아 엔티티 추출 결과 저장
    :param fin: 정규화된 텍스트가 들어있는 JSONL (normalized.jsonl)
    :param fout: 엔티티 결과 저장할 JSONL (entities.jsonl)
    """
    fin, fout = Path(fin), Path(fout)
    fout.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    with fin.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            text = row.get("normalized") or row.get("text") or ""
            if not text:
                continue

            entities = extract_entities_from_text(text)
            out_rows.append({
                "source_id": row.get("id"),
                "text": text[:200],  # 미리보기용 일부
                "entities": entities
            })

    # 저장
    with fout.open("w", encoding="utf-8") as fw:
        for row in out_rows:
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_rows
