# nwwpkg/utils/loader.py
import json

def load_jsonl(path: str):
    """JSONL 파일을 안전하게 로드 (깨진 라인 무시)"""
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return data
