import json, csv, os, difflib
from pathlib import Path

root = Path(os.environ.get('ROOT','data/b01'))
fin = str(root / 'frames.llm.jsonl')
vocab_path = r'config/frames.master.json'
out_csv = r'configs/frame_map.auto.csv'
Path('configs').mkdir(exist_ok=True)

with open(vocab_path, encoding='utf-8') as f:
    vocab = [x.strip() for x in json.load(f)]

seen = {}
with open(fin, encoding='utf-8') as f:
    for line in f:
        j = json.loads(line)
        # LLM 출력 필드가 무엇이든 대응
        fs = j.get('frames') or j.get('frame') or j.get('llm_frames') or j.get('labels') or []
        if isinstance(fs, str): fs=[fs]
        for fr in fs:
            fr = str(fr).strip()
            if fr: seen[fr] = seen.get(fr,0)+1

rows=[('from','to','count','match_score')]
for src,cnt in seen.items():
    match = difflib.get_close_matches(src, vocab, n=1, cutoff=0.6)
    if match:
        score = difflib.SequenceMatcher(None, src, match[0]).ratio()
        rows.append((src, match[0], cnt, f'{score:.2f}'))
    else:
        rows.append((src, 'other', cnt, '0.00'))

with open(out_csv, 'w', newline='', encoding='utf-8') as w:
    csv.writer(w).writerows(rows)
print('WROTE', out_csv, 'rows=', len(rows))
