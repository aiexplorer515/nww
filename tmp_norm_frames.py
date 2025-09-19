import json, csv, os
from pathlib import Path

root = Path(os.environ.get('ROOT','data/b01'))
fin = str(root / 'frames.llm.jsonl')
map_csv = r'configs/frame_map.auto.csv'
out = str(root / 'frames.llm.norm.jsonl')

m={}
with open(map_csv, encoding='utf-8') as f:
    r=csv.DictReader(f)
    for row in r:
        m[row['from'].strip()] = row['to'].strip()

def norm(fr_list):
    S=[]
    for x in fr_list:
        x=str(x or '').strip()
        x=m.get(x,x)
        if x and x!='other':
            S.append(x)
    # 중복 제거(안정 순서)
    out=[]
    for x in S:
        if x not in out: out.append(x)
    return out

with open(out,'w',encoding='utf-8') as w, open(fin, encoding='utf-8') as f:
    for line in f:
        j=json.loads(line)
        fs = j.get('frames') or j.get('frame') or j.get('llm_frames') or j.get('labels') or []
        if isinstance(fs, str): fs=[fs]
        j['frames']=norm(fs)
        w.write(json.dumps(j, ensure_ascii=False)+'\n')

print('WROTE', out)
