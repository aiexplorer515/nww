import json, csv, sys
from pathlib import Path

def load_map(p):
    m={}
    with open(p, encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            m[(row['from'] or '').strip()] = (row['to'] or '').strip()
    return m

def norm_list(fr_list, m):
    out=[]
    for x in fr_list:
        x=str(x or '').strip()
        x=m.get(x, x)
        if x and x!='other':
            out.append(x)
    # 중복 제거(안정 순서)
    uniq=[]
    for x in out:
        if x not in uniq: uniq.append(x)
    return uniq

def extract_frames(j):
    fs = j.get('frames') or j.get('frame') or j.get('llm_frames') or j.get('labels') or []
    if isinstance(fs, str): fs=[fs]
    return fs

def main(fin, cmap, fout):
    Path(fout).parent.mkdir(parents=True, exist_ok=True)
    m = load_map(cmap)
    with open(fin, encoding='utf-8') as f, open(fout, 'w', encoding='utf-8') as w:
        for line in f:
            j = json.loads(line)
            fs = extract_frames(j)
            j['frames'] = norm_list(fs, m)
            w.write(json.dumps(j, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    fin, cmap, fout = sys.argv[1], sys.argv[2], sys.argv[3]
    main(fin, cmap, fout)

