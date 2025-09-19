import os, json, csv, pathlib
root = pathlib.Path(os.environ['ROOT'])
fin  = root/'frames.llm.jsonl'
fmap = pathlib.Path('configs/frame_map.auto.csv')  # 있으면 정규화까지 같이
out  = root/'frames.llm.norm.jsonl'

def to_label(x):
    if isinstance(x,str): return x.strip()
    if isinstance(x,dict):
        for k in ('label','name','frame','tag','id'):
            if k in x and isinstance(x[k], (str,int,float)):
                return str(x[k]).strip()
    return ''

# optional mapping
MAP={}
if fmap.exists():
    with open(fmap, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            MAP[(r['from'] or '').strip().lower()] = (r['to'] or '').strip()

def map_std(s):
    return MAP.get(s.lower(), s)

with open(out,'w',encoding='utf-8') as w, open(fin,encoding='utf-8') as f:
    for L in f:
        j=json.loads(L)
        fs = j.get('frames') or j.get('frame') or j.get('llm_frames') or j.get('labels') or []
        if isinstance(fs, dict): fs=[fs]
        if not isinstance(fs, list): fs=[fs]
        lab = []
        for x in fs:
            s = to_label(x)
            if s: 
                s = map_std(s)
                if s and s!='other' and s not in lab:
                    lab.append(s)
        j['frames'] = lab
        w.write(json.dumps(j, ensure_ascii=False)+'\n')
print("WROTE", out)
