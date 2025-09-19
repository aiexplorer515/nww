import json, sys, pathlib
def _coerce(fr, default_conf: float):
    if fr is None: fr=[]
    if isinstance(fr, (str, int, float, dict)): fr=[fr]
    out=[]
    for x in fr:
        if isinstance(x, dict):
            lab = x.get('label') or x.get('frame') or x.get('name') or x.get('tag') or x.get('id')
            try: conf = float(x.get('conf', x.get('score', default_conf)))
            except: conf = default_conf
            if lab: out.append({'label': str(lab).strip(), 'conf': conf})
        else:
            s = str(x).strip()
            if s: out.append({'label': s, 'conf': float(default_conf)})
    best={}
    for d in out:
        l=d['label']
        if l not in best or d['conf']>best[l]['conf']:
            best[l]=d
    return list(best.values())

def process(fin, fout, default_conf):
    n=0
    with open(fin, encoding='utf-8') as f, open(fout, 'w', encoding='utf-8') as w:
        for L in f:
            if not L.strip(): continue
            j=json.loads(L)
            src = j.get('frames') or j.get('frame') or j.get('labels') or j.get('llm_frames')
            j['frames'] = _coerce(src, default_conf)
            w.write(json.dumps(j, ensure_ascii=False)+'\n'); n+=1
    print(f'WROTE {fout} n={n}')
if __name__=='__main__':
    fin, fout, conf = sys.argv[1], sys.argv[2], float(sys.argv[3])
    process(fin, fout, conf)
