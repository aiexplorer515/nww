import json, difflib, sys
from pathlib import Path

root = Path(sys.argv[1])

def load_jsonl(p):
    rows=[]
    with open(p, encoding='utf-8') as f:
        for L in f:
            L=L.strip()
            if L: rows.append(json.loads(L))
    return rows

def save_jsonl(rows, p):
    with open(p, 'w', encoding='utf-8') as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False)+'\n')

def norm(s): 
    return (s or '').strip().lower()

def to_dict_frames(fr, default_conf):
    # 어떤 형태든 list[dict(label,conf)] 로 강제
    if fr is None: fr=[]
    if isinstance(fr, (str,int,float,dict)): fr=[fr]
    out=[]
    for x in fr:
        if isinstance(x, dict):
            lab = x.get('label') or x.get('frame') or x.get('name') or x.get('tag') or x.get('id')
            try: conf=float(x.get('conf', x.get('score', default_conf)))
            except: conf=default_conf
            if lab: out.append({'label': str(lab).strip(), 'conf': conf})
        else:
            s=str(x).strip()
            if s: out.append({'label': s, 'conf': float(default_conf)})
    # 라벨별 최고 conf만 남김
    best={}
    for d in out:
        l=d['label']
        if l not in best or d['conf']>best[l]['conf']:
            best[l]=d
    return list(best.values())

clean = load_jsonl(root/'clean.jsonl')
frames= load_jsonl(root/'frames.jsonl')
llm   = load_jsonl(root/'frames_llm.jsonl') if (root/'frames_llm.jsonl').exists() else []

# clean 인덱스
by_url   = {norm(r.get('url')): r.get('id') for r in clean if r.get('id')}
by_title = {norm(r.get('title')): r.get('id') for r in clean if r.get('id')}
title_list=list(by_title.keys())

def reindex(rows, default_conf):
    fixed=[]; hit=0; total=0
    for r in rows:
        total+=1
        cid = r.get('id')
        if cid not in by_title.values():
            cid = None  # 신뢰하지 않음
        # 1) url로 찾기
        if not cid:
            uid = by_url.get(norm(r.get('url')))
            if uid: cid=uid
        # 2) title 유사도로 찾기
        if not cid and r.get('title'):
            t = norm(r.get('title'))
            cand = difflib.get_close_matches(t, title_list, n=1, cutoff=0.85)
            if cand:
                cid = by_title[cand[0]]
        if cid:
            r['id']=cid; hit+=1
        # frames 형식 강제
        r['frames'] = to_dict_frames(r.get('frames') or r.get('frame') or r.get('labels') or r.get('llm_frames'), default_conf)
        fixed.append(r)
    return fixed, hit, total

frames2, h1, t1 = reindex(frames, 0.8)
save_jsonl(frames2, root/'frames.jsonl')

if llm:
    llm2, h2, t2 = reindex(llm, 0.6)
    save_jsonl(llm2, root/'frames_llm.jsonl')
    print(f'REINDEX frames: {h1}/{t1}, llm: {h2}/{t2}')
else:
    print(f'REINDEX frames: {h1}/{t1}, llm: 0/0')
