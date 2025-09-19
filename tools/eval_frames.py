# -*- coding: utf-8 -*-
# Robust eval_frames.py : rule vs llm vs merged 비교
import sys, json, pathlib, collections

def read_jsonl(p):
    rows=[]
    with open(p, encoding='utf-8') as f:
        for L in f:
            s=L.strip()
            if s: rows.append(json.loads(s))
    return rows

def _to_label(x):
    if isinstance(x, str): return x.strip()
    if isinstance(x, dict):
        for k in ('label','frame','name','tag','id'):
            v = x.get(k)
            if isinstance(v, (str,int,float)): return str(v).strip()
    return ""

def _as_list(fr):
    # 어떤 형태든 list 로
    if fr is None: return []
    if isinstance(fr, list): return fr
    return [fr]

def coerce(fr_list):
    # list[*] -> list[dict(label, conf)]
    out=[]
    for x in _as_list(fr_list):
        if isinstance(x, dict):
            lab = _to_label(x)
            try: conf = float(x.get('conf', x.get('score', 0.0)))
            except: conf = 0.0
            if lab: out.append({'label':lab,'conf':conf})
        else:
            lab=_to_label(x)
            if lab: out.append({'label':lab,'conf':0.0})
    return out

def top1(fr_list):
    fr=coerce(fr_list)
    if not fr: return (None, 0.0)
    best=max(fr, key=lambda d: float(d.get('conf',0.0)))
    return (best['label'], float(best.get('conf',0.0)))

def index_by_id(rows):
    return {r.get('id'): r for r in rows if r.get('id')}

def coverage(rows):
    n=len(rows); 
    if n==0: return 0.0
    hit=sum(1 for r in rows if top1(r.get('frames'))[0])
    return round(hit/max(1,n),3)

def main(f_rule, f_llm, f_merge):
    R = read_jsonl(f_rule)
    L = read_jsonl(f_llm)
    M = read_jsonl(f_merge)

    idR = index_by_id(R); idL = index_by_id(L); idM = index_by_id(M)
    ids = sorted(set(idR.keys()) & set(idL.keys()))

    agree_rl = 0; agree_rm = 0; agree_lm = 0; n_agree = 0
    conf_rule = []; conf_llm = []; conf_merge = []

    for i in ids:
        r = idR[i]; l = idL[i]; m = idM.get(i, {})
        lr, cr = top1(r.get('frames')); conf_rule.append(cr)
        ll, cl = top1(l.get('frames')); conf_llm.append(cl)
        lm, cm = top1(m.get('frames')) if m else (None,0.0); conf_merge.append(cm)

        if lr and ll:
            n_agree += 1
            if lr == ll: agree_rl += 1
        if lr and lm:
            if lr == lm: agree_rm += 1
        if ll and lm:
            if ll == lm: agree_lm += 1

    result = {
        'counts': {
            'rule': len(R), 'llm': len(L), 'merged': len(M),
            'ids_intersection_rl': len(ids), 'agree_pairs_count': n_agree
        },
        'coverage': {
            'rule': coverage(R), 'llm': coverage(L), 'merged': coverage(M),
        },
        'avg_conf': {
            'rule': round(sum(conf_rule)/max(1,len(conf_rule)),3),
            'llm':  round(sum(conf_llm)/max(1,len(conf_llm)),3),
            'merged':round(sum(conf_merge)/max(1,len(conf_merge)),3),
        },
        'agreement': {
            'rule_vs_llm': round(agree_rl/max(1,n_agree),3),
            'rule_vs_merged': round(agree_rm/max(1,len(ids)),3),
            'llm_vs_merged':  round(agree_lm/max(1,len(ids)),3),
        }
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('USAGE: eval_frames.py <frames.rule.jsonl> <frames_llm.jsonl> <frames.jsonl>', file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
