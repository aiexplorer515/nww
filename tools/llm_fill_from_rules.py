import json, sys, pathlib
root = pathlib.Path(sys.argv[1])  # e.g., data/b01
fin_rule = root/'frames.jsonl'
fin_llm  = root/'frames.llm.jsonl'
fout     = root/'frames.llm.filled.jsonl'

def get_frames(j):
    fs = j.get('frames') or j.get('frame') or j.get('labels') or j.get('llm_frames') or []
    if isinstance(fs, dict): fs=[fs]
    if isinstance(fs, str): fs=[fs]
    if not isinstance(fs, list): fs=[fs]
    out=[]
    for x in fs:
        if isinstance(x, str): 
            s=x.strip()
        elif isinstance(x, dict):
            s=str(x.get('label') or x.get('name') or x.get('frame') or x.get('tag') or '')
        else:
            s=str(x or '')
        s=s.strip()
        if s and s not in out: out.append(s)
    return out

# read rule frames by id
R={}
with open(fin_rule, encoding='utf-8') as f:
    for L in f:
        j=json.loads(L); R[j.get('id')]=get_frames(j)

n_fill=0; n_total=0
with open(fin_llm, encoding='utf-8') as f, open(fout,'w',encoding='utf-8') as w:
    for L in f:
        j=json.loads(L); n_total+=1
        llm = get_frames(j)
        if not llm and j.get('id') in R:
            j['frames'] = R[j['id']]
            j['llm_source'] = 'fallback_rule_copy'
            n_fill += 1
        else:
            j['frames'] = llm
        w.write(json.dumps(j, ensure_ascii=False)+'\n')

print(f'FILLED {n_fill}/{n_total} -> {fout}')
