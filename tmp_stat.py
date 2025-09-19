import os, json, pathlib
root = pathlib.Path(os.environ['ROOT'])

def to_label(x):
    if isinstance(x, str): return x.strip()
    if isinstance(x, dict):
        for k in ('label','name','frame','tag','id'):
            if k in x and isinstance(x[k], (str,int,float)):
                return str(x[k]).strip()
        return ''  # dict이지만 라벨 필드 없으면 무시
    return str(x).strip()

def extract_frames(j):
    fs = j.get('frames') or j.get('frame') or j.get('llm_frames') or j.get('labels') or []
    if isinstance(fs, dict): fs = [fs]
    if not isinstance(fs, list): fs = [fs]
    out = [to_label(x).lower() for x in fs if to_label(x)]
    return out

def stats(p):
    S=set(); nonempty=0; total=0
    with open(p, encoding='utf-8') as f:
        for L in f:
            j=json.loads(L); frs=extract_frames(j)
            if frs: nonempty+=1; S.update(frs)
            total+=1
    return nonempty, total, len(S)

ln, lt, lu = stats(root/'frames.llm.jsonl')
rn, rt, ru = stats(root/'frames.jsonl')
print("llm_nonempty =", ln, "/", lt, "uniq_llm =", lu)
print("rule_nonempty=", rn, "/", rt, "uniq_rule=", ru)
print("intersection =", "TBD after coercion")
