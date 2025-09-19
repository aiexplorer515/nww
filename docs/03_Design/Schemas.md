# JSONL Schemas (NWW v1.2+)

## articles.jsonl
- id, ts, title, text, domain, region, source, byline, url, lang

## articles.norm.jsonl
- id, ts, lang_final, hash, norm_text, segments[]

## kyw_sum.jsonl
- id, kw[], summary, actors[], frames[], span_evidence[]

## gated.jsonl
- id, hits[{indicator,val,conf,span}], z, logit, rep_adj

## scores.jsonl
- IS: {id, stage:"IS", score, detail{}, rep_adj}
- DBN: {id, stage:"DBN", score}
- LLM: {id, stage:"LLM", score, rationale, evidence_ids[], contra}
- FUSE: {id, stage:"FUSE", fused, p_calib, ci:[low,high], ema, state}

## block_hits.jsonl
- id, block_id, slot_coverage{}, delta, block_ver, rule_hash

## scenarios.jsonl
- scenario_id, title, claims[{text,evidence_id}], score{credo,fact,coverage}, lang

## alerts.jsonl
- id, alert, intl_flag, why[], ts

## ledger/*.jsonl
- step, hash_in, hash_out, params{}, seed, ver, prev_hash
