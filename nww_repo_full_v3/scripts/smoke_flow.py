
import json, os
os.makedirs("reports", exist_ok=True)
demo = {
  "doc":"tests/golden/cleandoc.demo.json","event":"tests/golden/event.demo.json",
  "evidence":"tests/golden/evidence.demo.json","framescore":"tests/golden/framescore.demo.json",
  "fpd":"tests/golden/fpd.golden.json","esd":"tests/golden/esd.golden.json",
  "ipd":"tests/golden/ipd.golden.json","final":"tests/golden/final_report.demo.json",
  "alert":"tests/golden/alert.demo.json"
}
result = {"timestamp":"2025-08-10T22:25:11Z","trace":[]}
for k,p in demo.items():
    ok = os.path.exists(p)
    result["trace"].append({"stage":k,"ok":ok})
json.dump(result, open("reports/smoke_result.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
print("smoke_result: reports/smoke_result.json")
