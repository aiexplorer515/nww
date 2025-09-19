#!/usr/bin/env python3
"""
Extract keywords and phrases from clean data
"""
import json
from pathlib import Path
from nwwpkg.analyze.keywords import build_keywords, mine_phrases

def main():
    # Set up paths
    input_file = "data/b01/clean.jsonl"
    keywords_output = "data/b01/keywords.jsonl"
    phrases_output = "data/b01/phrases.jsonl"
    
    print("🔍 Extracting keywords and phrases...")
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    try:
        # Extract keywords
        print("📝 Building keywords...")
        rows, cnt = build_keywords(input_file, keywords_output, topn=100)
        print(f"✅ Keywords extracted: {cnt} rows")
        
        # Extract phrases
        print("🔤 Mining phrases...")
        phr = mine_phrases(input_file, topk=30)
        print(f"✅ Phrases extracted: {len(phr)} phrases")
        
        # Save phrases
        print("💾 Saving phrases...")
        phrases_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in phr)
        Path(phrases_output).write_text(phrases_text, encoding="utf-8")
        
        print("🎉 Done keywords/phrases extraction!")
        print(f"📁 Keywords saved to: {keywords_output}")
        print(f"📁 Phrases saved to: {phrases_output}")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
