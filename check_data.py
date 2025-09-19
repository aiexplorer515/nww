#!/usr/bin/env python3
"""
Check data files in bundles
"""
import json
from pathlib import Path

def count_lines(file_path):
    """Count lines in a file"""
    try:
        return sum(1 for _ in file_path.open(encoding="utf-8"))
    except:
        return -1

def main():
    # Set up paths - you can modify these variables
    ROOT = Path("data/bundles")
    B = "b01"  # or whatever bundle you want to check
    
    root = ROOT / B
    pf = root / "frames.jsonl"
    ps = root / "scores.jsonl"
    
    print(f"Checking bundle: {B}")
    print(f"frames: {count_lines(pf)} | scores: {count_lines(ps)}")
    
    if pf.exists() and count_lines(pf) > 0:
        try:
            with pf.open(encoding="utf-8") as f:
                first_line = f.readline()
                first_frame = json.loads(first_line)
                print("frames first:", first_frame)
        except Exception as e:
            print(f"Error reading frames: {e}")

if __name__ == "__main__":
    main()
