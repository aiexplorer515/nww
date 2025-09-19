#!/usr/bin/env python3
"""
Fix the regex pattern in normalize.py
"""

def fix_normalize():
    with open('nwwpkg/preprocess/normalize.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix line 223 (index 222) - the problematic regex pattern
    lines[222] = '        text = re.sub(r"[\'\'\']", "\'", text)\n'
    
    with open('nwwpkg/preprocess/normalize.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Fixed the regex pattern in normalize.py")

if __name__ == "__main__":
    fix_normalize()



