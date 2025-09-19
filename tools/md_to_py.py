import os
import re
import argparse

def extract_code(md_text: str) -> str:
    code_blocks = re.findall(r"```python(.*?)```", md_text, re.S)
    return "\n\n".join([block.strip() for block in code_blocks])

def md_to_py(md_path: str, py_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    code = extract_code(md_text)
    if not code:
        print(f"[WARN] No code found in {md_path}")
        return
    os.makedirs(os.path.dirname(py_path), exist_ok=True)
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated from Markdown spec\n\n")
        f.write(code)
    print(f"[OK] Generated {py_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0].replace("_", "/")
                py_path = os.path.join(args.output, module_name + ".py")
                md_to_py(md_path, py_path)
