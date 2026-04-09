import re
from pathlib import Path

def batch_extract(input_dir: str, output_file: str):
    all_sonnets = []
    for path in sorted(Path(input_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        L = r"[^\n]+"
        B = r"\n[ \t]*\n"
        pattern = r"([^\n]+\n){4}\n[ \t]?([^\n]+\n){4}\n[ \t]?([^\n]+\n){3}\n[ \t]?([^\n]+\n){3}\n[ \t]?" 
        matches = list(re.finditer(pattern, text))
        matches = [
            m for m in matches
            if all(len(l) <= 80 for l in m.group().splitlines() if l.strip())
        ]
        all_sonnets.extend(m.group().strip() for m in matches)
        print(f"  {path.name}: {len(matches)} sonnets")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n\n".join(all_sonnets))
    print(f"\nTotal: {len(all_sonnets)} sonnets → {output_file}")

batch_extract("raw-texts-utf8/", "extracted_sonnets.txt")
