import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Callable, Sequence


VERSE_WINDOW_SIZE = 14
VERSE_MIN_QUALIFYING_LINES = 8
VERSE_MAX_LINE_LENGTH = 60
VERSE_MAX_UPPERCASE_RATIO = 0.6


def normalize_text(text:str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
    return normalized


def remove_gutenberg_boilerplate(text: str) -> str:
    start_pattern = re.compile(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
        re.IGNORECASE | re.DOTALL,
    )
    end_pattern = re.compile(
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*",
        re.IGNORECASE | re.DOTALL,
    )
    start = start_pattern.search(text)
    end = end_pattern.search(text)
    if start and end and end.start() > start.end():
        return text[start.end() : end.start()]
    if start:
        return text[start.end() :]
    if end:
        return text[: end.start()]
    return text


def remove_source_urls(text: str) -> str:
    return re.sub(r"\bSource.*?\.net\b", "", text, flags=re.IGNORECASE)


def remove_tables_of_contents(text: str) -> str:
    pattern = re.compile(
        r"(TABLE(.|\s)*?\n{2,})")
    match = pattern.search(text)
    if match:
        return text[: match.start()] + text[match.end() :]
    pattern_inline = re.compile(
        r"(\bTable(.|\n|\r)*?\.\.\.\s*\d{2,})"
    )
    match_inline = pattern_inline.search(text)
    if match_inline:
        return text[: match_inline.start()] + text[match_inline.end() :]
    return text


def remove_cover_sections(text: str) -> str:
    match = re.search(r"\bTous droits r[ée]servés\b.*?\n", text)
    if match:
        return text[match.end() :]
    return text


def _looks_like_verse_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > VERSE_MAX_LINE_LENGTH:
        return False
    if stripped.endswith(":"):
        return False
    alpha_chars = sum(1 for char in stripped if char.isalpha())
    if alpha_chars:
        uppercase_chars = sum(1 for char in stripped if char.isupper())
        if uppercase_chars / alpha_chars > VERSE_MAX_UPPERCASE_RATIO:
            return False
    else:
        return False
    return True


def _find_poem_body_index(text:str, start_offset:int = 0) -> int | None:
    if start_offset >= len(text):
        return None
    segment = text[start_offset:]
    lines = segment.splitlines(keepends=True)
    if not lines:
        return None
    line_offsets:list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    is_verse_line:list[bool] = [_looks_like_verse_line(line) for line in lines]
    
    l = 0
    while l < len(lines):
        if is_verse_line[l]:
            i = l
            while is_verse_line[l]:
                l += 1
            j = l
            if (j - i) >= VERSE_WINDOW_SIZE:
                return line_offsets[i]
        else:
            l += 1
    return None


def remove_prefaces(text: str) -> str:
    preface_pattern = re.compile(
        r"\b(PR[EÉ]FACE|AVERTISSEMENT|INTRODUCTION)\b", re.IGNORECASE
    )
    preface = preface_pattern.search(text)
    if preface:
        body_index = _find_poem_body_index(text, preface.end())
    else:
        body_index = _find_poem_body_index(text, 0)
    if body_index is not None:
        return text[body_index:]
    return text


def remove_footnotes(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[(.|\s)*?\]", "", text)
    text = re.sub(r"\n\s*\d+\.?\s+", "\n", text)
    return text


def collapse_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


Step = Callable[[str], str]


def _run_step(text:str, step:Step) -> str:
    updated = step(text)
    return updated

def build_pipeline() -> Sequence[Step]:
    return(
        normalize_text,
        remove_gutenberg_boilerplate,
        remove_source_urls,
        remove_tables_of_contents,
        remove_prefaces,
        remove_footnotes,
        collapse_whitespace,
    )

def clean_text(raw_text: str) -> str:
    text = raw_text
    for step in build_pipeline():
        text = _run_step(text, step)
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./raw-texts-utf8")
    parser.add_argument("--output-dir", default="./clean-texts")
    parser.add_argument("--full-text-path",default="./full-text.txt")
    return parser.parse_args()


def build_corpus(args) -> None:
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_texts = []

    for path in sorted(input_dir.glob("*.txt")):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_text(raw_text)
        output_path = output_dir / path.name
        output_path.write_text(cleaned_text, encoding="utf-8")
        full_texts.append(cleaned_text)

    full_text = "\n\n".join(full_texts)
    Path(args.full_text_path).write_text(full_text, encoding="utf-8")
    

def main() -> None:
    args =  parse_args()
    build_corpus(args)

if __name__ == "__main__":
    main()
