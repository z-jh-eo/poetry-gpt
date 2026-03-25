import os
import re


def clean_text(text: str) -> str:
    # Removal of project gutenberg headings / endings
    start_marker = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*\*\*\*\n"
    end_marker = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK"

    start_match = re.search(start_marker, text)
    end_match = re.search(end_marker, text)

    if start_match is not None or end_match is not None:
        text = text[start_match.end() : end_match.start()]

    # Removal of poesies.net endings
    url_match = re.search(r"\bSource.*?\.net\b", text)
    if url_match is not None:
        text = "\n".join([text[: url_match.start()], text[url_match.end() :]])

    # Removal of Table des matieres
    table_match = re.search(
        r"(\bTABLE(.|\s)*?(\n|\r){4,}|\bTable(.|\r|\n)*\s*\.+\s*\d{2,})", text
    )
    if table_match is not None:
        text = "\n".join([text[: table_match.start()], text[table_match.end() :]])

    # Removal of the book cover part
    cover_match = re.search(r"\bTous droits réservés\b", text)
    if cover_match is not None:
        text = text[cover_match.end() :]

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Removal of the prefaces

    return text


INPUT_DIR = "./raw-texts-utf8/"

CLEAN_DIR = "./clean-texts/"
os.makedirs(CLEAN_DIR, exist_ok=True)

all_texts = []

for filename in os.listdir(INPUT_DIR):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(CLEAN_DIR, filename)

    with open(input_path, encoding="utf-8") as f:
        text = clean_text(f.read())

    all_texts.append(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

full_text = "\n\n".join(all_texts)

with open("./full_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
