import os
import chardet


def normalize_to_utf8(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, "rb") as f:
            raw = f.read()

        detected = chardet.detect(raw)
        encoding = detected["encoding"] or "latin-1"
        confidence = detected["confidence"]

        print(f"{filename}: detected {encoding} (confidence: {confidence:.0%})")

        text = raw.decode(encoding, errors="replace")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)


normalize_to_utf8("./raw-texts/", "./raw-texts-utf8/")
