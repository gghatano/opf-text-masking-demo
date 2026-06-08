"""GiNZA qualitative smoke on the same Japanese sentences as the OPF smoke (#9).

Runs ja_ginza NER and dumps detected entities (text + GiNZA label) to
outputs/ginza_smoke.txt (file output avoids Windows console mojibake).
GiNZA uses the Extended Named Entity hierarchy, so its label set differs from
OPF's 8 categories — that label gap is itself a comparison axis (see #3).
"""
from __future__ import annotations

import io
from pathlib import Path

import spacy

OUT = Path(__file__).resolve().parent.parent / "outputs" / "ginza_smoke.txt"

SENTENCES = [
    "佐藤花子さん（72歳）、東京都千代田区在住。電話090-1234-5678、受付番号A-0012。",
    "患者の山田太郎は2025年1月3日に○○病院を受診。担当は鈴木医師。",
    "Contact: 田中一郎, tanaka@example.co.jp, 03-1234-5678",
]


def main() -> None:
    # split_mode override: newer spaCy/confection strictly reject the default
    # None for compound_splitter.split_mode (GiNZA 5.2 + spaCy 3.8). "C" = full mode.
    nlp = spacy.load("ja_ginza", config={"components.compound_splitter.split_mode": "C"})
    buf = io.StringIO()
    buf.write(f"# GiNZA smoke (model=ja_ginza, spaCy={spacy.__version__})\n\n")
    for i, text in enumerate(SENTENCES, 1):
        doc = nlp(text)
        buf.write(f"## 文{i}: {text}\n")
        if not doc.ents:
            buf.write("  (no entities)\n")
        for ent in doc.ents:
            buf.write(f"  - {ent.text!r:24}  label={ent.label_}\n")
        buf.write("\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
