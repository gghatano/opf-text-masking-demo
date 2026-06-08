"""Generate a small SYNTHETIC labeled eval set (pilot) for the fair comparison.

Per #26 (Claude Code / rule-based synthesis; no real data, no collision check)
and #29 annotation rules (honorifics excluded; full address=ADDRESS, bare
place=REGION). Char offsets are computed by searching the surface string, so
they are exact by construction (no hand-counting).

Output: data/eval/pilot.jsonl  (schema: text + label[{category,start,end}] + info)
This is a tiny pilot to exercise the pipeline end-to-end, NOT the 300-doc set (#5).
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "pilot.jsonl"

# (domain, text, [(label, surface), ...])  surfaces must appear verbatim in text.
DATA = [
    ("医療",
     "佐藤花子（72歳）は2025年1月3日に北海道札幌市中央区北1条西2丁目の北海道大学病院を受診した。担当は内科医の鈴木一郎。患者番号は P-002931。",
     [("PERSON", "佐藤花子"), ("AGE", "72歳"), ("DATE", "2025年1月3日"),
      ("ADDRESS", "北海道札幌市中央区北1条西2丁目"), ("ORGANIZATION", "北海道大学病院"),
      ("OCCUPATION", "内科医"), ("PERSON", "鈴木一郎"), ("ID", "P-002931")]),
    ("医療",
     "山田太郎さんの被保険者番号は 12345678、連絡先は 090-1234-5678。",
     [("PERSON", "山田太郎"), ("ID", "12345678"), ("PHONE", "090-1234-5678")]),
    ("医療",
     "退院サマリ：田中美咲、85歳、世田谷区在住。次回受診は令和7年2月10日。",
     [("PERSON", "田中美咲"), ("AGE", "85歳"), ("REGION", "世田谷区"),
      ("DATE", "令和7年2月10日")]),
    ("医療",
     "紹介状の宛先は東京都港区赤坂1-2-3 港区医療センターの高橋健。",
     [("ADDRESS", "東京都港区赤坂1-2-3"), ("ORGANIZATION", "港区医療センター"),
      ("PERSON", "高橋健")]),
    ("医療",
     "看護記録：本日、佐々木様より自宅電話 03-5555-1234 へ連絡希望。メールは sasaki@example.jp。",
     [("PERSON", "佐々木"), ("PHONE", "03-5555-1234"), ("EMAIL", "sasaki@example.jp")]),
    ("自治体",
     "相談者：中村剛、44歳、職業は教員。住所は大阪府大阪市北区梅田3-1-1。受付番号 R2025-0098。",
     [("PERSON", "中村剛"), ("AGE", "44歳"), ("OCCUPATION", "教員"),
      ("ADDRESS", "大阪府大阪市北区梅田3-1-1"), ("ID", "R2025-0098")]),
    ("自治体",
     "生活福祉相談：伊藤さんの世帯。連絡先 080-9876-5432。担当ケースワーカーは渡辺。",
     [("PERSON", "伊藤"), ("PHONE", "080-9876-5432"), ("OCCUPATION", "ケースワーカー"),
      ("PERSON", "渡辺")]),
    ("自治体",
     "問い合わせはメール kobayashi@city.example.lg.jp へ。横浜市在住の小林より。",
     [("EMAIL", "kobayashi@city.example.lg.jp"), ("REGION", "横浜市"), ("PERSON", "小林")]),
    ("自治体",
     "千葉市役所 福祉課の山口あかねが対応。会員番号 M-77881。",
     [("ORGANIZATION", "千葉市役所"), ("PERSON", "山口あかね"), ("ID", "M-77881")]),
    ("自治体",
     "面談日：2024/12/05、対象者は名古屋市中区の加藤、62歳。",
     [("DATE", "2024/12/05"), ("REGION", "名古屋市中区"), ("PERSON", "加藤"), ("AGE", "62歳")]),
    ("その他",
     "コールセンター記録：松本由美様、電話 0120-111-222、注文番号 ORD-55012。",
     [("PERSON", "松本由美"), ("PHONE", "0120-111-222"), ("ID", "ORD-55012")]),
    ("その他",
     "アンケート：私は福岡県福岡市早良区に住む会社員の井上です。",
     [("REGION", "福岡県福岡市早良区"), ("OCCUPATION", "会社員"), ("PERSON", "井上")]),
    ("その他",
     "担当者の木村、勤務先は富士通株式会社。内線は 03-2222-3333。",
     [("PERSON", "木村"), ("ORGANIZATION", "富士通株式会社"), ("PHONE", "03-2222-3333")]),
    ("その他",
     "学生の斉藤（20歳）、所属は東京大学。メール saito@u-tokyo.example.ac.jp。",
     [("OCCUPATION", "学生"), ("PERSON", "斉藤"), ("AGE", "20歳"),
      ("ORGANIZATION", "東京大学"), ("EMAIL", "saito@u-tokyo.example.ac.jp")]),
    ("その他",
     "2026年6月8日、契約者番号 C-99001 の本人確認を実施。氏名は渡邊大輔。",
     [("DATE", "2026年6月8日"), ("ID", "C-99001"), ("PERSON", "渡邊大輔")]),
]


def spans_for(text: str, items: list[tuple[str, str]]) -> list[dict]:
    spans = []
    cursor: dict[str, int] = {}
    for label, surface in items:
        start = text.find(surface, cursor.get(surface, 0))
        if start < 0:
            raise ValueError(f"surface {surface!r} not found in {text!r}")
        end = start + len(surface)
        cursor[surface] = end  # handle repeats
        assert text[start:end] == surface
        spans.append({"category": label, "start": start, "end": end})
    return spans


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, (domain, text, items) in enumerate(DATA):
            rec = {"text": text, "label": spans_for(text, items),
                   "info": {"id": f"pilot_{i:03d}", "domain": domain, "split": "eval"}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} records to {OUT}")


if __name__ == "__main__":
    main()
