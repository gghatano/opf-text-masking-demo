"""Generate the 300-doc SYNTHETIC evaluation set with gold PII labels (#5).

Approach (#26): rule-based synthesis = fixed templates with named-entity pools.
No real data is used, so colliding with real individuals is not a concern.
Annotation rules (#29):
  - honorifics (様/さん/氏/医師 …) are NOT part of the span (本人名のみ),
  - full address (都道府県+市区町村+丁番地) = ADDRESS, bare place (○○市) = REGION,
  - third parties are also in scope (関係語 自体「長男」等 は対象外),
  - 1 span = 1 label, 病院名 = ORGANIZATION,
  - span match is char IoU>=0.5 downstream (#19) — gold offsets here are exact.

Offsets are computed by *accumulating* the text fragment-by-fragment (not by
str.find), so they are exact by construction and robust to repeated/overlapping
surfaces. Every span is asserted to satisfy text[start:end] == value.

Output: data/eval/eval_300.jsonl
  schema: {"text", "label":[{"category","start","end"}], "info":{id,domain,subtype,split}}

The 10-label scheme (spec.txt §5): PERSON ADDRESS PHONE EMAIL DATE ID
(direct) / AGE REGION OCCUPATION ORGANIZATION (quasi).

Note: light surface variety (和暦/略記/外国人氏名/部分匿名) is mixed in for
realism. A dedicated *hard-case* set and the dev/test partition are #20.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_300.jsonl"

# --------------------------------------------------------------------------
# Entity pools
# --------------------------------------------------------------------------
SURNAMES = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "清水", "斎藤",
    "前田", "藤田", "後藤", "岡田", "長谷川", "村上", "近藤", "石川", "中島", "西村",
]
GIVEN_F = ["花子", "美咲", "由美", "あかね", "陽子", "彩", "麻衣", "さくら", "綾乃", "千尋"]
GIVEN_M = ["太郎", "一郎", "健", "剛", "大輔", "翔", "拓也", "直樹", "亮", "誠"]
# 外国人氏名（難例・少数混在）/ 部分匿名（PIIスパンを持たない＝偽陽性テスト）
FOREIGN_NAMES = ["グエン・バン・アン", "リ・ウェイ", "ジョン・スミス", "キム・ミンス", "マリア・サントス"]
ANON_NAMES = ["A氏", "X様", "本人", "申請者", "相談者ご本人"]  # 続柄/匿名語: PIIスパンなし

OCCUPATIONS = [
    "医師", "内科医", "看護師", "教員", "会社員", "学生", "ケースワーカー",
    "介護士", "公務員", "自営業", "保健師", "薬剤師",
]
# (prefecture+city+ward base, sample chome-style) → ADDRESS は base+丁番地
ADDR_BASE = [
    "東京都千代田区", "東京都港区赤坂", "北海道札幌市中央区", "大阪府大阪市北区梅田",
    "神奈川県横浜市西区", "愛知県名古屋市中区", "福岡県福岡市早良区", "京都府京都市左京区",
    "宮城県仙台市青葉区", "広島県広島市中区", "兵庫県神戸市中央区", "埼玉県さいたま市浦和区",
]
# bare place（単独地名）→ REGION
REGIONS = [
    "世田谷区", "横浜市", "名古屋市中区", "福岡市早良区", "札幌市", "仙台市",
    "千葉市", "大阪市", "京都市", "神戸市", "川崎市", "さいたま市", "新宿区", "中央区",
]
HOSPITALS = ["北海道大学病院", "港区医療センター", "中央総合病院", "さくら内科クリニック",
             "市立第一病院", "東和会病院", "緑ヶ丘リハビリテーション病院"]
COMPANIES = ["富士通株式会社", "山田商事株式会社", "みらい電機株式会社", "日本テクノ株式会社",
             "あおぞら物産株式会社"]
SCHOOLS = ["東京大学", "大阪府立北高校", "市立第三中学校", "みどり幼稚園", "京都工業大学"]
GOV_OFFICES = ["千葉市役所 福祉課", "横浜市役所 市民課", "世田谷区役所 高齢福祉課",
               "大阪市北区役所 保険年金課"]
EMAIL_DOMAINS = ["example.jp", "city.example.lg.jp", "u-tokyo.example.ac.jp",
                 "example.com", "sample.co.jp"]
ID_KINDS = [  # (前置きラベル, 値生成キー)
    ("患者番号", "P"), ("被保険者番号", "num8"), ("受付番号", "R"), ("会員番号", "M"),
    ("注文番号", "ORD"), ("契約者番号", "C"), ("整理番号", "num8"),
]


def jp_phone(rng: random.Random) -> str:
    kind = rng.choice(["mobile", "fixed", "free"])
    if kind == "mobile":
        head = rng.choice(["090", "080", "070"])
        return f"{head}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
    if kind == "free":
        return f"0120-{rng.randint(100,999)}-{rng.randint(100,999)}"
    area = rng.choice(["03", "06", "052", "011", "045", "092"])
    return f"{area}-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"


def jp_date(rng: random.Random) -> str:
    y, m, d = rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28)
    style = rng.choice(["seireki", "seireki", "slash", "wareki", "abbr"])  # 西暦多め
    if style == "seireki":
        return f"{y}年{m}月{d}日"
    if style == "slash":
        return f"{y}/{m:02d}/{d:02d}"
    if style == "wareki":  # 令和（難例）
        return f"令和{y-2018}年{m}月{d}日"
    return f"R{y-2018}.{m}.{d}"  # 略記（難例）


def jp_email(rng: random.Random) -> str:
    local = rng.choice(["sato", "yamada", "kobayashi", "saito", "info", "k.tanaka", "user01"])
    return f"{local}@{rng.choice(EMAIL_DOMAINS)}"


def jp_id(rng: random.Random) -> tuple[str, str]:
    label, key = rng.choice(ID_KINDS)
    if key == "num8":
        val = str(rng.randint(10_000_000, 99_999_999))
    elif key == "ORD":
        val = f"ORD-{rng.randint(10000,99999)}"
    elif key == "R":
        val = f"R{rng.randint(2023,2026)}-{rng.randint(1,9999):04d}"
    else:  # P / M / C
        val = f"{key}-{rng.randint(10000,99999)}"
    return label, val


def address(rng: random.Random) -> str:
    base = rng.choice(ADDR_BASE)
    style = rng.choice(["dash", "dash", "kanji", "sapporo"])
    if style == "dash":
        return f"{base}{rng.randint(1,9)}-{rng.randint(1,30)}-{rng.randint(1,30)}"
    if style == "kanji":
        return f"{base}{rng.randint(1,9)}丁目{rng.randint(1,30)}番{rng.randint(1,30)}号"
    return f"{base}北{rng.randint(1,12)}条西{rng.randint(1,20)}丁目"


def person(rng: random.Random, *, allow_foreign: bool = True) -> str:
    r = rng.random()
    if allow_foreign and r < 0.06:
        return rng.choice(FOREIGN_NAMES)
    given = rng.choice(GIVEN_F + GIVEN_M)
    return rng.choice(SURNAMES) + given


def surname(rng: random.Random) -> str:
    return rng.choice(SURNAMES)


# --------------------------------------------------------------------------
# Templates: each returns (subtype, parts).  A part is either a literal str
# or a (category, value) tuple that becomes a labelled span.
# --------------------------------------------------------------------------
def t_medical_record(rng):
    name, age = person(rng), f"{rng.randint(1,99)}歳"
    label, idv = jp_id(rng)
    return "診療録", [
        "診療録：", ("PERSON", name), "（", ("AGE", age), "）は",
        ("DATE", jp_date(rng)), "に", ("ADDRESS", address(rng)), "の",
        ("ORGANIZATION", rng.choice(HOSPITALS)), "を受診。担当は",
        ("OCCUPATION", rng.choice(["内科医", "医師", "外科医"])), "の",
        ("PERSON", surname(rng) + rng.choice(GIVEN_M)), "。", label, "は ",
        ("ID", idv), "。",
    ]


def t_nursing(rng):
    return "看護記録", [
        "看護記録：本日、", ("PERSON", person(rng)), "様より自宅電話 ",
        ("PHONE", jp_phone(rng)), " へ連絡希望。メールは ",
        ("EMAIL", jp_email(rng)), "。担当看護師は",
        ("PERSON", surname(rng) + rng.choice(GIVEN_F)), "。",
    ]


def t_discharge(rng):
    return "退院サマリ", [
        "退院サマリ：", ("PERSON", person(rng)), "、", ("AGE", f"{rng.randint(60,99)}歳"),
        "、", ("REGION", rng.choice(REGIONS)), "在住。次回受診は",
        ("DATE", jp_date(rng)), "。主治医 ",
        ("PERSON", surname(rng) + rng.choice(GIVEN_M)), "。",
    ]


def t_referral(rng):
    return "紹介状", [
        "紹介状の宛先は", ("ADDRESS", address(rng)), " ",
        ("ORGANIZATION", rng.choice(HOSPITALS)), "の", ("PERSON", person(rng)),
        "先生。患者は", ("PERSON", person(rng)), "（", ("AGE", f"{rng.randint(1,99)}歳"),
        "）、長男（", ("PERSON", person(rng)), "）が付き添い。",
    ]


def t_city_consult(rng):
    label, idv = jp_id(rng)
    return "市民相談記録", [
        "相談者：", ("PERSON", person(rng)), "、", ("AGE", f"{rng.randint(18,90)}歳"),
        "、職業は", ("OCCUPATION", rng.choice(OCCUPATIONS)), "。住所は",
        ("ADDRESS", address(rng)), "。", label, " ", ("ID", idv), "。",
    ]


def t_welfare(rng):
    return "福祉相談記録", [
        "生活福祉相談：", ("PERSON", surname(rng)), "さんの世帯。連絡先 ",
        ("PHONE", jp_phone(rng)), "。担当", ("OCCUPATION", "ケースワーカー"), "は",
        ("PERSON", surname(rng)), "。内縁の夫（", ("PERSON", person(rng)), "）も同席。",
    ]


def t_inquiry(rng):
    return "問い合わせ記録", [
        ("ORGANIZATION", rng.choice(GOV_OFFICES)), "の",
        ("PERSON", surname(rng) + rng.choice(GIVEN_F)), "が対応。問い合わせはメール ",
        ("EMAIL", jp_email(rng)), " へ。", ("REGION", rng.choice(REGIONS)),
        "在住の", ("PERSON", surname(rng)), "より、", ("DATE", jp_date(rng)), "受付。",
    ]


def t_callcenter(rng):
    label, idv = jp_id(rng)
    return "コールセンター記録", [
        "コールセンター記録：", ("PERSON", person(rng)), "様、電話 ",
        ("PHONE", jp_phone(rng)), "、", label, " ", ("ID", idv),
        "。勤務先は", ("ORGANIZATION", rng.choice(COMPANIES)), "。",
    ]


def t_survey(rng):
    return "アンケート自由記述", [
        "アンケート：私は", ("REGION", rng.choice(REGIONS)), "に住む",
        ("OCCUPATION", rng.choice(["会社員", "学生", "自営業", "公務員"])), "の",
        ("PERSON", person(rng)), "です。所属は", ("ORGANIZATION", rng.choice(SCHOOLS + COMPANIES)),
        "。連絡は ", ("EMAIL", jp_email(rng)), " まで。",
    ]


def t_survey_anon(rng):
    # 部分匿名（難例）: 本人名は伏字 → PERSON スパンなし。年齢・地名のみPII。
    return "アンケート自由記述", [
        "アンケート：", ("REGION", rng.choice(REGIONS)), "在住、",
        ("AGE", f"{rng.randint(20,80)}歳"), "の", rng.choice(ANON_NAMES),
        "です。受診日は", ("DATE", jp_date(rng)), "でした。",
    ]


TEMPLATES = {
    "医療": [t_medical_record, t_nursing, t_discharge, t_referral],
    "自治体": [t_city_consult, t_welfare, t_inquiry],
    "その他": [t_callcenter, t_survey, t_survey, t_survey_anon],  # survey 多め, anon 少数
}


def build(parts) -> tuple[str, list[dict]]:
    text, labels = "", []
    for p in parts:
        if isinstance(p, tuple):
            cat, val = p
            start = len(text)
            text += val
            labels.append({"category": cat, "start": start, "end": len(text)})
        else:
            text += p
    return text, labels


def validate(text: str, labels: list[dict]) -> None:
    prev_end = -1
    for sp in sorted(labels, key=lambda s: s["start"]):
        assert 0 <= sp["start"] < sp["end"] <= len(text), f"bad span {sp} in {text!r}"
        assert text[sp["start"]:sp["end"]], "empty span"
        assert sp["start"] >= prev_end, f"overlapping spans in {text!r}"
        prev_end = sp["end"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=20260609)
    ap.add_argument("--n-per-domain", type=int, default=100)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records, label_counts, domain_counts = [], Counter(), Counter()
    seen_texts: set[str] = set()
    idx = 0
    for domain, builders in TEMPLATES.items():
        made = 0
        guard = 0
        while made < args.n_per_domain:
            guard += 1
            if guard > args.n_per_domain * 50:
                raise RuntimeError(f"could not reach {args.n_per_domain} unique docs for {domain}")
            subtype, parts = rng.choice(builders)(rng)
            text, labels = build(parts)
            if text in seen_texts:  # dedup (leakage/duplication guard, #20)
                continue
            seen_texts.add(text)
            validate(text, labels)
            records.append({
                "text": text,
                "label": labels,
                "info": {"id": f"eval_{idx:04d}", "domain": domain,
                         "subtype": subtype, "split": "eval"},
            })
            for sp in labels:
                label_counts[sp["category"]] += 1
            domain_counts[domain] += 1
            made += 1
            idx += 1

    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {args.out}")
    print("by domain :", dict(domain_counts))
    print("by label  :", dict(sorted(label_counts.items(), key=lambda kv: -kv[1])))
    print(f"total spans: {sum(label_counts.values())}")


if __name__ == "__main__":
    main()
