"""Markdown レポート → 自己完結 HTML(GitHub Pages 公開用)ビルダー。

特徴:
- figures/*.png を base64 で埋め込み、HTML 単体で配布・閲覧できる
- 目次(TOC)サイドバー + ヒーロー + ページ切替タブ
- ```mermaid ブロックを描画(抽出→div 注入→mermaid.js。codehilite に壊されない)
- ページ間リンク(.md)→ .html、その他のリポジトリ相対リンク → GitHub blob URL に書換
- 出典タグ [n](href="#refN")を上付きチップに整形(CSS の属性セレクタ)
- 区分バッジ(📘/🔎/📑)で始まる blockquote を色分け

使い方:
  1. 下の CONFIG を自分のプロジェクトに合わせて編集
  2. `pip install markdown pygments`
  3. `python scripts/build_html.py`  → htmls/ に出力

依存は markdown と pygments のみ(重い依存は不要なので CI でも高速)。
"""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import markdown

# ============================ ここを編集 ============================
ROOT = Path(__file__).resolve().parent.parent          # リポジトリ直下
OUTDIR = ROOT / "htmls"                                  # 出力先(Pages 公開元)

CONFIG = {
    "repo_url": "https://github.com/gghatano/opf-text-masking-demo",
    "upstream_url": "https://github.com/openai/privacy-filter",
    "hero_title": "OPF 日本語適用性 実証実験 — 評価レポート",
    "pages": [
        {"md": "REPORT.md", "out": "index.html", "key": "report", "nav": "📄 サマリ",
         "subtitle": "OpenAI Privacy Filter の日本語PII検出・段階的な数値検証（要約）"},
        {"md": "methods/opf.md", "out": "method-opf.html", "key": "opf", "nav": "🅾 OPF",
         "subtitle": "手法詳細: OpenAI Privacy Filter"},
        {"md": "methods/ginza.md", "out": "method-ginza.html", "key": "ginza", "nav": "🌿 GiNZA",
         "subtitle": "手法詳細: GiNZA (ja_ginza)"},
    ],
}
# ===================================================================


def embed_images(md_text: str) -> str:
    def repl(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2)
        img = ROOT / path
        if not img.exists():
            return m.group(0)
        b64 = base64.b64encode(img.read_bytes()).decode()
        return f"![{alt}](data:image/png;base64,{b64})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", repl, md_text)


def extract_mermaid(md_text: str):
    blocks: list[str] = []

    def repl(m: re.Match) -> str:
        blocks.append(m.group(1).strip())
        return f"\n\nxMERMAIDBLOCKx{len(blocks) - 1}x\n\n"

    return re.sub(r"```mermaid\s*\n(.*?)```", repl, md_text, flags=re.DOTALL), blocks


def inject_mermaid(html: str, blocks: list[str]) -> str:
    for i, src in enumerate(blocks):
        html = html.replace(f"<p>xMERMAIDBLOCKx{i}x</p>", f'<div class="mermaid">\n{src}\n</div>')
    return html


def rewrite_links(html: str, pages: list[dict], repo_url: str) -> str:
    # ページ間 .md リンク → 同一フォルダ .html
    md_to_html = {p["md"]: p["out"] for p in pages}
    for md, out in md_to_html.items():
        html = html.replace(f'href="{md}"', f'href="{out}"')
    site_files = {p["out"] for p in pages}

    def repl(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:", "data:")) or href in site_files:
            return m.group(0)
        return f'href="{repo_url}/blob/main/{href}"'

    return re.sub(r'href="([^"]+)"', repl, html)


def style_badges(html: str) -> str:
    mapping = {"📘": "badge-doc", "🔎": "badge-note", "📑": "badge-legend"}

    def repl(m: re.Match) -> str:
        ws, emoji = m.group(1), m.group(2)
        return f'<blockquote class="{mapping[emoji]}">{ws}<p>{emoji}'

    return re.sub(r"<blockquote>(\s*)<p>(📘|🔎|📑)", repl, html)


CSS = """
:root { --fg:#1a1a1a; --muted:#666; --accent:#c0392b; --line:#e3e3e3; --bg:#fff; --code:#f6f8fa; --sidebar:#fbfbfc; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family: -apple-system,"Segoe UI","Hiragino Sans","Yu Gothic UI","Meiryo",sans-serif;
  color: var(--fg); background: #f3f4f6; line-height: 1.85; margin: 0; }
.hero { background: linear-gradient(135deg,#2c3e50 0%,#c0392b 130%); color:#fff; padding: 36px 24px 0; }
.hero .inner { max-width: 1180px; margin: 0 auto; }
.hero h1 { margin: 0 0 .3em; font-size: 1.8rem; border: none; color:#fff; }
.hero p { margin: .2em 0; opacity: .92; font-size: .94rem; }
.hero a { color:#ffe; }
.nav { max-width: 1180px; margin: 16px auto 0; display: flex; gap: 6px; }
.nav a { padding: 9px 18px; border-radius: 8px 8px 0 0; background: rgba(255,255,255,.14); color:#fff;
  text-decoration: none; font-size: .9rem; border: 1px solid rgba(255,255,255,.25); border-bottom: none; }
.nav a.active { background: #f3f4f6; color: var(--accent); font-weight: 600; }
.nav a:hover:not(.active) { background: rgba(255,255,255,.26); }
.layout { max-width: 1180px; margin: 0 auto; display: grid; grid-template-columns: 250px 1fr; gap: 32px; padding: 24px 24px 96px; }
nav.toc { position: sticky; top: 18px; align-self: start; max-height: calc(100vh - 36px); overflow-y: auto;
  background: var(--sidebar); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; font-size: .86rem; }
nav.toc strong { display: block; margin-bottom: 8px; color: var(--accent); }
nav.toc ul { list-style: none; padding-left: 0; margin: 0; } nav.toc ul ul { padding-left: 12px; }
nav.toc li { margin: 3px 0; } nav.toc a { color:#34495e; text-decoration: none; display: block; padding: 2px 0; }
nav.toc a:hover { color: var(--accent); }
article { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 12px 40px 56px;
  box-shadow: 0 1px 10px rgba(0,0,0,.04); min-width: 0; }
article > h1:first-of-type { display: none; }
h2 { font-size: 1.45rem; margin-top: 2.2em; border-bottom: 1px solid var(--line); padding-bottom: .3em; scroll-margin-top: 16px; }
h3 { font-size: 1.16rem; margin-top: 1.7em; color:#222; scroll-margin-top: 16px; }
h4 { font-size: 1.0rem; margin-top: 1.3em; color: var(--accent); }
a { color:#1565c0; text-decoration: none; } a:hover { text-decoration: underline; }
a[href^="#ref"] { font-size:.72em; vertical-align:super; line-height:0; color:#1565c0; background:#eef4fb;
  border:1px solid #cfe0f3; border-radius:4px; padding:0 4px; margin:0 1px; text-decoration:none; white-space:nowrap; }
a[href^="#ref"]:hover { background:#d7e8fb; }
:target { scroll-margin-top: 18px; }
code { background: var(--code); padding: .15em .4em; border-radius: 4px; font-size: .88em;
  font-family: "Cascadia Code",Consolas,"SF Mono",monospace; }
pre { background: var(--code); padding: 15px 18px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--line); }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1.2em 0; font-size: .9rem; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 8px 11px; text-align: left; white-space: nowrap; }
th { background: #f2f4f7; font-weight: 600; } tr:nth-child(even) td { background: #fbfbfc; }
img { max-width: 100%; height: auto; display: block; margin: 1.2em auto; border: 1px solid var(--line); border-radius: 8px; }
.mermaid { background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px; margin:1.4em 0; text-align:center; overflow-x:auto; }
blockquote { border-left: 4px solid var(--accent); margin: 1.2em 0; padding: .4em 1.2em; background:#fdf3f2; color:#444; border-radius: 0 6px 6px 0; }
blockquote.badge-doc { border-left-color:#1565c0; background:#eef4fb; color:#21303f; }
blockquote.badge-note { border-left-color:#b9770e; background:#fdf6e9; color:#3a2e12; }
blockquote.badge-legend { border-left-color:#607d8b; background:#eef1f3; color:#243; }
hr { border: none; border-top: 1px solid var(--line); margin: 2.2em 0; }
footer { max-width: 1180px; margin: 0 auto; padding: 24px; color: var(--muted); font-size: .85rem; text-align: center; }
@media (max-width: 860px) { .layout { grid-template-columns: 1fr; } nav.toc { position: static; max-height: none; } article { padding: 12px 20px 40px; } }
"""

MERMAID_JS = """
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
</script>
"""


def build_nav(active_key: str, pages: list[dict]) -> str:
    links = []
    for p in pages:
        cls = ' class="active"' if p["key"] == active_key else ""
        links.append(f'<a href="{p["out"]}"{cls}>{p["nav"]}</a>')
    return f'<nav class="nav">{"".join(links)}</nav>'


def render(page: dict, pages: list[dict]) -> str:
    cfg = CONFIG
    md_text = embed_images((ROOT / page["md"]).read_text(encoding="utf-8"))
    md_text, mermaid_blocks = extract_mermaid(md_text)
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "codehilite", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False}, "toc": {"toc_depth": "2-3"}},
    )
    body = md.convert(md_text)
    body = inject_mermaid(body, mermaid_blocks)
    body = rewrite_links(body, pages, cfg["repo_url"])
    body = style_badges(body)
    toc = md.toc
    nav = build_nav(page["key"], pages)
    upstream = (f' · Upstream: <a href="{cfg["upstream_url"]}">{cfg["upstream_url"]}</a>'
                if cfg["upstream_url"] else "")
    mermaid_js = MERMAID_JS if mermaid_blocks else ""
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{cfg['hero_title']}</title>
<style>{CSS}</style>
</head>
<body>
<header class="hero"><div class="inner">
  <h1>{cfg['hero_title']}</h1>
  <p>{page['subtitle']}</p>
</div>{nav}</header>
<div class="layout">
  <nav class="toc"><strong>目次</strong>{toc}</nav>
  <article>{body}</article>
</div>
<footer>Source: <a href="{cfg['repo_url']}">{cfg['repo_url']}</a>{upstream}</footer>
{mermaid_js}
</body>
</html>"""


def main() -> None:
    pages = [p for p in CONFIG["pages"] if (ROOT / p["md"]).exists()]
    if OUTDIR.exists():
        shutil.rmtree(OUTDIR)
    OUTDIR.mkdir(parents=True)
    (OUTDIR / ".nojekyll").write_text("")
    for p in pages:
        html = render(p, pages)
        (OUTDIR / p["out"]).write_text(html, encoding="utf-8")
        print(f"wrote {OUTDIR.name}/{p['out']} ({len(html.encode()) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
