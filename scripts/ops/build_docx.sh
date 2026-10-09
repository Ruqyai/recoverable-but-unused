#!/usr/bin/env bash
# Rebuild the two Word deliverables from their sources. Kept in-repo because the
# original build script lived in a throwaway scratchpad and was lost once.
#
#   new_supmtion_paper_v2/paper.docx         <- new_supmtion_paper_v2/acl_latex.tex  (English, via pandoc)
#   new_supmtion_paper_v2/paper_arabic.docx  <- new_supmtion_paper_v2/draft_ar.md    (Arabic mirror +
#                                                                                     per-paragraph RTL <w:bidi/>)
#
# The Arabic .docx is a hand-translated mirror of the .tex (NOT English-in-RTL), so
# any numeric change in acl_latex.tex must ALSO be edited into draft_ar.md
# before running this, or the two will drift. (v1 sources live under new_supmtion_paper/;
# v2 is the active review revision — draft_ar.md was reconciled to the v2 tex, seed 42.)
#
# Usage:  bash scripts/ops/build_docx.sh
set -uo pipefail
cd "$(dirname "$0")/../.."          # repo root
PAPER_DIR=new_supmtion_paper_v2
EN_TEX=$PAPER_DIR/acl_latex.tex
AR_MD=$PAPER_DIR/draft_ar.md
EN_OUT=$PAPER_DIR/paper.docx
AR_OUT=$PAPER_DIR/paper_arabic.docx
TMP=$(mktemp -d)

command -v pandoc >/dev/null || { echo "FATAL: pandoc not found"; exit 1; }

echo "[docx] English: $EN_TEX -> $EN_OUT"
( cd "$PAPER_DIR" && pandoc acl_latex.tex -o paper.docx \
    --bibliography=references.bib --resource-path=.:figures ) \
  || { echo "FATAL: English pandoc build failed"; exit 2; }

echo "[docx] Arabic (base): $AR_MD -> tmp"
pandoc "$AR_MD" -o "$TMP/ar_base.docx" --resource-path="$PAPER_DIR:$PAPER_DIR/figures" \
  || { echo "FATAL: Arabic pandoc build failed"; exit 3; }

echo "[docx] Arabic (RTL inject) -> $AR_OUT"
python3 - "$TMP/ar_base.docx" "$AR_OUT" <<'PY'
import sys, zipfile, re, os
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src, "r") as z:
    doc = z.read("word/document.xml").decode("utf-8")
    infos = z.infolist()
    blobs = {i.filename: z.read(i.filename) for i in infos}

def fix_p(m):
    p = m.group(0)
    if "<w:pPr>" in p:
        if "<w:bidi" not in p:
            p = p.replace("<w:pPr>", "<w:pPr><w:bidi/>", 1)
    else:
        p = re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr><w:bidi/></w:pPr>", p, count=1)
    return p

doc = re.sub(r"<w:p\b.*?</w:p>", fix_p, doc, flags=re.S)
print("  paragraphs given <w:bidi/>:", doc.count("<w:bidi/>"))
blobs["word/document.xml"] = doc.encode("utf-8")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
    for i in infos:
        zout.writestr(i, blobs[i.filename])
print("  wrote", dst)
PY

rm -rf "$TMP"
echo "[docx] done:"
for f in "$EN_OUT" "$AR_OUT"; do
  unzip -t "$f" >/dev/null 2>&1 && echo "  ok  $f" || echo "  BAD $f"
done
