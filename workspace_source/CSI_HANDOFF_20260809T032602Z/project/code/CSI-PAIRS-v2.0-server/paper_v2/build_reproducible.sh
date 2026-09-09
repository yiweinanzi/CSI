#!/usr/bin/env bash
set -euo pipefail

PAPER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_INPUT="${1:?usage: build_reproducible.sh UNUSED_OUTPUT_DIRECTORY}"
LATEXMK_BIN="${CSI_PAIRS_LATEXMK:-latexmk}"

if [[ -e "${OUTPUT_INPUT}" ]]; then
  echo "refusing to overwrite paper build directory" >&2
  exit 4
fi
mkdir -p "${OUTPUT_INPUT}"
OUTPUT="$(cd "${OUTPUT_INPUT}" && pwd)"

cp -p "${PAPER_ROOT}/main.tex" "${OUTPUT}/"
cp -p "${PAPER_ROOT}/references.bib" "${OUTPUT}/"
STYLE_ROOT="${PAPER_ROOT}/../paper/official_style/iclr2027"
cp -p "${STYLE_ROOT}/iclr2027_conference.sty" "${OUTPUT}/"
cp -p "${STYLE_ROOT}/iclr2027_conference.bst" "${OUTPUT}/"
cp -p "${STYLE_ROOT}/fancyhdr.sty" "${OUTPUT}/"
cp -p "${STYLE_ROOT}/natbib.sty" "${OUTPUT}/"

export SOURCE_DATE_EPOCH=1785888000
export FORCE_SOURCE_DATE=1
export TZ=UTC

cd "${OUTPUT}"
"${LATEXMK_BIN}" -gg -pdf -interaction=nonstopmode -halt-on-error main.tex
echo "${OUTPUT}/main.pdf"
