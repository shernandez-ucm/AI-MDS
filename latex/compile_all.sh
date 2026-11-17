#!/bin/bash
# Compile every lecture (top-level .tex with \documentclass) in this directory
# and move the resulting PDFs to ../clases.
#
# Usage: ./compile_all.sh              # all lectures
#        ./compile_all.sh clase_3_2    # only the given ones (with or without .tex)

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$(cd "$SRC_DIR/.." && pwd)/clases"
BUILD_DIR="$SRC_DIR/.build"

mkdir -p "$OUT_DIR" "$BUILD_DIR"
cd "$SRC_DIR" || exit 1

if [ $# -gt 0 ]; then
    FILES=()
    for f in "$@"; do FILES+=("${f%.tex}.tex"); done
else
    FILES=($(grep -l '^\\documentclass' *.tex))
fi

ok=0
failed=()
for tex in "${FILES[@]}"; do
    name="${tex%.tex}"
    if [ ! -f "$tex" ]; then
        echo "[skip] $tex not found"
        failed+=("$name")
        continue
    fi

    # The svg package calls Inkscape, which requires shell escape
    opts=()
    if grep -qE '\\usepackage(\[[^]]*\])?\{svg\}' "$tex"; then
        opts+=(-shell-escape)
    fi

    if latexmk -pdf -interaction=nonstopmode -halt-on-error "${opts[@]}" \
               -outdir="$BUILD_DIR" "$tex" > "$BUILD_DIR/$name.latexmk.log" 2>&1; then
        mv -f "$BUILD_DIR/$name.pdf" "$OUT_DIR/"
        echo "[ ok ] $name"
        ok=$((ok + 1))
    else
        echo "[FAIL] $name  (see .build/$name.log)"
        failed+=("$name")
    fi
done

echo
echo "$ok compiled, ${#failed[@]} failed. PDFs in $OUT_DIR"
[ ${#failed[@]} -eq 0 ] || { echo "Failed: ${failed[*]}"; exit 1; }
