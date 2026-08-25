#!/bin/bash

# Define the directory to clean (current directory by default)
DIR=${1:-.}

# Define common LaTeX auxiliary file extensions
EXTENSIONS=(
    "aux" "log" "blg" "bbl" "out" "toc" "lof" "lot" "idx" "ilg" "ind" "synctex.gz"
    "fls" "fdb_latexmk" "nav" "snm" "vrb" "nlo" "nls" "bak" "gz" "xdv" "dvi" "ps"
)

echo "Cleaning LaTeX auxiliary files in: $DIR"

for ext in "${EXTENSIONS[@]}"; do
    find "$DIR" -type f -name "*.$ext" -delete
done

echo "Cleaning complete."
