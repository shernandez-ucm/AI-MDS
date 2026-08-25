# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for **MDS221 — Inteligencia Artificial**, Magíster en Data Science, Universidad Católica del Maule (instructor: Sergio Hernández). There is no application code: the "source" is a set of **LaTeX Beamer** lecture decks, and the "build output" is PDF slides. All content is written in **Spanish**.

It is not a git repository — there is no history to consult and no commits to make.

## Layout

- `latex/` — the only editable source. One `.tex` file per lecture, self-contained (each has its own full preamble; there is no shared style file).
- `latex/figures/` — figures included by the decks. `ai2026_*.pdf` are charts cropped out of the Stanford HAI *AI Index Report 2026* (each slide cites its figure number); `fig_<n>.<n>.<n>.pdf` are the older equivalents from the *2025* edition, still used by `clase_1_1_2025.tex`. To add another AI Index chart: download the report PDF, extract the page with `mutool merge -o page.pdf report.pdf <n>`, then `pdfcrop --margins 2 --bbox "llx lly urx ury" page.pdf figures/name.pdf` — crop to the white card, since the report's tinted page backgrounds otherwise bleed into the figure. Hand-made diagrams are kept as an `.svg` **and** the `.pdf` exported from it (`xor_problem`, `capacity-vs-error`) — edit the SVG (inkscape is installed), then re-export the PDF; `\includegraphics` references the extensionless name.
- `latex/layers/` — vendored [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) styles (`Ball.sty`, `Box.sty`, `RightBandedBox.sty`, `init.tex`), pulled in by `\subimport{./layers/}{init}` for CNN architecture diagrams. Used only by `clase_2_2.tex` and `clase_2_3.tex`.
- `clases/` — delivered slide PDFs handed to students. Publishing a deck means compiling it in `latex/` and copying the PDF here under the same basename.
- `libros/`, `MDS221(...).pdf`, `rubrica.xls`, `FORMATO DE PLANIFICACIÓN_DIRDOC.docx` — reference books, syllabus and admin paperwork. Read-only inputs; nothing is generated from them. (The syllabus PDF is a scan — `pdftotext` returns nothing.)

## Building

Everything must be compiled **from inside `latex/`**, because `\subimport{./layers/}` and `figures/...` are resolved relative to the working directory.

```bash
cd latex
pdflatex -interaction=nonstopmode -halt-on-error clase_2_3.tex   # one deck
```

**Run `pdflatex` twice.** No deck has a `\tableofcontents` or a bibliography, but the current-style title pages use TikZ `remember picture, overlay`, whose full-bleed background is silently dropped on the first pass — you get a blank title slide. All decks compile cleanly with the installed TeX Live (`neuralnetwork`, `tikzlings`, `pgfplots`, `tcolorbox`, `import` are all present).

To compile without littering the source tree:

```bash
pdflatex -interaction=nonstopmode -output-directory=/tmp/build clase_2_3.tex
```

`./clean.sh [dir]` deletes the LaTeX aux files (`.aux .log .nav .snm .toc .out .synctex.gz` …). Note it also deletes **any** `*.gz` and `*.bak` in the tree, so don't run it over directories holding real data.

## Deck conventions

File naming is `clase_<unidad>_<clase>.tex` (unit 1 = fundamentos/redes neuronales, unit 2 = visión, CNNs, generativos). `clase_1_5.tex` and `clase_1_5_2.tex` are two different decks on loss functions — `_2` is the one that was delivered as `clases/clase_1_5.pdf`. `clase_1_1_2025.tex` is the archived pre-2026 version of the intro deck (legacy style, 2025 AI Index figures); don't edit it, and note that `clean.sh` would have deleted it had it been left named `*.bak`.

Every deck now shares one style, taken from `redes_neuronales.tex`: 16:9 `\documentclass[aspectratio=169, 10pt]{beamer}`, UCM palette (`UCMBlue` RGB 4,171,226 for frame titles / `UCMNavy` for diagrams), `spanish` babel, `T1` fontenc, a full-bleed TikZ title background with a per-deck math glyph, and `\formula{...}` (a tcolorbox wrapper) for highlighted equations. Copy the preamble from any deck when starting a new one — the preambles are deliberately self-contained, so a style change has to be applied to each file.

Two things the migration from the old 4:3 style left behind, both intentional:

- `mitblue` and `mitgray` are still `\definecolor`d in every preamble, because the decks' inline TikZ refers to them heavily (`fill=mitblue!20`). `mitblue` is now an alias for `UCMNavy`, so those diagrams follow the UCM palette without touching the frame bodies.
- A handful of frames carry `\begin{frame}[shrink]` (LeNet-5 and ResNet in `clase_2_2`, the derivations in `clase_1_5`, the bounding-box losses in `clase_1_5_2`, and a few others). Their content already overflowed in 4:3 and 16:9 is vertically shorter; `[shrink]` is a no-op when the content fits, so leave it in place. Any new overflow shows up as `Overfull \vbox` in the log — all decks currently build with zero.

Diagrams are drawn inline with TikZ/pgfplots rather than imported as images; expect long TikZ blocks inside frames. Commented-out `\includegraphics` lines referencing `figures/mlp` and `figures/mindslab.png` point at files that do not exist — leave them commented.
