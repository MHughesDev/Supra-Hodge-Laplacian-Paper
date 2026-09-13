# arXiv Submission Checklist

## Current manuscript entrypoints

- Upload entrypoint: `main.tex`
- Main manuscript source: `LatexCode.tex`

## Recommended next steps

1. Run a clean LaTeX build locally.
2. Inspect the generated PDF for layout issues, float placement, and bibliography formatting.
3. Prepare a minimal arXiv source bundle.
4. Commit the manuscript state once the PDF is confirmed.

## Minimal arXiv bundle

Include:

- `main.tex`
- `LatexCode.tex`

Exclude:

- `.env`
- `__pycache__/`
- local TeX build artifacts such as `.aux`, `.log`, `.out`, `.toc`, `.fdb_latexmk`, `.fls`, `.synctex.gz`
- Python/demo files unless you intentionally want them as ancillary material

## Pre-upload checks

- Confirm the paper builds from `main.tex`.
- Verify title, author block, and email are exactly as desired.
- Verify the bibliography appears in full and every citation resolves.
- Re-read the abstract, introduction, and discussion for claim calibration.
- Keep the framing honest: the current experiments are synthetic proof-of-concept experiments, not benchmark-scale LLM evaluations.

## Notes from the latest cleanup pass

- The manuscript now uses a stable empty `\date{}` rather than `\today`.
- Previously uncited bibliography items are now cited in the application and limitations sections.
- Common LaTeX build artifacts are now ignored in `.gitignore`.

## Current build status

- A local portable `tectonic.exe` build succeeded from `main.tex`.
- `main.pdf` is available for visual review before upload.
