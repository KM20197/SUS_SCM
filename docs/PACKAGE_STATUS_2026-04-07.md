# Package status as of 2026-04-07

## What was updated

- Replaced the manuscript shipped in the original scaffold with `paper/Manuscript_2026-04-01.docx`.
- Moved the previous manuscript to `paper/archive/Manuscript_2026-02-07_UPDATED_consistency_v13.docx`.
- Added a precise list of server-side source files still missing from `src/`.
- Updated repository notes to distinguish between files included now and files still pending.

## What is present

- Repository scaffold (`README.md`, `LICENSE`, `.gitignore`, `requirements.txt`).
- Runner shell scripts under `scripts/`.
- Minimal documentation under `docs/`.
- Current and archived manuscript DOCX files.

## What is still missing before a final public GitHub push

### A. Core source code
- `src/Code4_colab_fixed_v12_actionscale_minpatch.py`
- `src/code4_diag_hooks.py`
- `src/code4_tf_predict_safe_patch.py` (if imported by the main runner)
- any other local modules imported by the main runner

### B. Materials promised in the manuscript but not available in the current workspace
- validation CSVs for the statistical replication pipeline
- synthetic-parameter files and routing-coordinate instances
- post-processing/table-generation scripts used to create final tables

### C. Items that should normally be attached to a GitHub Release, not committed to Git history
- large audit bundles (`*_AUDIT.tar.gz`)
- matching checksum files (`*.sha256`)
- bulky run folders under `runs/`

## Recommendation

Use this package as the reviewed repository scaffold. Before publishing, copy the missing source files and any public validation/synthetic/post-processing materials that you intend to share, then rerun the release checklist in `docs/RELEASE_CHECKLIST.md`.
