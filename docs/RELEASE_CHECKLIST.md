# Release checklist

- Copy the core `src/` files from the execution environment/server:
  - `Code4_colab_fixed_v12_actionscale_minpatch.py`
  - `code4_diag_hooks.py`
  - `code4_tf_predict_safe_patch.py` (if imported by the runner)
  - any additional local modules imported by the runner
- Confirm dependencies and pin versions if needed.
- Decide which manuscript-declared public materials will live inside the repository:
  - validation CSVs
  - synthetic-parameter/routing-instance files
  - post-processing or table-generation scripts
- Keep raw SUS microdata out of GitHub.
- Attach large audit bundles (`*_AUDIT.tar.gz` + `.sha256`) to the GitHub Release instead of committing them to git history.
- Update `CITATION.cff` repository metadata and URL before publication.
