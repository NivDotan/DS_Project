Professor helper bundle for `Multi-Class Solar Flare Prediction Notebook.ipynb`.

Contents:
- `sources/`: copied source files that contain the original Stage 1 / Stage 3 definitions used by the notebook.
- `runtime.py`: lazy loader that reconstructs the notebook symbols from those copied sources.
- `paths.py`: relative path resolver rooted at the notebook folder.
- `symbol_index.json`: quick mapping from notebook symbol name to bundled source file.

Main behavior:
- `section6_runtime_bundle_minimal.py` imports this package and injects the notebook's missing external symbols into the IPython user namespace on import.
- `load_block(...)` rebuilds the requested code block from the bundled source copies instead of relying on the original notebook history.

Files to send with the notebook:
- `Multi-Class Solar Flare Prediction Notebook.ipynb`
- `section6_runtime_bundle_minimal.py`
- `project_paths.py`
- `stage1_w72_notebook_block.py`
- `stage1_multirun_render.py`
- `stage2_notebook_results.py`
- `stage2_section_helpers.py`
- `stage3_matrix_helpers.py`
- `stage3_notebook_results.py`
- this `professor_notebook_helpers/` folder

Environment note:
- MiniRocket helpers still require `sktime` if those specific blocks are loaded or executed.
- The notebook result helper files expect the project data/results folders to sit relative to the notebook root.
