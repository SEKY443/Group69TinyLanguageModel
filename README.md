# Group69TinyLanguageModel

CITS4012 group project (Group 69). The dataset is PIQA (Physical Interaction QA). The main model is **DACT** (Difference-Aware Contrastive Transformer), trained from scratch.

| Path | Contents |
|---|---|
| `CITS4012_69.ipynb` | Submitted notebook: data processing, model, baselines, experiments, analysis |
| `src/` | The same code as Python modules (the notebook's module cells are copied from here) |
| `tools/sync_notebook.py` | Copies `src/` into the notebook; `--check` reports cells that are out of date |
| `outputs/` | JSON-lines training logs, result files and figures of the Colab runs |
| `references.bib` | BibTeX for the report |
| `docs/DEVELOPMENT_LOG.md` | Step-by-step record of everything done on the project |

## Running
1. Put the provided data zip at `MyDrive/Group69/PIQA.zip`.
2. Open `CITS4012_69.ipynb` in Colab and select a GPU runtime.
3. Run `Runtime → Run all`. This takes about 60 minutes on an A100.

## Keeping the notebook and `src/` in sync
After editing a module in `src/`, run:

```bash
python tools/sync_notebook.py
```
