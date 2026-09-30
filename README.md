# Group 69: PIQA / Difference-Aware Contrastive Transformer

## Where to find things

```text
Group69TinyLanguageModel/
├── CITS4012_69.ipynb        # executed submission candidate
├── README.md               # start here
├── requirements*.txt       # local and historical environments
├── src/                    # model, data, training and evaluation
├── tools/                  # build, audit, experiment and checking commands
├── notebooks/              # current-source reproduction candidate
├── templates/              # original course notebook template
├── docs/                   # development log, audit, handover and folder guide
├── report_support/         # report notes, tables, figures and verification records
├── outputs/                # unchanged original A100 experimental evidence
├── experiments/
│   ├── completed/          # local main-model and lexical replications
│   └── development/        # smoke tests, checks and interrupted attempt
└── evidence/               # executed checks and preserved historical snapshots
```

Run commands from this repository directory. Supplied data remains at `../../PIQA`; the assignment
brief remains alongside it. See [the folder guide](docs/PROJECT_STRUCTURE.md) for artifact roles,
old-path compatibility and backup advice. `.venv/` and `tmp/` are local tooling/scratch directories.

The main model is a randomly initialised, shared Transformer with joint goal-option self-attention,
cross-solution comparison and difference-biased pooling. Pretrained RoBERTa/Qwen are baselines only.

**Start with [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md).** The original A100 experiment is preserved.
The audit found one normalised exact test item appearing twice in training, directional alignment that
limits raw-input option-swap invariance, missing historical weights/predictions, and overconfident prose.
These limitations are disclosed, not removed from history by changing scores or splits.

| Artifact | Meaning |
|---|---|
| `CITS4012_69.ipynb` | Historical executed notebook; all 27 original code cells and outputs retained; narrative corrected |
| `evidence/historical/` | Exact original notebook, source, builder and SHA-256 manifest |
| `src/` | Refined implementation: provenance, exports, validation loss, finite-loss checks, efficient MLM; historical preprocessing remains default |
| `notebooks/CITS4012_69_reproducible.ipynb` | Generated current-source notebook; full Colab run pending; do not attach historical outputs |
| `evidence/CITS4012_69_development_smoke*.ipynb` | Actually executed development notebook(s); tiny subsets, no B3/B4, no official test inference |
| `outputs/` | Original A100 logs, results, tokenizer and six attention figures; unchanged |
| `experiments/completed/outputs_replication_efficient/` | Completed separate local three-seed replication: test 62.5680 ± 0.2720%; all checkpoints/predictions retained |
| `experiments/completed/outputs_replication_baselines/` | Completed separate local B0/B1 comparison; B1 test 60.8814% |
| `evidence/CITS4012_69_replication_verification.ipynb` | Executed verification of saved local training logs, predictions, comparisons and figures; not a training-execution notebook |
| `report_support/` | Verified data/results tables, equations, architecture, curves, citations and report story |

The complete handover is `docs/REFINEMENT_SUMMARY.md`; `docs/DEVELOPMENT_LOG.md` sections 22 onward record
the audit and actual experiment outcomes. Read `report_support/replication.md` for separate local
results, paired uncertainty, sampled errors and attention interpretation. Do not splice the local
DACT row into the historical A100 ablation table.

## Local setup (PowerShell, repository directory)

```powershell
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# For the local NVIDIA GPU; choose a wheel compatible with your driver:
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch
```

The provided four data files are currently under `../../PIQA`. No external datasets are downloaded.
`requirements-a100.txt` records the versions printed by the historical notebook. A later environment is
not guaranteed to reproduce floating-point trajectories, and the original CUDA run was nondeterministic.

## Audit and tests

```powershell
.venv/Scripts/python.exe tools/audit_evidence.py --data-dir ../../PIQA
.venv/Scripts/python.exe tools/verify_model.py --data-dir ../../PIQA
.venv/Scripts/python.exe tools/report_figures.py
.venv/Scripts/python.exe tools/build_notebook.py --output notebooks/CITS4012_69_reproducible.ipynb --force
.venv/Scripts/python.exe tools/smoke_notebook.py
.venv/Scripts/python.exe tools/analyse_replication.py
.venv/Scripts/python.exe tools/final_checks.py
```

The builder refuses to overwrite **any executed notebook**, including with `--force`. Its default output
is the separate reproduction candidate. Smoke execution uses a fresh kernel, 128-example subsets and a
validation-derived diagnostic holdout, and skips downloads/training of pretrained baselines. Its metrics
must never be reported as official test results. Full-model functional tests check masks, padding
invariance, encoded swaps, gradients, explicit/fused attention agreement and selected-position MLM
objective/gradient equivalence. A reduced DACT must overfit 16 training examples.

## Fixed main-model reproduction

```powershell
.venv/Scripts/python.exe -u tools/reproduce_main.py --data-dir ../../PIQA --out-dir experiments/completed/outputs_replication_new
```

This predeclares seeds 42/43/44 and the validation-selected historical configuration, reuses the saved
training-only tokenizer, trains all seeds before test evaluation, and retains every checkpoint plus
validation/test predictions. Windows uses zero loader workers. It does not search hyperparameters or
enable canonical alignment. The MLM head projects supervised positions only; a numerical test verifies
the same mathematical loss/gradients. Finite-precision trajectories can differ.

All run directories must be fresh. Failed/interrupted evidence stays on disk. The initial
`experiments/development/outputs_replication/` attempt was interrupted for excessive memory use before a complete model was
trained; it is not a completed result. `outputs_*/` and weights are Git-ignored to avoid accidental large
commits, but are retained locally. No remote push was made by the audit.

## Clean Colab procedure

Open `notebooks/CITS4012_69_reproducible.ipynb` in a fresh GPU runtime (A100 recommended for historical B3/B4
settings). Put the supplied ZIP at `MyDrive/Group69/PIQA.zip`, or upload four files into `data/piqa/`.
Run all, then check no error outputs, all declared seeds, result tables, predictions, model sidecars and
attention figures. The last cell copies **all** evidence including checkpoints to a timestamped Drive
directory if Drive is mounted. Download the executed notebook and output directory before ending the
runtime. Do not overwrite the historical candidate until the new full outputs have been reviewed.

Local smoke tests do not verify Drive mounting, a clean Colab installation or the full modified B3/B4
paths. Those remain explicit checks before promoting the reproduction notebook. The historical saved
notebook is still the source of the complete baseline/ablation results.

## Submission

The brief requires exactly `CITS4012_69.ipynb` and `CITS4012_69.pdf`. The report must use ACL LaTeX final
mode, Group 69 only in the author field, at most six main-content pages, BibTeX references and actual
team contributions. **The PDF report has not been written by this audit.** The notes in `report_support/`
prepare the evidence; they are not a completed submission. Historical results and local replication must
be labelled separately, and original missing checkpoints must not be implied to have been recovered.
