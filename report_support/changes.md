# Changes and their marking relevance

The original implementation was fully read and docs/PROJECT_AUDIT.md was written before source changes.
The A100 outputs remain byte-for-byte unchanged. `evidence/historical` preserves original implementation
and notebook; the submission candidate retains original code/output pairs, with corrected prose and an
actually executed post-run audit appendix. Current source generates a separate reproduction candidate.

| File | Original behaviour | New behaviour | Reason | Assignment requirement addressed |
|---|---|---|---|---|
| `src/config.py` | Split seed ambiguous in per-run log; no deterministic option or runtime manifest; ignored amp flag downstream | Distinct run seed, opt-in deterministic algorithms, package/device provenance, UTF-8 logs | Reconstruct what was actually run | Reproducibility and justified training settings |
| `src/data.py` | String coercion could hide malformed fields; platform encoding; directional alignment undocumented; tokenizer overwritten | Strict nonempty-string/label validation, UTF-8, protected tokenizer, data/split hashes, optional canonical alignment | Detect bad data, retain evidence, distinguish historical/new protocols | Dataset pipeline, leakage audit, reproducibility |
| `src/model.py` | MLM head produced vocabulary logits at every position | Optional selected-position projection after the unchanged encoder | Lower memory while preserving the mathematical MLM objective; loss/gradient test executed | Efficient and correct training |
| `src/train.py` | No validation loss/environment/seed metadata; bare state dict; full MLM logits; all-unmasked batch could yield NaN; reused paths overwritten | Validation loss, sidecars, run/environment/split metadata, nonfinite checks, empty-MLM-batch skip, selected-position MLM, honoured AMP, refusal to overwrite runs | Durable, auditable training and checkpoint evidence | Training/evaluation logs and saved model reproducibility |
| `src/experiments.py` | Predictions only in memory; no per-run parameter count in result rows | All validation/test prediction rows saved; actual run seed and params; safer state-dict loading | Recompute accuracy and pairwise statistics from executed decisions | Quantitative comparison, error analysis and reproducibility |
| `src/evaluate.py` | Accuracy/bootstrap/McNemar only | Structured prediction export and Holm adjustment helper | Separate original unadjusted tests from an explicit comparison-family correction | Statistical rigour; no fabricated predictions |
| `src/baselines.py` | B1/B3 test inference during training-stage cells; B3 weights discarded; B4 revision absent | Deferred test inference in new notebook; B3 model/tokenizer retained; B3/B4 resolved revision logging; model seed utility | Make final inference stage explicit and preserve baseline evidence | Own baselines, test protection and reproducibility |
| `src/viz.py` | Fixed-size 120-dpi images; no numeric attention export or explicit prediction label | 300-dpi images, size based on token count, common pooling colour scale, explicit gold/prediction, numeric JSON sidecars, slice caveat | Readable and auditable attention evidence | Successful/failed attention analysis |
| `tools/build_notebook.py` | Default regeneration erased saved outputs | Default separate reproduction candidate; refuses any executed target; UTF-8 and valid v4.5 cell IDs | Protect authentic evidence | Complete reproducible notebook and code/results consistency |
| `tools/refinement_notes.py` | Unsupported static claims embedded in builder | Corrected historical analysis plus fresh-run instructions/prose; offline smoke skips B3/B4 | Avoid attaching old conclusions to new or tiny runs | Critical analysis and honest reporting |
| `tools/review_historical_notebook.py` | No controlled narrative-only update route | Updates markdown from archived notebook and checks original code/output equality | Correct interpretation without disguising old experiments | Report/code consistency |
| `tools/append_audit_cells.py` | Data/gradient audit absent in official notebook | Adds actual locally executed statistics and original-model checks | Keep relevant audit outputs in the implementation deliverable | Saved logs/results and correct attention implementation |
| `CITS4012_69.ipynb` | 48-cell historical run with overclaims | Corrected text; original 27 code cells and outputs preserved; 2 executed audit code cells and explanatory markdown added | Preserve complete study while disclosing findings | Submission evidence and critical analysis |
| `notebooks/CITS4012_69_reproducible.ipynb` | Missing | Complete current-source notebook, generated separately; full Colab run pending | Provide a safer repeatable execution route | Clean runtime compatibility |
| `tools/audit_evidence.py` | Manual/incomplete checks | Repeatable raw lengths, duplicates/conflicts, split/tokenizer hashes, log/JSON/CSV checks, counts, Holm reporting, training curves | Produce report tables from real artifacts | Dataset documentation, fair experiments, actual results |
| `tools/verify_model.py` | Encoded-swap print only | Asserted full-model masks, gradients, padding, swaps, fused attention; reduced-model overfit; checkpoint/overwrite and MLM equivalence tests | Demonstrate correctness before expensive training | Implementation soundness |
| `tools/smoke_notebook.py` | Full pretrained downloads in smoke path | Fresh-kernel offline tiny execution with timestamped saved outputs | Test notebook order without using official test or pretending to run baselines | Reproducibility |
| `tools/reproduce_main.py` | Original weights absent locally | Predeclared fixed-recipe replication with all seeds before test, retained checkpoints/predictions, duplicate sensitivity and attention sidecars | Recover independently auditable current evidence, not original artifacts | Training, evaluation, attention and error evidence |
| `tools/replicate_lexical_baselines.py` | Original B0/B1 predictions absent | Separate actual local B0/B1 runs with fitted LR/vectorizer and predictions | Enable local paired comparison without altering historical table | Own baseline experiments |
| `tools/report_figures.py` | Only ASCII architecture drawing | Scalable SVG and 300-dpi PNG matching actual architecture | Usable methodology figure | Architecture diagram |
| `README.md`, `requirements*.txt` | One-line README, unpinned setup | Exact commands, artifact roles, historical dependency pins, clean Colab checklist and limitations | Make execution/review actionable | Reproducibility and submission preparation |
| `report_support/` | Missing | Dataset/equation/architecture/experiment/qualitative/limitation/story notes; verified tables; bibliography; actual test manifests | Supply evidence for a concise ACL report | All report-support requirements |
| `evidence/` | Missing | Original snapshots/hashes and actually executed development notebooks | Establish artifact provenance | Preservation and consistency |
| `.gitignore` | Temporary PDF renders unignored | `tmp/` ignored | Keep scratch review files out of deliverables | Repository hygiene |

No architecture family, hidden size, loss, historical split, historical length cap or historical tag
algorithm was replaced. `symmetric_diff_tags=True` is opt-in and has no historical result attributed to it.
No remote repository was created, modified or pushed. No final report or team-contribution claims were invented.

## Completed replication analysis and handover

| File | Original behaviour | New behaviour | Reason | Assignment requirement addressed |
|---|---|---|---|---|
| `tools/analyse_replication.py` | No independent local item-level verification | Checks 17,250 rows against supplied text/labels, metrics/log maxima/CI/figure metadata; paired bootstrap/McNemar/Holm; seeded error sample; hashes | Make new comparisons auditable | Actual results, statistical rigour, error analysis |
| `tools/replication_report_figures.py` | Long complete maps impractical at report size | Compact measured top-six pooling bars plus omitted-mass sum, SVG/PNG | Legible success/failure contrast without changing weights | Attention interpretation/report figures |
| `tools/build_replication_evidence.py` | Local training artifacts only in files | Executes a fresh verification notebook reading genuine saved logs and recomputing metrics | Store analysis outputs in a notebook with explicit provenance | Saved outputs/reproducibility |
| `tools/final_checks.py` | Final consistency check not repeatable | Verifies original hashes/code-output pairs, notebook schemas/errors, source embedding, smoke equality and local artifact hashes | Catch source/evidence drift | Code/results consistency |
| `tools/write_file_inventory.py` | Artifact groups only | Enumerates every modified/new project file and completed local run artifact | Clear handover and backup scope | Reproducibility/deliverable management |
| `docs/REFINEMENT_SUMMARY.md` | Missing | Complete requested 20-section handover, tables and readiness checklist | Answer all review questions in one entry point | Report/submission planning |
| `docs/DEVELOPMENT_LOG.md` | Earlier A100 history only, some outdated interpretations | Sections 22–23 preserve history and record corrections, actual fixes/checks/results and remaining work | Fulfil explicit log-maintenance request | Evidence provenance |
| `report_support/replication*`, `final_checks.json`, `files_changed.md`, compact attention/training figures | Missing | Separate local results, sampled errors, measured attention, manifests and checks | Preserve an independent replication without mixing studies | Experimental investigation and critical analysis |
| `evidence/CITS4012_69_replication_verification.ipynb` | Missing | Actually executed log-reading/metric-verification/figure notebook; does not impersonate training cells | Traceable saved analysis | Notebook evidence |
