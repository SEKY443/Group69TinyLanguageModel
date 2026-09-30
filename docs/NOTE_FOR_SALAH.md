# Note for Salah: please review `merge-review` before anything goes into `main`

Hi Salah,

While you were working on `main` (your commit `e08e80c`, "updates", 30 Sep 23:53), I was working on a separate branch, `revision-2`. The two lines of work went in different directions, so I did **not** push anything to `main`. I combined both on a review branch, **`merge-review`**, and **`main` is exactly as you left it**. Please check it and decide whether we merge. Nothing happens until you and the group agree.

Everything below is recorded in detail in `docs/DEVELOPMENT_LOG.md`, sections 22–29. Your sections are kept in full as S1–S3.

---

## 1. What `revision-2` changed (in short)

| | `main` (your commit) | `merge-review` |
|---|---|---|
| Model | original DACT (A100 run 1) | DACT **+ lexical head** (hashed unigram/bigram weights, trained jointly from scratch) |
| DACT val / test | 60.1 / 60.4 % | **61.1 / 61.6 %** (3 seeds, free Colab T4) |
| vs TF-IDF + LR (61.2 % test) | below on test | above on both splits; **the test gap is not significant** (McNemar p = 0.88) |
| RoBERTa / Qwen | 68.5 (1 seed) / 75.4 | 67.9 ± 0.8 (3 seeds) / 75.1 |
| Report | notes in `report_support/` | full ACL draft: `report/CITS4012_69.pdf`, 6 pages |

Other changes on `revision-2`:
- **Difference-aware truncation**, which fixes the cobbler item whose inputs became identical.
- **Attention controls:** the focus on differing tokens turns out to be learned (73 % with the bias removed, against 25 % uniform).
- Two tested explanations (H1/H2) for the gap to bag-of-words.
- An error-overlap analysis.
- RoBERTa with 3 seeds.
- Crash-safe resume.
- The bf16-emulation fix for the T4.

**Please read this before judging the numbers:**
- **Protocol:** the lexical head was configured on **validation only**, in a pilot where the test split wasn't loaded. But the decision to *try* a lexical component came after we had seen the earlier test results. The report and notebook both say this and treat the small test gap as indicative only.
- **AI use:** the `revision-2` work, including the report draft, was done with an AI coding assistant (Claude Code). The log records this. The current report has **no AI-use statement**, because it was removed on request. The group needs to decide what to declare under the unit's policy, and that affects all of us.

## 2. What happened to your work in the merge

**Kept as you wrote it:**
- `evidence/`, `report_support/`, `notebooks/`, `experiments/`, `templates/`
- your docs (`PROJECT_AUDIT.md`, `REFINEMENT_SUMMARY.md`, `PROJECT_STRUCTURE.md`, manifests) and `requirements*.txt`
- all of your `tools/` except `reproduce_main.py` (see section 3)

**Code in `src/`:**
- **Combined.** Your determinism settings, `run_seed`, input validation, `symmetric_diff_tags` (still opt-in), prediction export, provenance, validation loss, finite-loss check, selected-position MLM head, the "never overwrite evidence" guards, and your `viz` improvements are all in, alongside my lexical head, truncation and resume.
- **One fix to your code:** your `run_experiment` now stores `run_seed` per seed in the saved config. That stopped my resume check from ever matching, so the check now ignores `run_seed`. Nothing else in your logic changed.

**Moved, not deleted.** Your README said `outputs/` and the top-level notebook are the *unchanged original A100 evidence*. On `merge-review` they hold the final run, so I moved the originals:

| Original location | Now at |
|---|---|
| original A100 `outputs/`, byte-identical to `e08e80c` | `evidence/historical/a100_outputs/` (not `outputs_a100/`, which your `.gitignore` excludes) |
| your audited top-level notebook (corrected narrative and audit cells) | `evidence/historical/CITS4012_69_audited_A100.ipynb` |

**Rewritten:**
- **`README.md`:** it now describes both tracks, and your statements that stopped being true now point to `evidence/historical/`. Please check that I represented your work correctly.
- **`docs/DEVELOPMENT_LOG.md`:** your sections 22–24 were renumbered **S1–S3** because they clashed with my 22–24. The text is unchanged.

**Used in the report:** your audit findings are now in the report:
- the train/test overlap and training duplicates;
- the order-dependent alignment (103 test items), which qualifies the equivariance claim;
- the early loading of the test file;
- the lost A100 checkpoints and predictions.

The notebook's protocol cell no longer claims that "Steps 1-4 never touch the test split". Your local replication is mentioned as separate and is **not** mixed into the report's tables.

## 3. Please check these: things the merge may have broken for you

1. **`tools/reproduce_main.py` (already changed; please confirm).** It read the "historical" config and tokenizer from `outputs/`, which now holds the new run. The old config also lacks the new options, whose defaults are now on, so it would silently have retrained the *new* model. It now reads from `evidence/historical/a100_outputs/` and pins `use_lexical=False, diff_aware_truncation=False`. It compiles, but I have **not** re-run the replication.
2. **Not changed; your call.** These tools still read `outputs/` as if it were the A100 evidence:
   - `tools/audit_evidence.py` (tokenizer, `final_results.csv`, `*_val.json`, logs)
   - `tools/verify_model.py` (tokenizer; it also builds `Config()`, which now has the lexical head **on** by default)
   - `tools/append_audit_cells.py`

   Point them at `evidence/historical/a100_outputs/`, or re-run them on the new outputs, whichever you intend.
3. **`notebooks/CITS4012_69_reproducible.ipynb`** was generated from the pre-merge `src/`, so it's now out of date. `tools/build_notebook.py` builds from `Config()`, which is now the lexical-head model.
4. **None of your tools have been run since the merge.** They need the data at `../../PIQA`, which I don't have. What I did run: a full smoke run of the merged `CITS4012_69.ipynb` (CPU, synthetic data, RoBERTa/Qwen stubbed). Every cell ran and the consistency check passed.

## 4. What's left before submission, whichever way we decide
- **One fresh Colab run of the merged notebook** (about 2 h on a free T4). Its saved outputs come from the `revision-2` code. Your additions don't change the model's defaults, but the submitted outputs should come from exactly the submitted code.
- **Team Contributions** in the report (a placeholder at the moment). That includes your audit and replication work.
- The group's decision on the AI-use statement.

## 5. Your decision

| Option | What happens |
|---|---|
| **A. Merge as is** | `merge-review` → `main`, then the fresh Colab run, then submit the lexical-head model and the current report. |
| **B. Merge with changes** | You adjust `merge-review` first (your tools in section 3, the README wording, the report story), then merge. |
| **C. Don't merge** | Keep `main` and the original model. Then the report must be rewritten around the original results (DACT below TF-IDF on test), and `revision-2`'s improvements aren't submitted. |

How to look at it:

```bash
git fetch origin
git checkout merge-review
git diff origin/main --stat
```

Useful starting points:
- `docs/DEVELOPMENT_LOG.md`: section 28 (the merge) and section 29 (the report changes).
- `README.md`.
- `report/CITS4012_69.pdf`.

Thanks for the audit. It caught real problems that the report now reports honestly.

[your name]
