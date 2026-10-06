# Proposed six-page ACL narrative

**Working title:** Difference-Aware Attention for Physical Commonsense Choice: A From-Scratch PIQA Study.

Use the official ACL template, `\usepackage[final]{acl}`, `\author{Group 69}`. The page allocations below total six main-content pages; references and truthful team contributions follow separately. Do not invent contributions.

| Space | Section | Claim and evidence |
|---|---|---|
| 0.7 pages | Introduction and relevant work | PIQA's two competing physical solutions; scratch-learning constraint; ask whether explicit comparison helps and what attention reveals. Cite PIQA, Transformer, ESIM. |
| 1.4 pages | Methodology | Joint goal-option encoding, tags, cross-solution comparison, difference-biased pooling and shared score. Use the exact equations and architecture.svg. Distinguish standard components from the PIQA adaptation; qualify equivariance. |
| 0.9 pages | Experimental setup | Actual course splits, training-only BPE/MLM, lengths, duplicate disclosure, validation selection, exact hyperparameters, accuracy and seed statistics. Summarise complementary baselines and controlled versus bundled ablations. |
| 1.2 pages | Quantitative results | One compact main table (DACT, vanilla, B0/B1/B2/B3/B4); one ablation delta table. DACT improves mean accuracy over vanilla/BiLSTM but not all ablations; lexical baseline is competitive. Explain representative-seed CIs and exploratory multiplicity correction. |
| 1.2 pages | Attention and error analysis | One success and one failure with legible goal/solution/prediction captions; cite the truncation failure in text. Explain initial pooling prior and avoid causal attention claims. Use actual selected cases, not generic categories. |
| 0.6 pages | Limitations and conclusion | Limited data, overfitting curves, overlap, alignment/truncation, uncertainty, artifact gaps, pretrained comparison confounds. Conclude useful scratch learning with weak evidence for each added component; no accuracy-maximising claim. |

Research questions: (1) Can a task-adapted scratch model learn beyond simple reference points? (2) Which
parts of explicit option comparison are supported by controlled ablations? (3) How do attention patterns
and concrete errors delimit the claim of physical commonsense learning? Treat the local reproduction as
a separate reproducibility check if included; do not splice it into the A100 table. There is no need to add
another family of architectures or many more experiments to support this story.
