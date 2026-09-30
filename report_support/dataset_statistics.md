# Dataset statistics (executed audit)

Lengths below use the saved historical tokenizer **before truncation**. Training/validation membership matches the historical 90/10 seed-42 split.

| Split | Items | Label 0 / 1 | Exact duplicate excess | Asymmetric tags | Identical encoded options |
|---|---:|---|---:|---:|---:|
| train | 14501 | 7247 / 7254 | 6 | 755 | 55 |
| validation | 1612 | 806 / 806 | 0 | 83 | 7 |
| test | 1838 | 910 / 928 | 0 | 103 | 2 |

| Split / text | Mean | Median | p95 | p99 | Maximum | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| train / goal | 8.299 | 8.0 | 16.0 | 20.0 | 37 | 0/14501 |
| train / candidates_combined | 22.277 | 14.0 | 64.0 | 108.0 | 492 | 426/29002 |
| validation / goal | 8.525 | 8.0 | 16.0 | 21.0 | 33 | 0/1612 |
| validation / candidates_combined | 22.710 | 15.0 | 66.0 | 112.0 | 257 | 44/3224 |
| test / goal | 8.465 | 8.0 | 16.0 | 21.0 | 28 | 0/1838 |
| test / candidates_combined | 22.220 | 15.0 | 65.0 | 105.2 | 223 | 58/3676 |

**Leakage limitation:** one normalised exact test item (index 1545) occurs twice in the training subset. No exact train-validation triple overlap. Repeated goals occur across splits (see JSON) and make IID interpretation imperfect. Historical results remain unchanged. A future duplicate-cleaned protocol would require fresh, separately labelled training and would not restore an unseen test set.

The tokenizer uses NFKC/lowercase and punctuation-sensitive BPE; special IDs are 0..4. Vocabulary size is 8,000. The 40/96 limits are retained from the historical study; training p95/p99 show the coverage tradeoff. Do not use test anomalies to choose new limits. See dataset_audit.json for unknown rates, vocabulary counts, schema and whitespace checks.
