# Verified primary references and where they fit

Sources checked during the audit (2026-09-30). These are conceptual attributions, not imported implementation code or external datasets. Do not use published benchmark scores in place of our experiments.

| Component / claim | Reference | Primary source |
|---|---|---|
| PIQA task and physical commonsense | Bisk, Zellers, Le Bras, Gao and Choi (2020), *PIQA: Reasoning about Physical Commonsense in Natural Language* | https://ojs.aaai.org/index.php/AAAI/article/view/6239 ; DOI 10.1609/aaai.v34i05.6239 |
| Multihead scaled dot-product attention / Transformer | Vaswani et al. (2017), *Attention Is All You Need* | https://arxiv.org/abs/1706.03762 |
| Concatenation, subtraction and product comparison features | Chen et al. (2017), *Enhanced LSTM for Natural Language Inference* | https://aclanthology.org/P17-1152/ ; DOI 10.18653/v1/P17-1152 |
| Additive attention lineage | Bahdanau, Cho and Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate* (2014 preprint; ICLR 2015) | https://arxiv.org/abs/1409.0473 |
| BPE subwords | Sennrich, Haddow and Birch (2016), *Neural Machine Translation of Rare Words with Subword Units* | https://aclanthology.org/P16-1162/ ; DOI 10.18653/v1/P16-1162 |
| Masked-token objective | Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | https://aclanthology.org/N19-1423/ ; DOI 10.18653/v1/N19-1423 |
| Pretrained encoder baseline | Liu et al. (2019), *RoBERTa: A Robustly Optimized BERT Pretraining Approach* | https://arxiv.org/abs/1907.11692 |
| Open-model baseline | Qwen: An Yang et al., *Qwen2.5 Technical Report* (v1 2024; v2 2025) | https://arxiv.org/abs/2412.15115v2 |
| Limits of attention-based explanation | Jain and Wallace (2019), *Attention is not Explanation* | https://aclanthology.org/N19-1357/ |

The supplied `references.bib` contains compact entries. For a final ACL report, prefer the publishers'
complete BibTeX exports, retain author spelling and select a consistent Qwen preprint version/year.
The baseline LSTM is a standard PyTorch LSTM; if a separate historical LSTM citation is desired,
verify the original Hochreiter/Schmidhuber publication before adding details. The pooling is not claimed
to reproduce Bahdanau's translation architecture; it adapts the additive scoring idea to solution summaries.

Suggested contribution wording: "We adapt established sequence encoding and matching components to
PIQA by supplying label-free solution-difference tags, comparing goal-conditioned rival solutions, and
biasing solution pooling towards their differences. We evaluate this combination from random
initialisation under the supplied-data constraint." Avoid "we invented attention", "novel Transformer",
or claims of causal physical reasoning unsupported by the experiments.
