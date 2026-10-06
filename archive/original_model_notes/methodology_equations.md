# DACT: mathematics of the implemented model

Notation: goal tokens G, candidate tokens A_1,A_2, B items, padded pair length L <= 139, hidden width d=256, heads H=4, head width d_h=64. Input option i is X_i=[CLS,G[:40],SEP,A_i[:96],SEP]. M_i identifies non-padding keys; S_i identifies solution tokens only. T_i is 0 for goal/special, 1 for aligned-shared solution tokens and 2 for unaligned solution tokens. Historical alignment uses directional SequenceMatcher after truncation.

## Embedding and four pre-LN blocks

\[
Z_i^{(0)}=\operatorname{Drop}_{.2}\left(\operatorname{LN}
 (E_{tok}(X_i)+E_{pos}+E_{seg}+E_{tag}(T_i))\right).
\]

Each head uses independent affine Q/K/V projections with biases:

\[
\operatorname{head}_h(U,V;M)=
\operatorname{Drop}_{.1}\!\left[\operatorname{softmax}_{k}
 \left(\frac{(UW_h^Q+b_h^Q)(VW_h^K+b_h^K)^T}{\sqrt{64}}+\mathcal M\right)\right]
(VW_h^V+b_h^V),
\]

where mask entries are 0 for allowed keys and -infinity otherwise. Concatenate heads and apply the affine output projection. Dropout is disabled in evaluation. For each of four encoder blocks:

\[
U=Z+\operatorname{Drop}_{.2}(\operatorname{MHA}(\operatorname{LN}_1 Z,\operatorname{LN}_1 Z;M)),
\qquad Z'=U+\operatorname{Drop}_{.2}(\operatorname{FFN}(\operatorname{LN}_2 U)).
\]

FFN = Linear(256,1024), GELU, dropout(.2), Linear(1024,256). A final encoder LN produces H_i. The same weights encode both options. Joint self-attention connects goal and candidate tokens; there is no separately encoded goal vector or standalone GRU.

## Rival-solution comparison

For j=3-i, set K_j=S_j union {CLS}. This includes a learned no-counterpart sink but no goal or padding keys.

\[
Q_i=\operatorname{LN}(H_i),\quad C_i=\operatorname{MHA}(Q_i,\operatorname{LN}(H_j);K_j),
\quad F_i=[Q_i;C_i;Q_i-C_i;Q_i\odot C_i],
\]
\[
\widetilde H_i=\operatorname{LN}_{out}
\left(H_i+\operatorname{Drop}_{.2}(W_2\operatorname{Drop}_{.2}(\operatorname{GELU}(W_1F_i+b_1))+b_2)\right).
\]

In code `fuse` includes dropout between its two linear maps and `self.drop` is applied to the complete fusion output. Dimensions: F has 1024 features; both fusion output stages have 256. The subtraction is **signed**, not absolute. All query positions are updated; pooling later selects solution positions and the scorer also uses the updated CLS.

## Pooling and shared scores

\[
e_{it}=v^T\tanh(W_p\widetilde h_{it}+b_p)+b^{tag}_{T_{it}},\quad
\alpha_{it}=\operatorname{softmax}_{t\in S_i}(e_{it}),\quad
p_i=\sum_{t\in S_i}\alpha_{it}\widetilde h_{it}.
\]

Tag bias initialises to [0,0,1]. The empty-solution fallback pools CLS; the actual course files have no empty fields. Then

\[
s_i=w_s^T\operatorname{Drop}_{.2}(\operatorname{GELU}(W_s[p_i;\widetilde h_{i,CLS}]+b_s))+b_o,
\qquad P(y=i-1\mid G,A_1,A_2)=\operatorname{softmax}(s_1,s_2)_i.
\]

The scorer maps 512->256->1. Scores have shape [B,2]. Gold label 0 selects sol1; 1 selects sol2. Training minimises mean cross-entropy with smoothed target q=(1-.1)one_hot(y)+.1/2; no pre-softmax is applied before PyTorch cross_entropy. The pointwise ablation uses mean BCE-with-logits against (1-.1)one_hot(y)+.5*.1. Its reported forced-choice decisions still use argmax across the scores; softmax values should not be interpreted as calibrated independent BCE probabilities.

## Training-only MLM

The selected historical configuration first trains for ten epochs on both option sequences from training items, with no validation/test text. Select 15% of non-special tokens; replace 80% with MASK, 10% with a random non-special vocabulary token and leave 10% unchanged. The head is Linear(256,256), GELU, LN followed by the transpose of the token embedding and a vocabulary bias. MLM loss ignores unselected positions. The warm-up uses scratch weights, not an external pretrained model. It does not invoke the cross-solution module or pooling/scorer, though tag embeddings are present in the encoded MLM inputs.

## What is original versus established

Transformer attention, additive pooling, MLM, BPE and concatenation/subtraction/product matching are established techniques. Our adaptation is their specific use for PIQA: label-free comparison tags, shared joint goal-option encoding, rival-solution attention with CLS sink, tag-biased solution pooling, and shared forced-choice scores. Claim a task-specific design, not invention of Transformer, co-attention or matching features. Equivariance holds for encoded pairs; the historical directional alignment limits the raw-input claim.
