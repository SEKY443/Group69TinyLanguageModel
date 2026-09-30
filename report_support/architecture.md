# Architecture and tensor flow

```mermaid
flowchart TD
 G[Goal G] --> X1[CLS G SEP A1 SEP]
 G --> X2[CLS G SEP A2 SEP]
 A1[Candidate A1] --> X1
 A2[Candidate A2] --> X2
 A1 --> T[Align truncated BPE candidates; shared/different tags]
 A2 --> T
 T --> E[Shared token + position + segment + tag embeddings]
 X1 --> E
 X2 --> E
 E --> H[4 shared pre-LN Transformer blocks; B x 2 x L x 256]
 H --> C[Each option attends to rival solution tokens and CLS]
 C --> F[Compare q, c, q-c, q*c; MLP + residual + LayerNorm]
 F --> P[Tag-biased additive pooling over solution tokens]
 F --> CLS[Updated CLS state]
 P --> S[Concatenate: 512; shared 512-256-1 scorer]
 CLS --> S
 S --> O[Two logits; softmax; choose candidate]
```

All neural weights are shared across options; batch flattening is [B,2,L] -> [2B,L] for encoding.
L <= 40+96+3=139, positional table capacity 160. Each attention module has four 64-dimensional heads.
Goal-candidate attention is joint self-attention inside the Transformer; rival attention is a separate step.
The optional MLM branch exits after the encoder, uses a tied vocabulary head and appears only during warm-up.

Use `figures/architecture.svg` for scalable report insertion and `figures/architecture.png` for a preview.
See methodology_equations.md for the exact affine maps, masks and losses.
