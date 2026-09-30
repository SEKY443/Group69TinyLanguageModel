# Historical attention case analysis

### Qualitative discussion (historical saved attention figures)

The representative checkpoint was chosen on validation (seed 44). The four confident cases are extreme
examples, not a representative sample of all errors. Confidence is a softmax score, not calibrated certainty.

* **Successes #804 and #280:** both select *baby wipes* correctly for cleaning tasks (about .91).
  Pooling favours the substituted nouns. Training phrase counts recorded in the development log
  (26 correct-only versus 5 wrong-only occurrences for baby wipe(s)) are consistent with a lexical-prior
  explanation, but the maps and counts do not prove that this caused either decision.
* **Failure #660:** gold is a cooler filled with ice; the model favours a small chest of water (about .85).
  Pooling concentrates on the differing span, while the cross-solution map is diffuse and includes shared
  words such as *put the*. This suggests a failure to use temperature-related evidence effectively.
  It does not establish that the model lacks a specific fact or uses a particular shortcut internally.
* **Failure #708:** the model favours folding foil in half over crinkling it into a ball (about .83).
  The substituted phrases receive attention; *crinkle* is split into subwords. Rare-word learning and
  weak action-goal matching are plausible limitations, not proven causes.
* **Uncertain failure #1433:** both retained prefixes are identical after the 96-token cap. The distinguishing
  coffee/sugar span lies beyond it, and the displayed probabilities are .50/.50. This is direct evidence
  of information loss during preprocessing. Difference-centred truncation is future work; it was not
  evaluated and must not be described as fixing the answer or improving accuracy.

The saved aggregate statistic is .637 attention mass on differing tokens versus .248 under uniform
pooling; correct/incorrect values are .630/.647 (Mann-Whitney p=.07855). The code excludes examples
without retained differing tokens. The difference bias begins at +1 and ends near +1.019, so much of
the preference is built in before training. This is not evidence that training discovered causal importance.
Goal-attention plots show a slice of last-layer self-attention and omit other keys; cross-solution plots
omit the CLS sink. Displayed rows therefore need not sum to one. Attention weights describe model
behaviour; they are not a complete or causal explanation (Jain and Wallace, 2019,
https://aclanthology.org/N19-1357/).

The shared network swaps scores when already-encoded options are swapped. However, the historical
SequenceMatcher alignment is directional: re-encoding swapped raw options changes tags in 755/14,501
training examples. Thus the full historical input pipeline is not guaranteed permutation-equivariant.
The optional canonical alignment in the refined source is a separately tested future protocol, not the
preprocessing that produced these figures or scores.
