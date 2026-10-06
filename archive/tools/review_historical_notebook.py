"""Correct narrative while preserving every historical code cell and output exactly."""
import json
from pathlib import Path
from refinement_notes import QUANTITATIVE,QUALITATIVE,AUDIT_NOTICE

ROOT=Path(__file__).resolve().parents[1]
archive=ROOT/'evidence/historical/CITS4012_69.ipynb'
nb=json.loads(archive.read_text(encoding='utf8'))
before=[c for c in nb['cells'] if c['cell_type']=='code']
for c in nb['cells']:
    if c['cell_type']!='markdown':continue
    s=''.join(c['source'])
    if s.startswith('### Quantitative discussion'):s=QUANTITATIVE.strip()
    elif s.startswith('### Qualitative discussion'):s=QUALITATIVE.strip()
    else:
        s=s.replace('8k merges vocabulary','8,000-entry vocabulary')
        s=s.replace('so **accuracy** is an unbiased metric','and **accuracy** directly measures correct forced-choice decisions')
        s=s.replace('Solutions are short (p99 of about 100 BPE tokens), so we truncate solutions to 96 and goals to 40 tokens.',
                    'The displayed table is post-truncation. An independent audit of untruncated training candidates gives p95 about 65 and p99 108 BPE tokens. The existing 96-solution/40-goal limits retain most inputs but can erase the distinguishing span.')
        s=s.replace('**permutation-equivariant** (swapping options swaps scores, verified below), so it cannot learn a position bias',
                    '**equivariant for encoded option pairs** (the check below); raw-text alignment is directional, so full-pipeline equivariance is not guaranteed')
        s=s.replace('Same budget, recurrent instead of our Transformer design','Same QA epoch cap; different learning rate/dropout, no MLM')
        s=s.replace('Steps 1-4 never touch the test split. Step 5 loads the saved checkpoints and evaluates every model on the test split once.',
                    'Selection uses validation. Test inputs/statistics are accessed earlier and B1/B3 cache test predictions in Step 4; test metrics are computed in Step 5. No test-based checkpoint selection occurs in this code.')
        s=s.replace('final test evaluation (the only use of the test split)','final test metrics (B1/B3 predictions were computed earlier)')
        s=s.replace('options are no longer compared','cross-solution module removed; difference tags still compare options')
        s=s.replace('3 seeds each, one change at a time','3 seeds each; vanilla is a joint ablation')
        s=s.replace('uniform average instead of attention pooling','uniform average, removing learned weights and tag bias')
        s=s.replace('The *test* split is touched only in the clearly marked *Final test evaluation* part of Section 3.',
                    'Final test metrics appear in Section 3; earlier cells inspect inputs/statistics and cache B1/B3 predictions. One normalised training/test duplicate was subsequently identified; see the audit addendum.')
        s=s.replace('any GPU works; we used an A100/L4 on Colab Pro','the saved run used an A100 40 GB; smaller GPUs may require separately documented memory settings')
        s=s.replace('No extra installation is needed: every package used', 'The original runtime supplied every package used')
        s=s.replace('is pre-installed on Colab; the first code cell prints their versions.', 'without installation; future Colab images may differ. The first code cell prints versions. If imports fail, install the recorded versions below before running.')
        if s.startswith('# Readme'):
            s += '\n\n'+AUDIT_NOTICE.strip()+'''\n\n### Historical dependency setup if needed
Run this in a setup cell before the first import (a restart may be needed if packages were already imported):
```python
%pip install torch==2.11.0 numpy==2.1.3 scikit-learn==1.6.1 scipy==1.16.3 pandas==2.2.3 matplotlib==3.10.0 tokenizers==0.23.1 transformers==5.16.1
```
CUDA wheel compatibility depends on the runtime. The saved environment used torch 2.11.0+cu128.
'''
    c['source']=s
after=[c for c in nb['cells'] if c['cell_type']=='code']
assert before==after,'Historical code or outputs changed'
(ROOT/'CITS4012_69.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf8')
print('Updated narrative only; all',len(after),'historical code cells and outputs are byte-equivalent as JSON values.')
