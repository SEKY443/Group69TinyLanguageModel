"""Execute a self-contained audit appendix against the ORIGINAL model/data implementation.

Original code cells and outputs are preserved. New outputs are captured from actually executed cells.
The notebook cells use its own DATA/Config/DACT; no filesystem dependency on this script is introduced.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evidence/historical/src'))
from config import Config,set_seed
from data import load_piqa,PIQADataset,collate,diff_tags
from model import DACT,count_parameters
from train import qa_loss
from tokenizers import Tokenizer
import torch
import numpy as np
import pandas as pd

DATA_CODE=r'''
# Post-evaluation audit: original split and tokenizer, no redesign or model selection.
from collections import Counter
AUDIT_ROWS=[]
AUDIT_KEYS={}
for split in ("train","val","test"):
    rows=DATA[split]
    goal_lengths=[len(DATA["tok"].encode(r["goal"]).ids) for r in rows]
    sol_lengths=[len(DATA["tok"].encode(r[k]).ids) for r in rows for k in ("sol1","sol2")]
    AUDIT_KEYS[split]=[tuple(r[k] for k in ("goal","sol1","sol2")) for r in rows]
    asymmetric=0; lost=0
    for r in rows:
        a,b=[DATA["tok"].encode(r[k]).ids for k in ("sol1","sol2")]
        ta,tb=diff_tags(a[:96],b[:96]);ub,ua=diff_tags(b[:96],a[:96])
        asymmetric+=int((ta,tb)!=(ua,ub))
        lost+=int(a!=b and a[:96]==b[:96])
    counts=Counter(r["label"] for r in rows)
    AUDIT_ROWS.append({"split":split,"items":len(rows),"label_0":counts[0],"label_1":counts[1],
       "goal_mean_raw":np.mean(goal_lengths),"goal_p95_raw":np.percentile(goal_lengths,95),
       "candidate_mean_raw":np.mean(sol_lengths),"candidate_median_raw":np.median(sol_lengths),
       "candidate_p95_raw":np.percentile(sol_lengths,95),"candidate_p99_raw":np.percentile(sol_lengths,99),
       "candidate_truncated":sum(n>96 for n in sol_lengths),"raw_swap_tag_changes":asymmetric,
       "lost_difference_items":lost,"duplicate_excess":len(rows)-len(set(AUDIT_KEYS[split]))})
print(pd.DataFrame(AUDIT_ROWS).round(3).to_string(index=False))
for a,b in (("train","val"),("train","test"),("val","test")):
    print(a,b,"exact normalised input-triple overlap:",len(set(AUDIT_KEYS[a])&set(AUDIT_KEYS[b])))
print("Historical split retained. One exact test item occurs twice in train. Raw-text option swapping is not guaranteed equivariant.")
'''

MODEL_CODE=r'''
# Executed audit of the ORIGINAL network, using training rows only; no benchmark accuracy reported.
AUDIT_DEVICE=torch.device("cpu")
torch.set_num_threads(4)
set_seed(123)
audit_cfg=Config(dropout=0,attn_dropout=0)
audit_ds=PIQADataset(DATA["train"][:4],DATA["tok"],audit_cfg)
audit_batch=collate([audit_ds[i] for i in range(4)])
audit_model=DACT(audit_cfg,DATA["vocab_size"]).eval()
audit_out=audit_model(audit_batch,return_attn=True)
audit_fast=audit_model(audit_batch)
assert torch.allclose(audit_out["logits"],audit_fast["logits"],atol=2e-5)
for w in audit_out["self_attn"]:
    invalid=(~audit_batch["attn_mask"])[:,:,None,None,:].expand_as(w)
    assert (w.masked_select(invalid)==0).all()
    assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-5)
assert (audit_out["pool"].masked_select(~audit_batch["sol_mask"])==0).all()
assert torch.allclose(audit_out["pool"].sum(-1),torch.ones(4,2),atol=1e-6)
swapped={k:v.flip(1) if v.ndim==3 else v for k,v in audit_batch.items()}
assert torch.allclose(audit_out["logits"],audit_model(swapped)["logits"].flip(-1),atol=2e-5)
qa_loss(audit_out["logits"],audit_batch["labels"],audit_cfg).backward()
for name in ("layers.0.attn.q.weight","cross.attn.q.weight","pool.v.weight","pool.tag_bias"):
    norm=float(dict(audit_model.named_parameters())[name].grad.norm())
    assert norm>0 and np.isfinite(norm)
    print(name,"gradient norm",norm)
print("PASS: explicit/fused logits, self-attention padding masks, normalisation, solution-only pooling and encoded option swaps.")
print("Total trainable parameters (including MLM-only head):",count_parameters(audit_model))
print("These are local functional checks, not a rerun of the historical training study.")
del audit_model,audit_out,audit_fast
'''

def main():
    cfg=Config(data_dir=str(ROOT/'../../PIQA'))
    tr,va,te=load_piqa(cfg)
    tok=Tokenizer.from_file(str(ROOT/'outputs/tokenizer.json'))
    namespace=dict(globals(),DATA={'train':tr,'val':va,'test':te,'tok':tok,'vocab_size':tok.get_vocab_size()})
    nb=json.loads((ROOT/'CITS4012_69.ipynb').read_text(encoding='utf8'))
    nb['cells']=[c for c in nb['cells'] if not c.get('metadata',{}).get('audit_appendix')]
    original=[c for c in nb['cells'] if c['cell_type']=='code']
    nb['cells'].append({'cell_type':'markdown','metadata':{'audit_appendix':True},'source':
        '## Post-run audit appendix (executed locally, 2026-09-30)\n\nThe following cells were actually executed against the original implementation and supplied data. They add diagnostics without modifying the historical model, split, code or results. Full refinement tests and any later replication have separate provenance in the repository.'})
    for i,source in enumerate((DATA_CODE,MODEL_CODE)):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(source,namespace)
        nb['cells'].append({'cell_type':'code','metadata':{'audit_appendix':True},'execution_count':None,
            'source':source.strip(),'outputs':[{'output_type':'stream','name':'stdout','text':stream.getvalue()}]})
    assert [c for c in nb['cells'] if c['cell_type']=='code' and not c['metadata'].get('audit_appendix')]==original
    for i,c in enumerate(nb['cells']):c.setdefault('id',f'group69-historical-{i:03d}')
    nb['nbformat_minor']=max(5,nb.get('nbformat_minor',0))
    (ROOT/'CITS4012_69.ipynb').write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf8')
    print('Added two actually executed audit cells; original',len(original),'code cells/outputs unchanged.')

if __name__=='__main__':main()
