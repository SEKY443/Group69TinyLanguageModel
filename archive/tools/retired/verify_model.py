"""Functional model tests and tiny training-only overfit. Never evaluate the final test set."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import Config,set_seed,get_device,environment_info
from data import load_split,encode_example,collate,PIQADataset,make_loader,diff_tags
from model import DACT,count_parameters
from train import to_device,qa_loss,train_qa,predict,mlm_warmup


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'../../PIQA')
    parser.add_argument('--cpu',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cpu') if args.cpu else get_device()
    set_seed(123)
    rows=load_split(args.data_dir,'train')[:128]
    tok=Tokenizer.from_file(str(ROOT/'outputs/tokenizer.json'))
    cfg=Config(dropout=0,attn_dropout=0,amp=False,batch_size=16,num_workers=0)
    dataset=PIQADataset(rows,tok,cfg)
    b=to_device(collate([dataset[i] for i in range(4)]),device)
    model=DACT(cfg,8000).to(device).eval()
    checks={};start=time.time()
    out=model(b,return_attn=True);fused=model(b)
    checks['shape']=list(out['logits'].shape)
    assert out['logits'].shape==(4,2)
    assert torch.allclose(out['logits'],fused['logits'],atol=2e-5,rtol=2e-4)
    checks['explicit_vs_fused_max_error']=float((out['logits']-fused['logits']).abs().max().detach())
    assert torch.isfinite(out['logits']).all()
    for w in out['self_attn']:
        invalid=~b['attn_mask'][:,:,None,None,:]
        assert w.masked_select(invalid.expand_as(w)).abs().max()==0
        assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-5)
    for k,w in enumerate(out['cross_attn']):
        valid=b['sol_mask'][:,1-k].clone();valid[:,0]=True
        assert w.masked_select((~valid)[:,None,None,:].expand_as(w)).abs().max()==0
        assert torch.allclose(w.sum(-1),torch.ones_like(w.sum(-1)),atol=1e-5)
    assert out['pool'].masked_select(~b['sol_mask']).abs().max()==0
    assert torch.allclose(out['pool'].sum(-1),torch.ones((4,2),device=device),atol=1e-6)
    checks['padding_keys_zero_and_attention_normalised']=True
    padded={k:torch.nn.functional.pad(v,(0,7)) if v.ndim==3 else v for k,v in b.items()}
    assert torch.allclose(out['logits'],model(padded)['logits'],atol=2e-5)
    checks['padding_invariance']=True
    swapped={k:v.flip(1) if v.ndim==3 else 1-v if k=='labels' else v for k,v in b.items()}
    assert torch.allclose(out['logits'],model(swapped)['logits'].flip(-1),atol=2e-5)
    checks['encoded_pair_swap_equivariance']=True
    loss=qa_loss(out['logits'],b['labels'],cfg);loss.backward()
    selected=['layers.0.attn.q.weight','cross.attn.q.weight','pool.v.weight','pool.tag_bias']
    gradients={name:float(dict(model.named_parameters())[name].grad.norm()) for name in selected}
    assert all(v>0 and np.isfinite(v) for v in gradients.values())
    checks['attention_gradient_norms']=gradients
    old=model.pool.mode;model.pool.mode='mean'
    delta=float((model(b)['logits']-out['logits']).abs().max().detach());model.pool.mode=old
    assert delta>1e-8;checks['pooling_intervention_logit_change']=delta
    # Compare full versus selected-position MLM objectives AND gradients with dropout disabled.
    model.zero_grad(set_to_none=True)
    ids=b['input_ids'].flatten(0,1);seg=b['segment'].flatten(0,1)
    tags=b['tags'].flatten(0,1);mask=b['attn_mask'].flatten(0,1)
    selected=(ids>=5)&(torch.arange(ids.shape[1],device=device)[None]%3==0)
    labels=ids.clone();labels[~selected]=-100
    full_logits=model.mlm_logits(ids,seg,tags,mask)
    full_loss=torch.nn.functional.cross_entropy(full_logits.reshape(-1,8000),labels.flatten())
    full_loss.backward()
    full_grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    selected_logits=model.mlm_logits(ids,seg,tags,mask,selected)
    selected_loss=torch.nn.functional.cross_entropy(selected_logits,labels[selected])
    selected_loss.backward()
    assert torch.allclose(full_loss,selected_loss,atol=1e-6)
    maximum=max(float((full_grads[n]-p.grad).abs().max()) for n,p in model.named_parameters() if p.grad is not None)
    assert maximum<2e-5,maximum
    checks['masked_position_MLM_loss_error']=float(abs(full_loss-selected_loss).detach())
    checks['masked_position_MLM_max_gradient_error']=maximum
    del model,out,fused,full_grads,full_logits,selected_logits
    # Covers ambiguous SequenceMatcher alignments on actual training rows.
    asym=0
    for row in rows:
        a,c=[tok.encode(row[k]).ids[:96] for k in ('sol1','sol2')]
        ta,tc=diff_tags(a,c);uc,ua=diff_tags(c,a)
        asym += int((ta,tc)!=(ua,uc))
        ta,tc=diff_tags(a,c,True);uc,ua=diff_tags(c,a,True)
        assert (ta,tc)==(ua,uc)
    checks['historical_raw_swap_tag_failures_in_128']=asym
    checks['canonical_alignment_failures_in_128']=0
    symmetric=cfg.but(symmetric_diff_tags=True)
    tiny=symmetric.but(d_model=64,n_heads=4,n_layers=2,d_ff=128,lr=.003,label_smoothing=0)
    ds=PIQADataset(rows[:16],tok,tiny)
    batch=to_device(collate([ds[i] for i in range(16)]),device)
    set_seed(123)
    model=DACT(tiny,8000).to(device)
    swap_rows=[{**r,'sol1':r['sol2'],'sol2':r['sol1'],'label':1-r['label']} for r in rows[:16]]
    swap_ds=PIQADataset(swap_rows,tok,tiny)
    raw_swap=to_device(collate([swap_ds[i] for i in range(16)]),device)
    model.eval()
    assert torch.allclose(model(batch)['logits'],model(raw_swap)['logits'].flip(-1),atol=2e-5)
    checks['canonical_raw_candidate_swap']=True
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
    history=[]
    for step in range(1,201):
        model.train();opt.zero_grad(set_to_none=True);o=model(batch)
        loss=qa_loss(o['logits'],batch['labels'],tiny)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step==1 or step%10==0:
            model.eval()
            with torch.no_grad():
                logits=model(batch)['logits'];acc=float((logits.argmax(-1)==batch['labels']).float().mean())
            history.append({'step':step,'train_loss':float(loss.detach()),'train_accuracy':acc})
            if acc==1 and step>=20:break
    assert history[-1]['train_accuracy']>=.9375,history
    checks['tiny_overfit']={'n':16,'config':tiny.to_dict(),'history':history,
                           'note':'Training-only diagnostic, reduced model; not a PIQA generalisation result.'}
    # A real QA/MLM/checkpoint round-trip on training subsets only.
    stamp=time.strftime('%Y%m%d_%H%M%S')
    run_cfg=tiny.but(out_dir=str(ROOT/'experiments/development/outputs_verification'/stamp),epochs=1,mlm_epochs=1,run_seed=123)
    model=mlm_warmup(model,ds,run_cfg,device,8000,'tiny_pipeline',verbose=False)
    loader=make_loader(ds,run_cfg,False,device)
    model,history=train_qa(model,loader,loader,run_cfg,device,'tiny_pipeline',verbose=False)
    assert Path(run_cfg.out_dir,'checkpoints/tiny_pipeline.pt.json').exists()
    assert np.isfinite(history[0]['val_loss'])
    checks['pipeline_checkpoint_roundtrip']=True
    try:
        train_qa(model,loader,loader,run_cfg,device,'tiny_pipeline',verbose=False)
        raise AssertionError('overwrite protection failed')
    except FileExistsError:
        checks['evidence_overwrite_guard']=True
    checks['runtime_seconds']=time.time()-start
    checks['environment']=environment_info(device)
    out=ROOT/'report_support';out.mkdir(exist_ok=True)
    (out/'sanity_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
