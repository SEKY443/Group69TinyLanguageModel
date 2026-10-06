"""Predeclared fixed-config replication, no search; historical artifacts never overwritten.

Uses the saved training-only tokenizer and exactly the historical data split/alignment. All seeds are
trained before final evaluation. This local replication is separate from the historical comparison table.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from config import Config,get_device,environment_info
from data import load_piqa_with_test_labels,PIQADataset,make_loader
from experiments import run_experiment,test_experiment
from model import DACT
from viz import select_cases,plot_attention_case
import matplotlib
matplotlib.use('Agg')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default=str(ROOT/'../../PIQA'))
    p.add_argument('--out-dir',default='experiments/completed/outputs_replication_new')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=p.parse_args()
    out=Path(args.out_dir)
    if out.exists():raise FileExistsError('Choose a fresh replication directory')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    # Merge note (2026-10-01): outputs/ now holds the final lexical-head run; the original A100 evidence
    # moved to evidence/historical/a100_outputs/. Options added after the A100 run are pinned to their
    # historical behaviour, because the saved config predates them and the new defaults differ.
    HIST=ROOT/'evidence/historical/a100_outputs'
    historical=json.loads((HIST/'results/dact_full_test.json').read_text())
    legacy={'use_lexical':False,'diff_aware_truncation':False}
    cfg=Config(**{**legacy,**historical['config']}).but(data_dir=args.data_dir,out_dir=str(out),num_workers=0)
    # Worker count changes scheduling only; documented as a local resource difference.
    device=get_device()
    tr,va,te=load_piqa_with_test_labels(cfg,'reproduce_main.py',str(ROOT/'outputs'))
    tok=Tokenizer.from_file(str(HIST/'tokenizer.json'));tok.save(str(out/'tokenizer.json'))
    data={'train':tr,'val':va,'test':te,'tok':tok,'vocab_size':tok.get_vocab_size()}
    for name,rows in [('train',tr),('val',va),('test',te)]:data[name+'_ds']=PIQADataset(rows,tok,cfg)
    protocol={'purpose':'fixed configuration replication; no tuning; historical split retained',
              'seeds':args.seeds,'config':cfg.to_dict(),'environment':environment_info(device),
              'tokenizer_sha256':hashlib.sha256((HIST/'tokenizer.json').read_bytes()).hexdigest(),
              'known_overlap_test_index':1545,'resource_difference':'num_workers=0 on Windows',
              'MLM_projection':'selected supervised positions only; objective/gradient equivalence checked; finite precision can differ',
              'evaluation_rule':'train every declared seed first; representative seed=max validation accuracy; final test once per seed'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    result=run_experiment('dact_replication',cfg,'dact',data,device,args.seeds)
    result=test_experiment(result,data,device)
    print('REPLICATION RESULT',json.dumps({k:result[k] for k in ('val','test')}),flush=True)
    # Saved representative predictions permit duplicate-excluded sensitivity, not model selection.
    pred=result['test_preds'];keep=[i for i in range(len(te)) if i!=1545]
    sensitivity={'excluded_index':1545,'reason':'exact training/test duplicate identified in pre-run audit',
                 'n':len(keep),'accuracy':float((pred['pred'][keep]==pred['label'][keep]).mean()),
                 'note':'sensitivity only, not a fresh unseen benchmark or replacement primary score'}
    (out/'results/duplicate_sensitivity.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf8')
    best=max(result['runs'],key=lambda r:r['val_acc'])
    model=DACT(cfg,data['vocab_size']).to(device)
    model.load_state_dict(torch.load(best['ckpt'],map_location=device,weights_only=True))
    cases=select_cases(pred['pred'],pred['prob'],pred['label'],k=2)
    (out/'figures').mkdir()
    (out/'results/selected_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf8')
    for kind in ('confident_correct','confident_wrong'):
        for i in cases[kind]:
            plot_attention_case(model,te[i],tok,cfg,device,title=f'Local replication {kind} #{i}',
                                save_path=out/'figures'/f'{kind}_{i}.png')
    print('Saved all checkpoints, validation/test predictions, configs and four attention cases.',flush=True)


if __name__=='__main__':main()
