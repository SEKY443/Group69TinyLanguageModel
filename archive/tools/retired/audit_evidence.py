"""Read-only audit of historical data/results; write derived evidence to report_support.

No model training, test-driven selection, or overwriting historical outputs occurs here.
Usage: python tools/audit_evidence.py --data-dir ../../PIQA
"""
import argparse
import collections
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import Config, environment_info
from data import load_split, diff_tags, encode_example
from evaluate import holm_adjust
from model import DACT, count_parameters
from baselines import BiLSTMAttention


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "../../PIQA")
    args = parser.parse_args()
    out = ROOT / "report_support"
    out.mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    full = load_split(args.data_dir, "train")
    test = load_split(args.data_dir, "test")
    ti, vi = train_test_split(np.arange(len(full)), test_size=.1, random_state=42,
                              stratify=[r["label"] for r in full])
    splits = {"train": [full[i] for i in ti], "validation": [full[i] for i in vi], "test": test}
    cfg = Config()
    tok = Tokenizer.from_file(str(ROOT / "outputs/tokenizer.json"))
    keys = ("goal", "sol1", "sol2")
    stats, integrity, rowkeys, goalkeys, canonical = [], {}, {}, {}, {}
    anomalies = []
    for name, rows in splits.items():
        rowkeys[name] = [tuple(r[k] for k in keys) for r in rows]
        canonical[name] = [(r['goal'].casefold(), *sorted([r['sol1'].casefold(),r['sol2'].casefold()])) for r in rows]
        answer_sets = collections.defaultdict(set)
        for key,r in zip(canonical[name],rows):
            answer_sets[key].add(r['sol1' if r['label']==0 else 'sol2'].casefold())
        goalkeys[name] = {r["goal"].casefold() for r in rows}
        counters = collections.Counter(rowkeys[name])
        integrity[name] = {"n": len(rows), "label_counts": dict(collections.Counter(r["label"] for r in rows)),
                           "exact_duplicate_excess": sum(n-1 for n in counters.values()),
                           "unique_goals_casefold": len(goalkeys[name]), "asymmetric_tag_items": 0,
                           "identical_encoded_options": 0, "lost_difference_by_truncation": 0,
                           "same_full_token_options": 0}
        integrity[name]['canonical_duplicate_excess'] = len(rows)-len(set(canonical[name]))
        integrity[name]['canonical_conflicting_answer_groups'] = sum(len(v)>1 for v in answer_sets.values())
        lengths = {k: [] for k in keys}
        unknown, tokens, words = 0, 0, collections.Counter()
        for i, r in enumerate(rows):
            ids = [tok.encode(r[k]).ids for k in keys]
            for k, sequence in zip(keys, ids):
                lengths[k].append(len(sequence)); tokens += len(sequence); unknown += sequence.count(1)
                words.update(r[k].lower().split())
            a, b = ids[1][:96], ids[2][:96]
            ta, tb = diff_tags(a,b); ub, ua = diff_tags(b,a)
            asymmetric = (ta,tb) != (ua,ub)
            integrity[name]["asymmetric_tag_items"] += int(asymmetric)
            integrity[name]["identical_encoded_options"] += int(a == b)
            integrity[name]["same_full_token_options"] += int(ids[1] == ids[2])
            integrity[name]["lost_difference_by_truncation"] += int(a == b and ids[1] != ids[2])
            if asymmetric or a == b:
                anomalies.append({"split":name,"index":i,"asymmetric_tags":asymmetric,
                                  "identical_encoded_options":a==b,"goal":r["goal"]})
        lengths["candidates_combined"] = lengths["sol1"] + lengths["sol2"]
        integrity[name].update(unknown_tokens=unknown, token_count=tokens, unknown_rate=unknown/tokens,
                               whitespace_word_types=len(words), word_hapaxes=sum(v==1 for v in words.values()))
        for field, lengths_ in lengths.items():
            limit = 40 if field == "goal" else 96
            stats.append({"split":name,"field":field,"count":len(lengths_),"mean":np.mean(lengths_),
                          "median":np.median(lengths_),"p95":np.percentile(lengths_,95),
                          "p99":np.percentile(lengths_,99),"max":max(lengths_),
                          "truncated_count":sum(n>limit for n in lengths_),"limit":limit})
    overlaps = []
    for a,b in itertools.combinations(splits,2):
        match = set(rowkeys[a]) & set(rowkeys[b])
        overlaps.append({"split_a":a,"split_b":b,"exact_triples":len(match),
                         "canonical_casefold_unordered_overlap":len(set(canonical[a])&set(canonical[b])),
                         "shared_goals_casefold":len(goalkeys[a]&goalkeys[b]),
                         "matches":[{"a_indices":[i for i,k in enumerate(rowkeys[a]) if k==m],
                                     "b_indices":[i for i,k in enumerate(rowkeys[b]) if k==m],
                                     "text":dict(zip(keys,m))} for m in sorted(match)]})
    raw = {}
    for split in ("train","test"):
        rows = [json.loads(line) for line in (args.data_dir/f"{split}.jsonl").read_text(encoding="utf8").splitlines()]
        raw[split] = {"schema_variants": sorted({tuple(sorted(r)) for r in rows}),
                      "nonstring_or_missing":sum(not isinstance(r.get(k),str) for r in rows for k in keys),
                      "empty_text":sum(not r[k].strip() for r in rows for k in keys),
                      "irregular_whitespace_items":sum(any(r[k] != ' '.join(r[k].split()) for k in keys) for r in rows),
                      "raw_identical_options":sum(r['sol1']==r['sol2'] for r in rows)}
    manifest = {"files":{f.name:digest(f) for f in args.data_dir.iterdir() if f.suffix in ('.jsonl','.lst')},
                "tokenizer_sha256":digest(ROOT/'outputs/tokenizer.json'),
                "train_indices":ti.tolist(),"validation_indices":vi.tolist(),"split_seed":42,
                "note":"Historical fixed split; audit does not alter membership."}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (out/'dataset_audit.json').write_text(json.dumps({"raw":raw,"splits":integrity,"overlaps":overlaps},indent=2),encoding='utf8')
    pd.DataFrame(stats).to_csv(out/'dataset_lengths.csv',index=False)
    pd.DataFrame(anomalies).to_csv(out/'preprocessing_anomalies.csv',index=False)
    lines = ['# Dataset statistics (executed audit)','',
             'Lengths below use the saved historical tokenizer **before truncation**. Training/validation membership matches the historical 90/10 seed-42 split.', '',
             '| Split | Items | Label 0 / 1 | Exact duplicate excess | Asymmetric tags | Identical encoded options |',
             '|---|---:|---|---:|---:|---:|']
    for name,d in integrity.items():
        lines.append(f"| {name} | {d['n']} | {d['label_counts'].get(0,0)} / {d['label_counts'].get(1,0)} | {d['exact_duplicate_excess']} | {d['asymmetric_tag_items']} | {d['identical_encoded_options']} |")
    lines += ['', '| Split / text | Mean | Median | p95 | p99 | Maximum | Truncated |','|---|---:|---:|---:|---:|---:|---:|']
    for s in stats:
        if s['field'] in ('goal','candidates_combined'):
            lines.append(f"| {s['split']} / {s['field']} | {s['mean']:.3f} | {s['median']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} | {s['max']} | {s['truncated_count']}/{s['count']} |")
    lines += ['', '**Leakage limitation:** one normalised exact test item (index 1545) occurs twice in the training subset. No exact train-validation triple overlap. Repeated goals occur across splits (see JSON) and make IID interpretation imperfect. Historical results remain unchanged. A future duplicate-cleaned protocol would require fresh, separately labelled training and would not restore an unseen test set.',
              '', 'The tokenizer uses NFKC/lowercase and punctuation-sensitive BPE; special IDs are 0..4. Vocabulary size is 8,000. The 40/96 limits are retained from the historical study; training p95/p99 show the coverage tradeoff. Do not use test anomalies to choose new limits. See dataset_audit.json for unknown rates, vocabulary counts, schema and whitespace checks.']
    (out/'dataset_statistics.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

    final = pd.read_csv(ROOT/'outputs/results/final_results.csv')
    experiment_rows,checks=[],[]
    names=['dact_full',*[f'abl{i}' for i in range(7)],'b2_bilstm']
    for p in sorted((ROOT/'outputs/results').glob('*_val.json')):
        r=json.loads(p.read_text());
        for run in r['runs']:
            log=[json.loads(s) for s in (ROOT/'outputs/logs'/f"{r['name']}_seed{run['seed']}.jsonl").read_text().splitlines()]
            ep=[s for s in log if s['event']=='epoch']; start=next(s for s in log if s['event']=='start')
            assert max(s['val_acc'] for s in ep)==run['val_acc'],p
            checks.append({'experiment':r['name'],'seed':run['seed'],'validation_matches_log':True})
        assert np.isclose(np.mean([x['val_acc'] for x in r['runs']]),r['val']['mean'])
    for index,name in enumerate(names):
        r=json.loads((ROOT/'outputs/results'/f'{name}_test.json').read_text())
        assert np.isclose(np.mean(r['test']['runs']),r['test']['mean'])
        assert np.isclose(np.std(r['test']['runs'],ddof=1),r['test']['std'])
        assert np.isclose(final.iloc[index]['test acc'],r['test']['mean'])
        c=Config(**r['config']);model=(DACT if r['arch']=='dact' else BiLSTMAttention)(c,8000)
        n=count_parameters(model); del model
        for j,run in enumerate(r['runs']):
            log=[json.loads(s) for s in (ROOT/'outputs/logs'/f"{name}_seed{run['seed']}.jsonl").read_text().splitlines()]
            assert next(s['n_params'] for s in log if s['event']=='start')==n
            experiment_rows.append({'model':final.iloc[index]['model'],'experiment':name,'seed':run['seed'],
                                    'parameters':n,'validation_accuracy':run['val_acc'],'test_accuracy':r['test']['runs'][j],
                                    'epochs':run['epochs'],'checkpoint_available':(ROOT/run['ckpt']).exists(),
                                    'provenance':'historical A100 saved logs/JSON; not rerun'})
        final.loc[index,'parameters']=n
        final.loc[index,'representative_seed']=r['test']['best_val_seed']
        final.loc[index,'representative_test_accuracy']=r['test']['best_val_seed_acc']
    final.rename(columns={'test 95% CI':'representative_seed_test_CI95','McNemar p vs DACT':'representative_seed_McNemar_p'},inplace=True)
    mask=final['representative_seed_McNemar_p'].notna()
    final.loc[mask,'Holm_p_12_comparisons']=holm_adjust(final.loc[mask,'representative_seed_McNemar_p'])
    final['provenance']='historical saved results; CIs/p-values not re-derived from predictions'
    final.to_csv(out/'results_table.csv',index=False)
    final.iloc[1:8].to_csv(out/'ablation_table.csv',index=False)
    pd.DataFrame(experiment_rows).to_csv(out/'experiment_summary.csv',index=False)
    (out/'consistency_checks.json').write_text(json.dumps({'checks':checks,'aggregate_arithmetic':'PASS','parameter_counts':'PASS',
        'historical_prediction_recomputation':'UNAVAILABLE: predictions and checkpoints absent',
        'environment':environment_info('audit')},indent=2),encoding='utf8')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for seed in (42,43,44):
        log=[json.loads(s) for s in (ROOT/'outputs/logs'/f'dact_full_seed{seed}.jsonl').read_text().splitlines()]
        ep=[s for s in log if s['event']=='epoch'];x=[s['epoch'] for s in ep]
        for ax,key in zip(axes,('train_loss','train_acc','val_acc')):
            ax.plot(x,[s[key] for s in ep],marker='.',label=f'seed {seed}');ax.set_xlabel('QA epoch');ax.set_ylabel(key.replace('_',' '));ax.grid(alpha=.2)
    axes[2].legend();fig.suptitle('Historical DACT QA training: validation loss was not logged');fig.tight_layout()
    fig.savefig(out/'figures/training_curves.png',dpi=300);plt.close(fig)
    print(json.dumps({'dataset':integrity,'overlaps':[{k:v for k,v in r.items() if k!='matches'} for r in overlaps],
                      'log_runs_verified':len(checks),'outputs':str(out)},indent=2))


if __name__=='__main__':
    main()
