"""Recompute local replication evidence; descriptive, post-training analysis only."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from project_paths import resolve_artifact
from evaluate import bootstrap_ci, holm_adjust, mcnemar_exact
from config import Config
from data import load_piqa_with_test_labels


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def write_csv(path, records):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    out = ROOT / 'report_support'
    run = ROOT / 'experiments/completed/outputs_replication_efficient'
    base = ROOT / 'experiments/completed/outputs_replication_baselines'
    result = read(run / 'results/dact_replication_test.json')
    baseline = read(base / 'results.json')
    _, raw_val, raw_test = load_piqa_with_test_labels(Config(data_dir=str(ROOT / '../../PIQA')),
                                                      'analyse_replication.py', str(ROOT / 'outputs'))
    records, checks, test_rows = [], [], {}
    for split, n in [('val', 1612), ('test', 1838)]:
        reference = None
        raw = raw_val if split == 'val' else raw_test
        for name in ['seed42', 'seed43', 'seed44', 'B0', 'B1']:
            path = (base / 'predictions' / f'{name}_{split}.jsonl' if name.startswith('B')
                    else run / 'predictions' / f'dact_replication_{name}_{split}.jsonl')
            data = rows(path)
            assert len(data) == n
            identity = [(x['index'], x['goal'], x['candidate_1'], x['candidate_2'], x['gold_label']) for x in data]
            if reference is None:
                reference = identity
                assert identity == [(i, x['goal'], x['sol1'], x['sol2'], x['label']) for i, x in enumerate(raw)]
            assert identity == reference
            assert [x['index'] for x in data] == list(range(n))
            for x in data:
                assert x['correct'] == (x['predicted_label'] == x['gold_label'])
                if 'probability_candidate_2' in x:
                    p = x['probability_candidate_2']
                    assert 0 <= p <= 1 and np.isclose(p + x['probability_candidate_1'], 1)
                    assert x['predicted_label'] == int(p > .5)
            acc = np.mean([x['correct'] for x in data])
            expected = (baseline[name][f'{split}_accuracy'] if name.startswith('B') else
                        result[split]['runs'][int(name[-2:]) - 42])
            assert np.isclose(acc, expected, atol=1e-12)
            records.append({'model': name, 'split': split, 'n': n, 'correct': sum(x['correct'] for x in data),
                            'accuracy': float(acc), 'source': str(path.relative_to(ROOT))})
            checks.append(str(path.relative_to(ROOT)))
            if split == 'test':
                test_rows[name] = data
    for split in ('val', 'test'):
        assert np.isclose(np.mean(result[split]['runs']), result[split]['mean'])
        assert np.isclose(np.std(result[split]['runs'], ddof=1), result[split]['std'])
    for seed_run in result['runs']:
        log = rows(run / 'logs' / f"dact_replication_seed{seed_run['seed']}.jsonl")
        epochs = [x for x in log if x['event'] == 'epoch']
        assert len(epochs) == seed_run['epochs']
        assert max(x['val_acc'] for x in epochs) == seed_run['val_acc']
        assert resolve_artifact(seed_run['ckpt']).exists()
        assert resolve_artifact(seed_run['ckpt'] + '.json').exists()
    best = max(result['runs'], key=lambda x: x['val_acc'])['seed']
    assert best == result['test']['best_val_seed']
    representative = test_rows[f'seed{best}']
    gold = np.array([x['gold_label'] for x in representative])
    pred = np.array([x['predicted_label'] for x in representative])
    assert np.allclose(bootstrap_ci(pred, gold), result['test']['ci95'])
    pairs = []
    # Exploratory family specified here: representative DACT vs local B0 and B1.
    # Same-item resampling preserves pairing; seed chosen only by validation.
    rng = np.random.default_rng(20260930)
    samples = rng.integers(0, len(gold), size=(10000, len(gold)))
    for name in ('B0', 'B1'):
        bpred = np.array([x['predicted_label'] for x in test_rows[name]])
        b, c, p = mcnemar_exact(pred, bpred, gold)
        difference = (pred == gold).astype(float) - (bpred == gold).astype(float)
        interval = np.quantile(difference[samples].mean(1), [.025, .975])
        pairs.append({'comparison': f'DACT seed {best} minus {name}', 'accuracy_difference_pp': difference.mean()*100,
                      'paired_bootstrap_low_pp': interval[0]*100, 'paired_bootstrap_high_pp': interval[1]*100,
                      'DACT_only_correct': b, 'baseline_only_correct': c, 'exact_McNemar_p': p})
    for record, adjusted in zip(pairs, holm_adjust([p['exact_McNemar_p'] for p in pairs])):
        record['Holm_p_two_comparisons'] = float(adjusted)
    tok = Tokenizer.from_file(str(run / 'tokenizer.json'))
    features = []
    for x in representative:
        ids = [tok.encode(x[f'candidate_{k}']).ids for k in (1, 2)]
        p = x['probability_candidate_2']
        features.append({**x, 'max_candidate_tokens': max(map(len, ids)),
                         'truncated': any(len(i) > 96 for i in ids),
                         'identical_truncated_tokens': ids[0][:96] == ids[1][:96],
                         'confidence': max(p, 1-p)})
    strata = []
    groups = {'all': features, 'truncated': [x for x in features if x['truncated']],
              'not truncated': [x for x in features if not x['truncated']],
              'identical retained option tokens': [x for x in features if x['identical_truncated_tokens']],
              'confidence >= .8': [x for x in features if x['confidence'] >= .8]}
    for label, group in groups.items():
        strata.append({'group': label, 'n': len(group), 'correct': sum(x['correct'] for x in group),
                       'accuracy': float(np.mean([x['correct'] for x in group])) if group else None})
    errors = [x for x in features if not x['correct']]
    sampled = sorted(np.random.default_rng(20260930).choice(len(errors), 12, replace=False))
    sample = [errors[i] for i in sampled]
    attention = []
    for path in sorted((run / 'figures').glob('*.png.json')):
        info = read(path)
        index = int(path.name.split('.')[0].split('_')[-1])
        item = representative[index]
        assert info['goal'] == item['goal'] and info['label'] == item['gold_label']
        assert info['pred'] == item['predicted_label']
        assert all(info[f'candidate_{k}'] == item[f'candidate_{k}'] for k in (1, 2))
        top = []
        for option in info['options']:
            weights = np.array(option['pool'])
            assert np.isclose(weights.sum(), 1, atol=1e-5)
            positions = option['sol_pos']
            top.append([{'token': option['tokens'][positions[j]], 'weight': float(weights[j]),
                         'difference_tag': option['diff'][j]} for j in np.argsort(-weights)[:5]])
        attention.append({'index': index, 'correct': item['correct'], 'top_pool_tokens_by_option': top,
                          'probability_delta_fp32_vs_saved_amp': abs(info['probs'][1]-item['probability_candidate_2'])})
    write_csv(out / 'replication_results.csv', records)
    write_csv(out / 'replication_comparisons.csv', pairs)
    write_csv(out / 'replication_error_strata.csv', strata)
    (out / 'replication_error_sample.json').write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding='utf8')
    (out / 'replication_attention_summary.json').write_text(json.dumps(attention, indent=2), encoding='utf8')
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for directory in (run, base) for p in sorted(directory.rglob('*')) if p.is_file()}
    (out / 'replication_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    summary = {'prediction_files_verified': checks, 'prediction_rows_verified': sum(x['n'] for x in records),
               'prediction_text_labels_match_supplied_data': True,
               'metrics_match': True, 'seed_aggregate_matches': True, 'validation_checkpoint_selection_matches': True,
               'representative_seed': best, 'bootstrap_CI_matches': True, 'attention_metadata_matches': True,
               'comparisons': pairs, 'error_strata': strata,
               'note': 'Exploratory post-run comparisons; local environment; no model selection from test.'}
    (out / 'replication_checks.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for seed in (42, 43, 44):
        ep = [x for x in rows(run/'logs'/f'dact_replication_seed{seed}.jsonl') if x['event']=='epoch']
        for ax, key in zip(axes, ('train_loss', 'val_loss', 'val_acc')):
            ax.plot([x['epoch'] for x in ep], [x[key] for x in ep], marker='.', label=f'seed {seed}')
            ax.set(xlabel='QA epoch', ylabel=key.replace('_', ' ')); ax.grid(alpha=.2)
    axes[-1].legend(); fig.suptitle('Local fixed-recipe replication (separate from historical A100 study)')
    fig.tight_layout(); fig.savefig(out/'figures/replication_training_curves.png', dpi=300); plt.close(fig)
    print(json.dumps(summary, indent=2))
    print('ERROR SAMPLE:', json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
