"""Compact plots of measured pooling weights, without changing or renormalising them."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.1))
    for row, (kind, index, title) in enumerate([
        ('confident_correct', 1672, 'Success #1672: lemon cookie ice cream'),
        ('confident_wrong', 551, 'Failure #551: removing a stuck light bulb')]):
        path = ROOT/'experiments/completed/outputs_replication_efficient/figures'/f'{kind}_{index}.png.json'
        data = json.loads(path.read_text(encoding='utf8'))
        for k, option in enumerate(data['options']):
            ax = axes[row, k]
            pool = np.array(option['pool']); selected = np.argsort(-pool)[:6]
            labels = [f"{option['tokens'][option['sol_pos'][i]]} [{i+1}]" for i in selected]
            values = list(pool[selected])
            colors = ['#be3455' if option['diff'][i] else '#396ea0' for i in selected]
            labels.append('All other tokens (sum)'); values.append(1-sum(values)); colors.append('#b5bdc5')
            ax.barh(range(7), values, color=colors); ax.set_yticks(range(7), labels)
            ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel('Pooling weight (no renormalisation)')
            tags = ('; GOLD' if data['label']==k else '') + ('; PREDICTED' if data['pred']==k else '')
            ax.set_title(f"{title}\nOption {k+1}: p={data['probs'][k]:.3f}{tags}", fontsize=9)
            ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Local DACT seed 43: top six pooling tokens per option', fontsize=12)
    fig.text(.5, .015, 'Red: difference-tagged tokens. Blue: other tokens. Brackets: 1-based solution token position.\n'
             'Grey sums omitted tokens. Attention weights are descriptive, not causal explanations.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.075,1,.965))
    for suffix in ('png', 'svg'):
        fig.savefig(ROOT/'report_support/figures'/f'replication_attention_compact.{suffix}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
