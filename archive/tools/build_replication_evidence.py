"""Execute a notebook verifying saved replication artifacts; never impersonate training execution."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT/'evidence/CITS4012_69_replication_verification.ipynb'
    if target.exists():
        raise FileExistsError('Preserve the executed verification notebook; choose a new name for another run.')
    nb = nbformat.v4.new_notebook()
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    intro = '''# Group 69 — actual local replication artifact verification

Executed on 30 September 2026. This notebook reads and verifies artifacts produced by the completed
three-seed training run; it does **not** claim that the training was executed in these cells. The
training implementation is in `src/` and `tools/reproduce_main.py`, its predeclared configuration,
logs, checkpoints and predictions are in `experiments/completed/outputs_replication_efficient/`. B0/B1 local evidence is
in `experiments/completed/outputs_replication_baselines/`. Run from the repository directory with supplied data at `../../PIQA`.

This is separate from the original A100 study in `CITS4012_69.ipynb`. It provides genuine executed
metric-recomputation, paired-comparison, error-analysis and figure outputs for the local replication.
The complete revised baseline/ablation notebook still needs a full fresh Colab execution before promotion.
'''
    log_code = '''from pathlib import Path
import json
root = Path.cwd()
print("SAVED PRE-TRAINING PROTOCOL")
print((root/'experiments/completed/outputs_replication_efficient/protocol.json').read_text(encoding='utf8'))
for path in sorted((root/'experiments/completed/outputs_replication_efficient/logs').glob('*.jsonl')):
    print("SAVED TRAINING LOG:", path.name)
    print(path.read_text(encoding='utf8'))
'''
    code = (ROOT/'tools/analyse_replication.py').read_text(encoding='utf8')
    code = code.replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
    figure_code = '''from IPython.display import Image, display
for name in ('replication_training_curves.png', 'replication_attention_compact.png'):
    display(Image(filename=str(ROOT/'report_support/figures'/name)))
'''
    nb.cells = [nbformat.v4.new_markdown_cell(intro), nbformat.v4.new_code_cell(log_code),
                nbformat.v4.new_code_cell(code), nbformat.v4.new_code_cell(figure_code),
                nbformat.v4.new_markdown_cell((ROOT/'report_support/replication.md').read_text(encoding='utf8'))]
    NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    nbformat.validate(nb)
    nbformat.write(nb, target)
    print('Executed artifact verification notebook:', target)


if __name__ == '__main__':
    main()
