"""Non-training final consistency checks; retains a machine-readable result."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import nbformat
from project_paths import resolve_artifact

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/'evidence/historical/sha256.json').read_text(encoding='utf8'))
    archived, outputs = 0, 0
    for name, digest in manifest.items():
        path = ROOT/'evidence/historical'/name
        if path.is_file():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
            archived += 1
        if name.startswith('outputs/'):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
            outputs += 1
    old = nbformat.read(ROOT/'evidence/historical/CITS4012_69.ipynb', as_version=4)
    current = nbformat.read(ROOT/'CITS4012_69.ipynb', as_version=4)
    old_code = [c for c in old.cells if c.cell_type=='code']
    current_code = [c for c in current.cells if c.cell_type=='code' and not c.metadata.get('audit_appendix')]
    assert len(old_code)==len(current_code)==27
    assert all(a.source==b.source and a.outputs==b.outputs for a,b in zip(old_code,current_code))
    notebooks = []
    for path in [ROOT/'CITS4012_69.ipynb', ROOT/'notebooks/CITS4012_69_reproducible.ipynb',
                 *sorted((ROOT/'evidence').rglob('*.ipynb'))]:
        if 'historical' in path.parts:
            continue
        nb = nbformat.read(path, as_version=4); nbformat.validate(nb)
        errors = [o for c in nb.cells if c.cell_type=='code' for o in c.outputs if o.output_type=='error']
        assert not errors, path
        notebooks.append(str(path.relative_to(ROOT)))
    generated = nbformat.read(ROOT/'notebooks/CITS4012_69_reproducible.ipynb', as_version=4)
    code = '\n'.join(c.source for c in generated.cells if c.cell_type=='code')
    sys.path.insert(0,str(ROOT/'tools'))
    from build_notebook import module
    for path in (ROOT/'src').glob('*.py'):
        source = module(path.stem)
        assert source.strip() in code, path
    smoke = nbformat.read(max((ROOT/'evidence').glob('CITS4012_69_development_smoke_*.ipynb')), as_version=4)
    assert [c.source for c in smoke.cells if c.cell_type=='code']==[c.source for c in generated.cells if c.cell_type=='code']
    local_manifest = json.loads((ROOT/'report_support/replication_manifest.json').read_text(encoding='utf8'))
    for name, digest in local_manifest.items():
        assert hashlib.sha256(resolve_artifact(name).read_bytes()).hexdigest()==digest, name
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    checks = {'original_archived_files_verified':archived,'original_output_files_unchanged':outputs,
              'original_code_output_pairs_preserved':27,'notebooks_validated_without_errors':notebooks,
              'generated_modules_match':True,'fresh_kernel_smoke_matches_current_code':True,
              'replication_artifact_hashes_verified':len(local_manifest),'git_diff_check':'PASS',
              'full_revised_Colab_and_pretrained_paths':'NOT EXECUTED',
              'final_report_pdf':'NOT CREATED'}
    (ROOT/'report_support/final_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
