"""List all changed/new project files and separately retained local experiment artifacts."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_paths(*args):
    result = subprocess.check_output(['git', *args, '-z'], cwd=ROOT).decode('utf8')
    return set(filter(None, result.split('\0')))


def main():
    changed = git_paths('diff', '--name-only')
    new = git_paths('ls-files', '--others', '--exclude-standard')
    target = ROOT/'report_support/files_changed.md'
    new.add(target.relative_to(ROOT).as_posix())
    lines = ['# Individual file inventory', '',
             'Generated from Git working-tree changes and untracked files. Original `outputs/` files',
             'are unchanged. The environment, caches and temporary renders are excluded. No push was made.', '',
             '| Status | File |', '|---|---|']
    for name in sorted(changed | new):
        lines.append(f"| {('Modified' if (ROOT/name).exists() else 'Moved from original location') if name in changed else 'New'} | `{name}` |")
    lines += ['', '## Completed experiment artifacts (Git-ignored, retained locally)', '',
              'These files are hashed in `replication_manifest.json`; preserve them when backing up the project.', '',
              '| File | Bytes |', '|---|---:|']
    for directory in ('experiments/completed/outputs_replication_efficient', 'experiments/completed/outputs_replication_baselines'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file():
                lines.append(f'| `{path.relative_to(ROOT).as_posix()}` | {path.stat().st_size} |')
    lines += ['', '## Development-only artifacts (not benchmark results)', '',
              '`experiments/development/outputs_replication/` contains the interrupted initial attempt; `experiments/development/outputs_verification/`',
              'contains small functional-test artifacts. Timestamped `experiments/development/outputs_reproduction_*/`',
              'directories hold development smoke outputs. These remain locally',
              'available; the genuinely executed smoke notebooks are listed above. `.venv/` and `tmp/`',
              'are tooling/scratch directories and are not submission content.', '']
    target.write_text('\n'.join(lines), encoding='utf8')
    print(f'{len(changed)} modified and {len(new)} new project files individually listed, plus completed run artifacts.')


if __name__=='__main__':
    main()
