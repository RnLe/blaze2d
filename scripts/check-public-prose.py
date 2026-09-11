"""Reject prohibited dash forms in authored, published text."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCOPES = [ROOT / 'README.md', ROOT / 'crates/python/README.md', ROOT / 'web/content',
          ROOT / 'web/components', ROOT / 'web/app', ROOT / 'web/lib', ROOT / 'paper', ROOT / 'typst']
PATTERN = re.compile(r'\u2014|&mdash;|&#0*8212;|&#x0*2014;|\\textemdash', re.IGNORECASE)
failures = []
for scope in SCOPES:
    for path in ([scope] if scope.is_file() else scope.rglob('*')):
        if path.suffix not in {'.md', '.mdx', '.tsx', '.ts', '.tex', '.typ'}:
            continue
        if any(part in {'build', 'out', 'node_modules', '.next'} for part in path.parts):
            continue
        for line, text in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            tex_dash = path.suffix == '.tex' and '---' in text.split('%', 1)[0]
            if PATTERN.search(text) or tex_dash:
                failures.append(f'{path.relative_to(ROOT)}:{line}: replace the em dash')
if failures:
    raise SystemExit('\n'.join(failures))
print('Published prose punctuation checked.')
