"""Record the source revision for builds from an unpacked source distribution."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
if not re.fullmatch(r'[0-9a-f]{40}', revision):
    raise SystemExit('Expected a complete Git source revision')
(root / 'crates/interface/source-revision.txt').write_text(revision + '\n', encoding='ascii')
