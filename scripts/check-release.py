"""Require the complete interpreter-specific release artifact matrix."""
from pathlib import Path
import sys
import os
import tarfile
import tomllib
from packaging.utils import parse_wheel_filename
from packaging.version import Version

root = Path(__file__).resolve().parents[1]
version = Version(tomllib.loads((root / 'Cargo.toml').read_text())['workspace']['package']['version'])
folder = Path(sys.argv[1])
if os.environ.get("GITHUB_REF", "").startswith("refs/tags/"):
    assert Version(os.environ["GITHUB_REF_NAME"].removeprefix("v")) == version, "Release tag and source version differ"
expected = {(f'cp3{minor}', arch) for minor in range(10,15) for arch in ['linux_x86_64','macos_x86_64','macos_arm64','windows_x86_64']}
found = set()
for wheel in folder.glob('*.whl'):
    name, built_version, build, tags = parse_wheel_filename(wheel.name)
    assert name == 'blaze2d' and built_version == version and not build, wheel.name
    for tag in tags:
        assert tag.interpreter == tag.abi, wheel.name
        platform = tag.platform
        if platform in ('manylinux2014_x86_64', 'manylinux_2_17_x86_64'): arch = 'linux_x86_64'
        elif platform == 'macosx_11_0_x86_64': arch = 'macos_x86_64'
        elif platform == 'macosx_11_0_arm64': arch = 'macos_arm64'
        elif platform == 'win_amd64': arch = 'windows_x86_64'
        else: raise AssertionError(f'Unexpected platform {platform}')
        found.add((tag.interpreter,arch))
assert len(list(folder.glob('*.whl'))) == 20 and found == expected, (found,expected-found)
sources = list(folder.glob('*.tar.gz'))
assert len(sources) == 1, sources
with tarfile.open(sources[0]) as archive:
    names = archive.getnames()
    assert any(name.endswith('/source-revision.txt') for name in names)
    assert any(name.endswith('/Cargo.lock') for name in names)
    assert not any(name.endswith(('.so','.pyd','.dylib')) for name in names)
print(f'{version}: 20 wheels and one source distribution verified.')
