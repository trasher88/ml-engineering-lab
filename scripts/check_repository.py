# Назначение файла: P0: проверка целостности публичной структуры проекта и ссылок в документах.
# Не выполняет обучение моделей и проверку полного набора экспериментов.
# Примечание: комментарии не влияют на работу программы.

"""Validate repository metadata and published links using Python stdlib only."""
from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

# Проверяет наличие обязательных файлов и внутреннюю связность документов.
def check() -> None:
    required = ['README.md', '.gitignore', '.gitattributes',
                'docs/visuals/architecture.svg', 'docs/DATA_PROVENANCE.md',
                'docs/GITHUB_PUBLICATION_PLAN.md', 'data/raw/manifest.json',
                'models/external_manifest.json']
    for name in required:
        assert (ROOT / name).is_file(), f'No published file: {name}'
    tree = ET.parse(ROOT / 'docs/visuals/architecture.svg')
    assert tree.getroot().tag.endswith('svg')
    raw = json.loads((ROOT / 'data/raw/manifest.json').read_text(encoding='utf-8'))
    assert len(raw['revision']) == 40 and re.fullmatch(r'[0-9a-f]{40}', raw['revision'])
    assert len(raw['sources']) == 4
    for source in raw['sources']:
        assert re.fullmatch(r'[0-9a-f]{64}', source['sha256'])
        assert source['url'].startswith('https://')
    models = json.loads((ROOT / 'models/external_manifest.json').read_text(encoding='utf-8'))
    assert len(models) >= 2
    for name, item in models.items():
        assert re.fullmatch(r'[0-9a-f]{40}', item['revision']), name
    for file in [ROOT / 'README.md', ROOT / 'docs/GITHUB_PUBLICATION_PLAN.md',
                 ROOT / 'docs/DATA_PROVENANCE.md']:
        doc = file.read_text(encoding='utf-8')
        for link in re.findall(r'(?<!\!)\]\(([^)]+)\)', doc):
            if link.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            target = link.partition('#')[0]
            assert (file.parent / target).is_file(), f'{file.relative_to(ROOT)}: broken link {link}'
    print('Repository metadata check: PASS (no model training or runtime tests)')

if __name__ == '__main__':
    try: check()
    except (AssertionError, KeyError, ValueError, OSError, ET.ParseError) as error:
        print(f'Repository metadata check: FAIL: {error}', file=sys.stderr)
        sys.exit(1)
