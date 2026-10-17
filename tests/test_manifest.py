"""manifest.json agrees with the code: voices, languages, settings, version, model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from voxcpm_addon import __version__, engine, voices
from voxcpm_addon import __main__ as addon

REPO = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO / 'manifest.json').read_text(encoding='utf-8'))
FIELDS = {field['key']: field for field in MANIFEST['config_schema']['fields']}


def test_identity_and_version():
    assert MANIFEST['id'] == 'voxcpm'
    assert MANIFEST['executable'] == 'bin/voxcpm-addon'
    assert MANIFEST['tasks'] == ['tts.synthesize']
    pyproject = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    assert MANIFEST['version'] == __version__ == addon.ADDON_VERSION


def test_voices_and_languages_match_the_code():
    assert MANIFEST['voices'] == voices.manifest_voices()
    assert MANIFEST['languages'] == list(voices.LANGUAGES)
    clone = [v for v in MANIFEST['voices'] if v.get('requires')]
    assert [v['id'] for v in clone] == [voices.CLONE]
    for voice in voices.VOICES:
        if voice.id not in (voices.CLONE, voices.DESCRIBED):
            assert voice.description, voice.id


def test_anchor_sentences_are_for_languages_the_model_speaks():
    assert set(voices.ANCHORS) <= set(voices.LANGUAGES)
    assert all(sentence.strip() for sentence in voices.ANCHORS.values())
    assert voices.anchor_for('pt-BR') == ('pt', voices.ANCHORS['pt'], True)
    assert voices.anchor_for('th') == ('en', voices.ANCHORS['en'], False)
    assert voices.anchor_for('') == ('en', voices.ANCHORS['en'], False)


def test_every_setting_the_code_reads_is_declared():
    read = set(addon._options())
    assert read <= set(FIELDS)
    assert 'skip_ref_text' in FIELDS  # read by the host, not the add-on
    assert FIELDS['skip_ref_text']['type'] == 'bool'


def test_setting_defaults_are_valid():
    for key, field in FIELDS.items():
        if field['type'] == 'select':
            assert field['default'] in [o['value'] for o in field['options']], key
        if field['type'] in ('int', 'number'):
            assert field['min'] <= field['default'] <= field['max'], key


def test_the_code_clamps_numbers_to_the_declared_range(monkeypatch):
    for key in ('steps', 'cfg', 'threads'):
        field = FIELDS[key]
        monkeypatch.setenv(f'VOXCPM_{key.upper()}', str(field['max'] * 10 + 1))
        assert addon._options()[key] == field['max'], key
        monkeypatch.setenv(f'VOXCPM_{key.upper()}', str(field['min'] - 1))
        assert addon._options()[key] == field['min'], key


def test_the_model_entry_is_the_pinned_revision():
    model = MANIFEST['models'][0]
    assert model['repo'] == engine.MODEL_REPO
    assert model['revision'] == engine.MODEL_REVISION
    assert model['size_mb'] == round(engine.MODEL_BYTES / 2**20)
    assert model['bundled'] is False
    assert MANIFEST['disk_required_mb'] > model['size_mb']


def test_precision_choices():
    assert engine.resolve_precision('auto', 'cpu') == ('bfloat16', True)
    assert engine.resolve_precision('bfloat16', 'cpu') == ('bfloat16', False)
    assert engine.resolve_precision('float32', 'cpu') == ('float32', False)
    assert engine.resolve_precision('auto', 'mps') == ('float32', False)
    assert engine.resolve_precision('bfloat16', 'mps') == ('float32', False)
    assert {o['value'] for o in FIELDS['precision']['options']} == {'auto', 'bfloat16', 'float32'}


def test_cpu_lists_parse():
    assert engine._cpu_list('0-3,8,10-11\n') == [0, 1, 2, 3, 8, 10, 11]
