"""F4.6: проверка чисел, SVG/HTML, целостности и строгой границы реального F4.5.

Синтетические величины в unit-тестах — исключительно фикстуры, не результаты Qwen.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from src.llm_f4.training_visualization import (
    Config, Point, VisualizationError, _decode_json, _number, _path,
    build_report, get_sft, load_config, parse_metrics, plan,
    render_files, validate_source, verify_report,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/f4_06_training_visualization.json'
CLI = ROOT / 'scripts/f4_training_visualization.py'


@pytest.fixture(scope='module')
def config():
    return load_config(CONFIG, root=ROOT)


@pytest.fixture(scope='module')
def sft(config):
    return get_sft(config, root=ROOT)


def fake_manifest(sft, profile='cpu'):
    """ИСКУССТВЕННЫЕ числа для unit-тестов, НЕ реальный результат модели."""
    p = sft.profiles[profile]
    return {
        'mode': profile, 'real_training_performed': True,
        'model_id': sft.model_id, 'model_revision': sft.model_revision,
        'config_sha256': sft.config_sha256,
        'optimizer_steps': p.max_steps, 'train_records_used': p.train_limit,
        'validation_records_used': p.validation_limit,
        'trainable_parameters': 540672, 'all_parameters_with_adapters': 494573440,
        'trainable_fraction': 540672 / 494573440,
        'initial_validation_loss': 3.0, 'final_validation_loss': 2.5,
        'evaluation_holdout_used_in_training': False, 'manual_quality_score': None,
    }


def fake_points(n=1):
    return tuple(Point(i, 3.0 - i*.1, .4 + i*.01, 1 if n == 1 else 4) for i in range(1, n+1))


def make_stats(sft, profile='cpu'):
    n = sft.profiles[profile].max_steps
    return validate_source(fake_manifest(sft, profile), fake_points(n), profile=profile, sft=sft)


def test_config_is_frozen(config):
    assert config.sft_config_file == ROOT / 'configs/f4_05_sft_lora.json'
    assert config.output_dir == ROOT / 'data/processed/f4_06'
    assert config.max_steps == 10000
    assert len(config.config_sha256) == 64


def test_plan_only_reads_upstream(config):
    result = plan(config, root=ROOT)
    assert result['splits'] == {'train': 56, 'validation': 12, 'test': 12}
    assert result['model_loading_performed'] is False
    assert isinstance(result['profiles']['cpu'], bool)


@pytest.mark.parametrize('bad', ['../secret', '/etc/passwd', 'C:/user/a', 'x\\y', 'a/../b', 'a/./b', 'a//b', '', 'C:\\data', 'x/'])
def test_unsafe_paths_rejected(bad):
    with pytest.raises(VisualizationError):
        _path(ROOT, bad, 'path')


@pytest.mark.parametrize('change', [
    {'schema_version': 2}, {'schema_version': True},
    {'sft_config_file': 'configs/other.json'},
    {'output_dir': 'data/processed/f4_05'},
    {'output_dir': 'data/processed/f4_05/inside'},
    {'output_dir': 'data/processed'},
    {'chart_width': 250}, {'chart_width': '920'},
    {'chart_height': 0}, {'chart_height': 2000},
    {'max_steps': 0}, {'max_steps': 10001},
])
def test_invalid_config(tmp_path, change):
    payload = json.loads(CONFIG.read_text(encoding='utf-8'))
    payload.update(change)
    path = tmp_path / 'broken.json'
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    with pytest.raises((VisualizationError, ValueError)):
        load_config(path, root=ROOT)


def test_config_extra_field_rejected(tmp_path):
    raw = json.loads(CONFIG.read_text(encoding='utf-8'))
    raw['secret_field'] = 'unknown'
    path = tmp_path / 'extra.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(VisualizationError):
        load_config(path, root=ROOT)


@pytest.mark.parametrize('payload', [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', b'{bad'])
def test_invalid_json(payload):
    with pytest.raises(VisualizationError):
        _decode_json(payload, 'test fixture')


def test_parse_metrics_one_step():
    raw = b'{"step":1,"train_loss":2.5,"grad_norm":0.375,"micro_batches":1}\n'
    rows = parse_metrics(raw, expected_steps=1, max_steps=24)
    assert rows == (Point(1, 2.5, .375, 1),)


def test_parse_metrics_twenty_four_steps():
    raw = b''.join(json.dumps({'step': i, 'train_loss': 3-i*.1, 'grad_norm': .25,
                               'micro_batches': 4}).encode() + b'\n' for i in range(1, 25))
    actual = parse_metrics(raw, expected_steps=24, max_steps=100)
    assert len(actual) == 24 and actual[-1].step == 24


@pytest.mark.parametrize('raw', [
    b'',
    b'{"step":1,"train_loss":1,"grad_norm":1,"micro_batches":1}',
    b'{}\n',
    b'{"step":0,"train_loss":1,"grad_norm":1,"micro_batches":1}\n',
    b'{"step":2,"train_loss":1,"grad_norm":1,"micro_batches":1}\n',
    b'{"step":1,"train_loss":-1,"grad_norm":1,"micro_batches":1}\n',
    b'{"step":1,"train_loss":1,"grad_norm":-2,"micro_batches":1}\n',
    b'{"step":1,"train_loss":1,"grad_norm":1,"micro_batches":0}\n',
    b'{"step":1,"train_loss":1,"grad_norm":1,"micro_batches":1,"learning_rate":1}\n',
    b'{"step":1,"train_loss":1,"grad_norm":NaN,"micro_batches":1}\n',
    b'{"step":1,"step":1,"train_loss":1,"grad_norm":1,"micro_batches":1}\n',
    b'{"step":true,"train_loss":1,"grad_norm":1,"micro_batches":1}\n',
    b'{"step":1,"train_loss":1,"grad_norm":1,"micro_batches":true}\n',
])
def test_invalid_metric_line(raw):
    with pytest.raises(VisualizationError):
        parse_metrics(raw, expected_steps=1, max_steps=20)


@pytest.mark.parametrize('bad', [0, 2, 100, True, -1])
def test_invalid_expected_length(bad):
    with pytest.raises(VisualizationError):
        parse_metrics(b'{"step":1,"train_loss":1,"grad_norm":1,"micro_batches":1}\n',
                      expected_steps=bad, max_steps=20)


def test_single_step_only_point_no_curve(config, sft):
    stats = make_stats(sft)
    chart = render_files(stats, fake_points(), config)['train_loss.svg'].decode('utf-8')
    assert '<circle ' in chart
    assert '<polyline' not in chart
    assert 'no trend' in chart
    assert stats['trend_interpretation'] == 'single_step_no_trend'


def test_gpu_has_actual_line_only_for_multiple_steps(config, sft):
    stats = make_stats(sft, profile='gpu')
    chart = render_files(stats, fake_points(24), config)['train_loss.svg'].decode('utf-8')
    assert '<polyline' in chart
    assert stats['trend_interpretation'] == 'trajectory_only_no_quality_claim'


def test_validation_math(sft):
    stats = make_stats(sft)
    assert stats['validation_loss_delta'] == pytest.approx(-.5)
    assert stats['validation_loss_relative_pct'] == pytest.approx(-100/6)
    assert stats['train_loss_step_mean'] == pytest.approx(2.9)
    assert stats['grad_norm_max_pre_clipping'] == pytest.approx(.41)


def test_validation_zero_denominator(sft):
    base = fake_manifest(sft)
    base['initial_validation_loss'] = 0
    stats = validate_source(base, fake_points(), profile='cpu', sft=sft)
    assert stats['validation_loss_relative_pct'] is None
    assert stats['validation_loss_delta'] == pytest.approx(2.5)


@pytest.mark.parametrize('change', [
    {'real_training_performed': False}, {'mode': 'gpu'},
    {'model_id': 'fake/model'}, {'model_revision': 'other'},
    {'config_sha256': '0'*64}, {'optimizer_steps': 12},
    {'train_records_used': 56}, {'validation_records_used': 12},
    {'trainable_parameters': 0}, {'all_parameters_with_adapters': 3},
    {'trainable_fraction': 0.5},
    {'initial_validation_loss': float('nan')},
    {'final_validation_loss': -1},
    {'manual_quality_score': 90},
    {'evaluation_holdout_used_in_training': True},
])
def test_bad_source_rejected(sft, change):
    m = fake_manifest(sft)
    m.update(change)
    with pytest.raises(VisualizationError):
        validate_source(m, fake_points(), profile='cpu', sft=sft)


def test_no_per_step_lr_disguised(config, sft):
    stats = make_stats(sft)
    assert stats['learning_rate_observed_per_step'] is False
    assert stats['learning_rate_scheduler_observed'] is False
    data = render_files(stats, fake_points(), config)
    lr_svg = data['learning_rate.svg'].decode('utf-8')
    assert 'NO per-step LR telemetry' in lr_svg
    assert '<polyline' not in lr_svg
    assert 'learning_rate_configured_not_observed' in data['metrics.csv'].decode()


def test_html_is_offline_and_disclaims_quality(config, sft):
    stats = make_stats(sft)
    doc = render_files(stats, fake_points(), config)['report.html'].decode('utf-8')
    assert 'Только один optimizer step' in doc
    assert 'F4.7' in doc
    assert 'не</strong> записывался' in doc
    assert '<script' not in doc and 'http://' not in doc and 'https://' not in doc
    assert 'train_loss.svg' in doc


def test_svg_is_well_formed(config, sft):
    stats = make_stats(sft)
    out = render_files(stats, fake_points(), config)
    for filename in ('train_loss.svg','grad_norm.svg','validation_loss.svg','learning_rate.svg'):
        root = ET.fromstring(out[filename])
        assert root.tag.endswith('svg') and 'title' in ''.join(x.tag for x in root)
        assert int(root.get('width')) == config.chart_width


def test_render_deterministic(config, sft):
    stats = make_stats(sft)
    assert render_files(stats, fake_points(), config) == render_files(stats, fake_points(), config)


def test_no_model_imports():
    import src.llm_f4.training_visualization as mod
    assert 'torch' not in mod.__dict__ and 'transformers' not in mod.__dict__
    assert 'peft' not in mod.__dict__ and 'matplotlib' not in mod.__dict__


def test_no_real_training_source_is_rejected(config, monkeypatch, tmp_path):
    patched_sft = replace(get_sft(config, root=ROOT), output_dir=tmp_path)
    from src.llm_f4 import training_visualization as mod
    # The source helper uses only actually verified F4.5 run; no fixture fallback.
    monkeypatch.setattr(mod, "get_sft", lambda *args, **kwargs: patched_sft)
    with pytest.raises(VisualizationError, match='F4.5 VERIFY'):
        mod._source(config, root=ROOT, profile='cpu')


def test_cli_help_and_plan():
    for args, expected in ((['--help'], '--build'), (['--plan'], 'F4.6 PLAN: PASS')):
        call = subprocess.run([sys.executable, str(CLI), *args], cwd=ROOT,
                              capture_output=True, text=True, check=True)
        assert expected in call.stdout


@pytest.mark.parametrize('encoding', ['cp1251', 'cp866', 'cp1252'])
def test_cli_windows_help(encoding):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = encoding
    call = subprocess.run([sys.executable, str(CLI), '--help'], cwd=ROOT,
                          capture_output=True, env=env, check=True)
    assert b'--verify' in call.stdout


def test_build_requires_real_source(config, monkeypatch):
    from src.llm_f4 import training_visualization as mod
    monkeypatch.setattr(mod, 'verify_sft_run', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('no real run')))
    with pytest.raises(RuntimeError, match='no real run'):
        build_report(config, root=ROOT, profile='cpu')


def test_output_path_no_rewrite(config, sft, monkeypatch, tmp_path):
    from src.llm_f4 import training_visualization as mod
    stats = make_stats(sft)
    monkeypatch.setattr(mod, '_source', lambda *args, **kwargs: (stats, fake_points(),
                         {'f45_run_manifest_sha256': 'a'*64, 'f45_metrics_sha256': 'b'*64, 'f46_config_sha256': config.config_sha256}))
    conf = replace(config, output_dir=tmp_path)
    first = build_report(conf, root=ROOT, profile='cpu')
    assert first['optimizer_steps'] == 1
    assert (tmp_path/'cpu'/'report.html').is_file()
    with pytest.raises(VisualizationError, match='уже существует'):
        build_report(conf, root=ROOT, profile='cpu')
    assert verify_report(conf, root=ROOT, profile='cpu')['optimizer_steps'] == 1
    (tmp_path/'cpu'/'train_loss.svg').write_bytes(b'corrupt')
    with pytest.raises(VisualizationError, match='Изменён'):
        verify_report(conf, root=ROOT, profile='cpu')


def test_verify_rejects_missing_or_extra(config, sft, monkeypatch, tmp_path):
    from src.llm_f4 import training_visualization as mod
    stats = make_stats(sft)
    monkeypatch.setattr(mod, '_source', lambda *args, **kwargs: (stats, fake_points(),
                         {'f45_run_manifest_sha256': 'a'*64, 'f45_metrics_sha256': 'b'*64, 'f46_config_sha256': config.config_sha256}))
    conf = replace(config, output_dir=tmp_path)
    build_report(conf, root=ROOT, profile='cpu')
    (tmp_path/'cpu'/'alien.txt').write_text('oops', encoding='utf-8')
    with pytest.raises(VisualizationError, match='посторонние'):
        verify_report(conf, root=ROOT, profile='cpu')


def test_source_rejects_tampered_metrics_after_verify(config, monkeypatch, tmp_path):
    from src.llm_f4 import training_visualization as mod
    sft = get_sft(config, root=ROOT)
    fake = fake_manifest(sft)
    fake['metrics_sha256'] = sha256(b'original').hexdigest()
    folder = tmp_path / 'cpu'
    folder.mkdir()
    (folder/'metrics.jsonl').write_bytes(b'altered')
    (folder/'run_manifest.json').write_bytes(json.dumps(fake).encode('utf-8'))
    monkeypatch.setattr(mod, 'get_sft', lambda *args, **kwargs: replace(sft, output_dir=tmp_path))
    monkeypatch.setattr(mod, 'verify_sft_run', lambda *args, **kwargs: fake)
    with pytest.raises(VisualizationError, match='изменились'):
        mod._source(config, root=ROOT, profile='cpu')


def test_configured_lr_from_upstream(sft):
    stats = make_stats(sft)
    assert stats['learning_rate_configured'] == pytest.approx(.0002)
    assert not hasattr(fake_points()[0], "learning_rate")


def test_f45_verifier_integration_with_isolated_fixture(config, monkeypatch, tmp_path):
    """Настоящий F4.5 verify_run + F4.6 _source на файлах фикстуры.

    Не создаёт и не утверждает подлинные веса модели: только тестирует контракт
    чтения/хеширования. Фактический inference/training здесь НЕ запускается.
    """
    from src.llm_f4 import training_visualization as mod
    from src.llm_f4.sft_lora import make_plan

    original = get_sft(config, root=ROOT)
    staging = tmp_path/'f45'
    sft = replace(original, output_dir=staging)
    p, _ = make_plan(original, root=ROOT)
    dst = staging/'cpu'
    adapter = dst/'adapter'
    adapter.mkdir(parents=True)
    (adapter/'adapter_config.json').write_bytes(b'{"test_fixture":true}')
    (adapter/'adapter_model.safetensors').write_bytes(b'not a real adapter')
    raw = b'{"step":1,"train_loss":2.4,"grad_norm":0.5,"micro_batches":1}\n'
    (dst/'metrics.jsonl').write_bytes(raw)
    record = fake_manifest(original)
    record.update({
        'train_sha256': p['train_sha256'],
        'validation_sha256': p['validation_sha256'],
        'holdout_sha256': p['holdout_sha256'],
        'metrics_sha256': sha256(raw).hexdigest(),
        'adapter_files_sha256': {f.name: sha256(f.read_bytes()).hexdigest() for f in adapter.iterdir()},
    })
    (dst/'run_manifest.json').write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    monkeypatch.setattr(mod, 'get_sft', lambda *args, **kwargs: sft)
    # НЕ подменяем verify_sft_run: он обязан проверить файлы реально.
    summary, rows, fingerprints = mod._source(config, root=ROOT, profile='cpu')
    assert rows == (Point(1, 2.4, .5, 1),)
    assert summary['optimizer_steps'] == 1
    assert fingerprints['f45_metrics_sha256'] == sha256(raw).hexdigest()
    report_cfg = replace(config, output_dir=tmp_path/'f46')
    assert build_report(report_cfg, root=ROOT, profile='cpu')['optimizer_steps'] == 1
    assert verify_report(report_cfg, root=ROOT, profile='cpu')['optimizer_steps'] == 1
    # После изменения F4.5 логов ни build, ни verify не должны подтверждаться.
    (dst/'metrics.jsonl').write_bytes(b'changed')
    with pytest.raises(VisualizationError, match='F4.5 VERIFY'):
        verify_report(report_cfg, root=ROOT, profile='cpu')
