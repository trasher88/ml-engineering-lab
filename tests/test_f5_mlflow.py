"""F5.7 — реальные unit invariants и отдельная SQLite-интеграция MLflow.

Все измерения в тестовом каталоге синтетические. В рабочие отчёты не попадают.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation_f5.unified_report import build_report, serialize_json
from src.tracking_f5.mlflow_tracking import (
    CONFIG, TrackingError, build_plan, canonical, digest, load_json,
    safe_path, sync_runs, tracking_storage, _find_prior, _readback,
)

BASE = Path(__file__).resolve().parents[1]


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """Реальный контракт конфигурации F5.1 + искусственные JSON измерений."""
    root = tmp_path / 'repo'
    for name in ('configs/f5_01_unified_evaluation.json', CONFIG):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((BASE / name).read_bytes())
    sources = {
        'reports/classical.json': {'test': {'accuracy': 0.8, 'macro_f1': 0.75, 'macro_precision': 0.76, 'macro_recall': 0.74}},
        'outputs/f2_6_diagnostics/metrics.json': {
            'format_version': 1, 'best_val_macro_f1': 0.69, 'best_epoch': 5, 'epochs': 5,
            'history': [{'epoch': 5, 'val_accuracy': 0.70, 'val_macro_f1': 0.69}]},
        'reports/public/f3_07/F3_07_comparison.json': {
            'metrics': {'bm25': {'5': {'recall_at_k': 0.4, 'mrr_at_k': 0.3}},
                        'vector': {'5': {'recall_at_k': 0.6, 'mrr_at_k': 0.45}},
                        'hybrid': {'5': {'recall_at_k': 0.5, 'mrr_at_k': 0.35}}}},
        'data/processed/f4_06/cpu/summary.json': {
            'optimizer_steps': 1, 'train_loss_last': 2.39,
            'validation_loss_before': 2.28, 'validation_loss_after': 2.27,
            'trainable_parameters': 540672, 'learning_rate_configured': 0.0002},
    }
    for path, content in sources.items():
        _write(root / path, content)
    _write_report(root)
    return root


def _write_report(root: Path):
    expected = build_report(root)
    assert len([s for s in expected['sources'] if s['metrics']]) == 4
    path = root / 'reports/public/f5_01/F5_01_unified_evaluation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(expected), encoding='utf-8')


class MemoryClient:
    """Контрактный mock для проверки ветвления; НЕ заменяет MLflow integration."""
    def __init__(self):
        self.experiment = None
        self.runs = {}
        self.artifacts = {}
        self.seq = 0

    def get_experiment_by_name(self, name):
        return self.experiment if self.experiment and self.experiment.name == name else None

    def create_experiment(self, name, artifact_location):
        self.experiment = SimpleNamespace(name=name, experiment_id='1', lifecycle_stage='active', artifact_location=artifact_location)
        return '1'

    def search_runs(self, experiment_ids, max_results=1000):
        return list(self.runs.values())

    def create_run(self, experiment_id, tags, run_name):
        self.seq += 1
        run_id = f'{self.seq:032x}'
        run = SimpleNamespace(info=SimpleNamespace(run_id=run_id, status='RUNNING'),
                              data=SimpleNamespace(tags=dict(tags), params={}, metrics={}))
        self.runs[run_id] = run
        return run

    def log_param(self, run_id, key, value):
        self.runs[run_id].data.params[key] = str(value)

    def log_metric(self, run_id, key, value):
        self.runs[run_id].data.metrics[key] = value

    def log_artifact(self, run_id, local_path):
        self.artifacts[run_id] = Path(local_path).read_bytes()

    def set_terminated(self, run_id, status):
        self.runs[run_id].info.status = status

    def get_run(self, run_id):
        return self.runs[run_id]

    def download_artifacts(self, run_id, artifact_path, dst_path):
        dest = Path(dst_path) / artifact_path
        dest.write_bytes(self.artifacts[run_id])
        return str(dest)


def test_plan_six_real_slots_from_synthetic_files(repo):
    plan = build_plan(repo)
    assert plan.status == 'PASS'
    assert len(plan.records) == plan.expected == 6
    assert [x.name for x in plan.records] == ['f1_classical','f2_pytorch','f3_bm25','f3_vector','f3_hybrid','f4_lora_sft']


def test_plan_does_not_log_cross_task_score(repo):
    plan = build_plan(repo)
    assert not any('score' in x.metrics for x in plan.records)
    assert plan.records[2].metrics == {'recall_at_5': 0.4, 'mrr_at_5': 0.3}
    assert plan.records[0].metrics['macro_f1'] == 0.75
    assert plan.records[1].metrics['macro_f1'] == 0.69
    assert plan.records[1].split == 'validation'
    assert plan.records[2].selectors['recall_at_5'] == 'metrics.bm25.5.recall_at_k'


def test_units_and_provenance(repo):
    row = build_plan(repo).records[5]
    assert row.task == 'llm' and row.units['validation_loss_after'] == 'loss'
    assert row.metrics['optimizer_steps'] == 1.0
    assert row.source_sha256 == digest((repo/row.source_path).read_bytes())
    assert row.summary()['claim'] == 'imported_observations_not_new_training_or_model_evaluation'


def test_no_source_text_is_copied(repo):
    row = build_plan(repo).records[0]
    assert 'reference' not in row.summary()
    assert 'questions' not in row.summary()
    assert 'source_sha256' in row.summary()


def test_plan_stale_source_detected(repo):
    path = repo / 'reports/classical.json'
    _write(path, {'test': {'accuracy': 0.6, 'macro_f1': 0.6}})
    with pytest.raises(TrackingError, match='устарел'):
        build_plan(repo)


def test_plan_stale_report_tampering_detected(repo):
    path = repo / 'reports/public/f5_01/F5_01_unified_evaluation.json'
    value = load_json(path)
    value['status'] = 'FAKE'
    _write(path, value)
    with pytest.raises(TrackingError, match='устарел'):
        build_plan(repo)


def test_plan_missing_report_is_blocker(repo):
    (repo / 'reports/public/f5_01/F5_01_unified_evaluation.json').unlink()
    with pytest.raises(TrackingError, match='Нет отчёта'):
        build_plan(repo)


def test_no_duplicate_run_id(repo):
    path = repo / CONFIG
    cfg = load_json(path)
    cfg['runs'].append(deepcopy(cfg['runs'][0]))
    _write(path, cfg)
    with pytest.raises(TrackingError, match='Дублирован run id'):
        build_plan(repo)


def test_invalid_required_metrics(repo):
    path = repo / CONFIG
    cfg = load_json(path)
    cfg['runs'][0]['required_metrics'] = ['macro_f1','macro_f1']
    _write(path, cfg)
    with pytest.raises(TrackingError, match='required_metrics'):
        build_plan(repo)


def test_missing_optional_metric_still_logs_factual_values(repo):
    path = repo / 'reports/classical.json'
    _write(path, {'test': {'macro_f1': 0.65}})
    _write_report_partial(repo)
    plan = build_plan(repo)
    record = plan.records[0]
    assert record.metrics == {'macro_f1': 0.65}


def _write_report_partial(root):
    report = build_report(root)
    (root/'reports/public/f5_01/F5_01_unified_evaluation.json').write_text(serialize_json(report), encoding='utf-8')


def test_missing_required_is_skipped_not_invented(repo):
    path = repo / 'reports/classical.json'
    _write(path, {'test': {'accuracy': 0.7}})
    _write_report_partial(repo)
    plan = build_plan(repo)
    assert plan.status == 'PARTIAL'
    assert len(plan.records) == 5
    assert plan.skipped[0]['name'] == 'f1_classical'


def test_missing_retrieval_source_skips_three(repo):
    (repo/'reports/public/f3_07/F3_07_comparison.json').unlink()
    _write_report_partial(repo)
    plan = build_plan(repo)
    assert len(plan.records) == 3
    assert [r['name'] for r in plan.skipped] == ['f3_bm25', 'f3_vector', 'f3_hybrid']


def test_none_available_fails_without_creating_anything(repo):
    for path in ('reports/classical.json','outputs/f2_6_diagnostics/metrics.json',
                 'reports/public/f3_07/F3_07_comparison.json','data/processed/f4_06/cpu/summary.json'):
        (repo/path).unlink()
    _write_report_partial(repo)
    plan = build_plan(repo)
    assert not plan.records and plan.status == 'PARTIAL'
    with pytest.raises(TrackingError, match='Ни один'):
        sync_runs(MemoryClient(), plan, 'file:///tmp/artifacts', create=True)


def test_safe_path_rejects_traversal(repo):
    for val in ('../outside','a/../../outside','C:/Users/abc','/etc/passwd','bad\\path','a//b','a/./b'):
        with pytest.raises(TrackingError):
            safe_path(repo,val)


def test_safe_path_symlink_escape(repo, tmp_path):
    outside = tmp_path / 'secret.json'
    outside.write_text('{}')
    symlink = repo / 'escape.json'
    try:
        symlink.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip('Windows не разрешает создавать symlink')
    with pytest.raises(TrackingError):
        safe_path(repo, 'escape.json')


def test_invalid_yaml_not_accepted(repo):
    path = repo / CONFIG
    path.write_text('oops: true',encoding='utf-8')
    with pytest.raises(TrackingError):
        build_plan(repo)


def test_nonfinite_json_is_rejected(repo):
    path = repo / 'reports/classical.json'
    path.write_text('{"test": {"macro_f1": NaN}}',encoding='utf-8')
    with pytest.raises(ValueError):
        build_plan(repo)


def test_canonical_json_is_deterministic():
    assert canonical({'b': 2, 'a': 1}) == canonical({'a':1, 'b':2})
    assert digest(canonical({'a':1})) == hashlib.sha256(canonical({'a':1})).hexdigest()


def test_identical_plan_is_stable(repo):
    assert build_plan(repo) == build_plan(repo)


def test_report_file_changes_do_not_create_fake_results(repo):
    plan = build_plan(repo)
    changed = load_json(repo/CONFIG)
    changed['experiment_name'] = 'separate-import'
    _write(repo/CONFIG, changed)
    new_plan = build_plan(repo)
    assert new_plan.records[0].signature != plan.records[0].signature


def test_storage_is_sqlite_and_local(tmp_path):
    """SQLite URL для Windows и POSIX отличается количеством / после sqlite:."""
    storage = (tmp_path / 'tracking').resolve()
    uri, art = tracking_storage(storage)
    database = storage / 'tracking.db'
    artifacts = storage / 'artifacts'

    # Windows: sqlite:///C:/...; POSIX: sqlite:////... .
    # Проверять полный абсолютный путь, а не количество начальных слешей.
    assert uri == 'sqlite:///' + database.as_posix()
    assert art == artifacts.as_uri()
    assert artifacts.is_dir()


def test_mock_import_and_idempotence(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    result = sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    assert result['verified_runs'] == 6 and result['status'] == 'PASS'
    assert {item['action'] for item in result['runs']} == {'CREATED'}
    second = sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    assert {item['action'] for item in second['runs']} == {'REUSED'}
    assert client.seq == 6
    verified = sync_runs(client, plan, 'file:///tmp/allowed', create=False)
    assert len(verified['runs']) == 6


def test_mock_verify_requires_prior_run(repo):
    with pytest.raises(TrackingError,match='Сначала'):
        sync_runs(MemoryClient(), build_plan(repo), 'file:///tmp/allowed', create=False)


def test_mock_mismatch_metric_blocks_reuse(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    client.runs['00000000000000000000000000000001'].data.metrics['macro_f1'] = 0.1
    with pytest.raises(TrackingError, match='не соответствует'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=True)


def test_mock_missing_metric_blocks_reuse(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    client.runs['00000000000000000000000000000001'].data.metrics.pop('macro_f1')
    with pytest.raises(TrackingError, match='не соответствует'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=False)


def test_mock_extra_metric_blocks_reuse(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    client.runs['00000000000000000000000000000001'].data.metrics['invented'] = 0.9
    with pytest.raises(TrackingError, match='неучтённые'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=False)


def test_mock_corrupt_artifact_fails(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    client.artifacts['00000000000000000000000000000001'] = b'{"fake":true}'
    with pytest.raises(TrackingError, match='artifact'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=False)


def test_mock_invalid_status_fails(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    client.runs['00000000000000000000000000000001'].info.status = 'FAILED'
    with pytest.raises(TrackingError, match='не завершён'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=False)


def test_mock_missing_param_fails(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    client.runs['00000000000000000000000000000001'].data.params.pop('dataset')
    with pytest.raises(TrackingError, match='param'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=False)


def test_mock_duplicate_detection(repo):
    client = MemoryClient()
    plan = build_plan(repo)
    sync_runs(client, plan, 'file:///tmp/allowed', create=True)
    a = client.runs['00000000000000000000000000000001']
    client.runs['x'] = deepcopy(a)
    with pytest.raises(TrackingError, match='дубли'):
        sync_runs(client, plan, 'file:///tmp/allowed', create=True)


def test_mock_new_data_creates_new_run(repo):
    client = MemoryClient()
    original = build_plan(repo)
    sync_runs(client, original, 'file:///tmp/allowed', create=True)
    path = repo/'reports/classical.json'
    current = load_json(path)
    current['test']['macro_f1'] = 0.77
    _write(path,current)
    _write_report(repo)
    changed = build_plan(repo)
    result = sync_runs(client, changed, 'file:///tmp/allowed', create=True)
    assert len([r for r in result['runs'] if r['action']=='CREATED']) == 1
    assert client.seq == 7


def test_real_mlflow_sqlite_roundtrip(repo, tmp_path):
    """При установленном mlflow — настоящие SQLite, artifact store, readback и idempotency."""
    if os.environ.get('F5_07_REQUIRE_MLFLOW') == '1':
        import mlflow  # noqa: F401; обязательная реальная интеграция CI
    else:
        pytest.importorskip('mlflow', reason='MLflow недоступен локально; в GitHub Actions устанавливается отдельно')
    from mlflow.tracking import MlflowClient
    uri, artifacts = tracking_storage(tmp_path/'mlflow')
    client = MlflowClient(tracking_uri=uri)
    plan = build_plan(repo)
    first = sync_runs(client, plan, artifacts, create=True)
    assert first['verified_runs'] == 6
    assert all(x['action']=='CREATED' for x in first['runs'])
    assert (tmp_path/'mlflow'/'tracking.db').exists()
    second = sync_runs(client, plan, artifacts, create=True)
    assert all(x['action']=='REUSED' for x in second['runs'])
    assert sync_runs(client, plan, artifacts, create=False)['verified_runs'] == 6
    experiment = client.get_experiment_by_name(plan.experiment)
    assert len(client.search_runs([experiment.experiment_id])) == 6
