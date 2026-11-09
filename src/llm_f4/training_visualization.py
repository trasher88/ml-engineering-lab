"""F4.6: проверяемая офлайн-визуализация ТОЛЬКО подтверждённых метрик F4.5.

Источник — metrics.jsonl + run_manifest.json от настоящего LoRA-тренинга.
Без GPU, ML-зависимостей, скачивания моделей, генерации фиктивных кривых.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from html import escape as html_escape
from io import StringIO
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape

from src.llm_f4.sft_lora import (
    Config as SFTConfig, SFTError, load_config as load_sft_config,
    make_plan as make_sft_plan, verify_run as verify_sft_run,
)


class VisualizationError(ValueError):
    """Нарушен контракт данных, исходных запусков или файлов визуализации."""


@dataclass(frozen=True, slots=True)
class Config:
    sft_config_file: Path
    output_dir: Path
    chart_width: int
    chart_height: int
    max_steps: int
    config_sha256: str


@dataclass(frozen=True, slots=True)
class Point:
    step: int
    train_loss: float
    grad_norm: float
    micro_batches: int


CONFIG_KEYS = {"schema_version", "sft_config_file", "output_dir", "chart_width", "chart_height", "max_steps"}
METRIC_KEYS = {"step", "train_loss", "grad_norm", "micro_batches"}
GENERATED = ("summary.json", "metrics.csv", "report.html", "train_loss.svg",
             "grad_norm.svg", "validation_loss.svg", "learning_rate.svg")


def _json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise VisualizationError(f"Повторный JSON-ключ: {key}")
        result[key] = value
    return result


def _constant(name: str) -> None:
    raise VisualizationError(f"Запрещённое JSON-значение: {name}")


def _decode_json(raw: bytes, name: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_json_pairs, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VisualizationError(f"Невозможно прочитать {name}: {exc}") from exc


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise VisualizationError(f"Отсутствует обязательный файл {path}: {exc}") from exc


def _path(root: Path, value: Any, field: str) -> Path:
    if (not isinstance(value, str) or not value or ":" in value or "\\" in value
            or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/"))):
        raise VisualizationError(f"{field}: нужен относительный POSIX-путь без '..'")
    base = root.resolve()
    result = (base / value).resolve()
    if not result.is_relative_to(base):
        raise VisualizationError(f"{field}: путь за пределами репозитория")
    return result


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise VisualizationError(f"{name}: ожидается целое от {low} до {high}")
    return value


def _number(value: Any, name: str, *, strict_positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise VisualizationError(f"{name}: ожидается конечное число")
    if (strict_positive and value <= 0) or (not strict_positive and value < 0):
        raise VisualizationError(f"{name}: отрицательное или нулевое значение недопустимо")
    return float(value)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2,
                       sort_keys=True) + "\n").encode("utf-8")


def load_config(path: Path, *, root: Path) -> Config:
    """Строгая конфигурация и защита путей от выхода за корень проекта."""
    content = _read(path)
    raw = _decode_json(content, str(path))
    if (not isinstance(raw, dict) or set(raw) != CONFIG_KEYS
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1):
        raise VisualizationError("Некорректная структура конфигурации F4.6")
    cfg = Config(
        _path(root, raw["sft_config_file"], "sft_config_file"),
        _path(root, raw["output_dir"], "output_dir"),
        _integer(raw["chart_width"], "chart_width", 600, 1600),
        _integer(raw["chart_height"], "chart_height", 240, 800),
        _integer(raw["max_steps"], "max_steps", 1, 10000),
        sha256(content).hexdigest(),
    )
    if cfg.sft_config_file != (root.resolve() / "configs/f4_05_sft_lora.json"):
        raise VisualizationError("F4.6 должен ссылаться на утверждённую конфигурацию F4.5")
    sft = load_sft_config(cfg.sft_config_file, root=root)
    if cfg.output_dir == sft.output_dir or cfg.output_dir.is_relative_to(sft.output_dir) or sft.output_dir.is_relative_to(cfg.output_dir):
        raise VisualizationError("Нельзя записывать графики внутрь артефактов обучения F4.5")
    return cfg


def get_sft(conf: Config, *, root: Path) -> SFTConfig:
    return load_sft_config(conf.sft_config_file, root=root)


def plan(conf: Config, *, root: Path) -> dict[str, Any]:
    """Проверить конфигурации F4.1–F4.5, не требуя готовых реальных логов."""
    sft = get_sft(conf, root=root)
    sft_plan, _ = make_sft_plan(sft, root=root)
    return {"source_stage": "F4.5", "model": sft.model_id,
            "revision": sft.model_revision, "splits": sft_plan["dataset_counts"],
            "profiles": {key: (sft.output_dir / key / "run_manifest.json").is_file()
                         for key in ("cpu", "gpu")},
            "model_loading_performed": False}


def parse_metrics(content: bytes, *, expected_steps: int, max_steps: int) -> tuple[Point, ...]:
    """JSONL без дубликатов, пропусков шагов, NaN или лишних показателей."""
    if type(expected_steps) is not int or not 1 <= expected_steps <= max_steps:
        raise VisualizationError("Количество шагов вне допустимого диапазона")
    if len(content) > max_steps * 1024:
        raise VisualizationError("Лог метрик слишком большой")
    if not content.endswith(b"\n"):
        raise VisualizationError("metrics.jsonl должен заканчиваться переводом строки")
    lines = content.splitlines()
    if len(lines) != expected_steps:
        raise VisualizationError("Число строк метрик не соответствует optimizer_steps")
    points = []
    for i, line in enumerate(lines, 1):
        raw = _decode_json(line, f"metrics.jsonl:{i}")
        if not isinstance(raw, dict) or set(raw) != METRIC_KEYS:
            raise VisualizationError(f"Неизвестные или отсутствующие поля metrics.jsonl:{i}")
        step = _integer(raw["step"], "step", 1, max_steps)
        if step != i:
            raise VisualizationError("Шаги optimizer должны начинаться с 1 и идти без пропусков")
        batch_count = _integer(raw["micro_batches"], "micro_batches", 1, 10000)
        points.append(Point(step, _number(raw["train_loss"], "train_loss"),
                            _number(raw["grad_norm"], "grad_norm"), batch_count))
    return tuple(points)


def validate_source(manifest: Mapping[str, Any], points: Sequence[Point], *, profile: str,
                    sft: SFTConfig) -> dict[str, Any]:
    """Разграничить реальный эксперимент, его метрики и незафиксированный LR."""
    if (profile not in sft.profiles or manifest.get("real_training_performed") is not True
            or manifest.get("mode") != profile or manifest.get("model_id") != sft.model_id
            or manifest.get("model_revision") != sft.model_revision
            or manifest.get("config_sha256") != sft.config_sha256):
        raise VisualizationError("Нет подтверждённого совместимого реального обучения F4.5")
    selected = sft.profiles[profile]
    if (type(manifest.get("optimizer_steps")) is not int or
            manifest["optimizer_steps"] != selected.max_steps or len(points) != selected.max_steps):
        raise VisualizationError("Длина реального обучения отличается от frozen F4.5 profile")
    if any(p.micro_batches != selected.gradient_accumulation for p in points):
        raise VisualizationError("Количество micro_batches не соответствует профилю F4.5")
    initial = _number(manifest.get("initial_validation_loss"), "initial_validation_loss")
    final = _number(manifest.get("final_validation_loss"), "final_validation_loss")
    trained = _integer(manifest.get("train_records_used"), "train_records_used", 1, 100000)
    validated = _integer(manifest.get("validation_records_used"), "validation_records_used", 1, 100000)
    if trained != selected.train_limit or validated != selected.validation_limit:
        raise VisualizationError("Объём train/validation не соответствует профилю")
    trainable = _integer(manifest.get("trainable_parameters"), "trainable_parameters", 1, 10**11)
    total = _integer(manifest.get("all_parameters_with_adapters"), "all_parameters_with_adapters", trainable, 10**12)
    fraction = _number(manifest.get("trainable_fraction"), "trainable_fraction", strict_positive=True)
    if not math.isclose(fraction, trainable/total, rel_tol=1e-10):
        raise VisualizationError("Доля обучаемых параметров не согласована")
    if manifest.get("evaluation_holdout_used_in_training") is not False or manifest.get("manual_quality_score", "missing") is not None:
        raise VisualizationError("F4.5 не должен использовать benchmark или утверждать ручную оценку")
    delta = final - initial
    relative = (delta / initial * 100) if initial > 0 else None
    return {
        "schema_version": 1, "stage": "F4.6", "source": "verified_real_F4.5_run",
        "profile": profile, "model_id": sft.model_id, "model_revision": sft.model_revision,
        "optimizer_steps": len(points), "train_records_used": trained, "validation_records_used": validated,
        "trainable_parameters": trainable, "all_parameters_with_adapters": total,
        "trainable_fraction": fraction, "learning_rate_configured": selected.learning_rate,
        "learning_rate_observed_per_step": False, "learning_rate_scheduler_observed": False,
        "train_loss_first": points[0].train_loss, "train_loss_last": points[-1].train_loss,
        "train_loss_step_mean": sum(p.train_loss for p in points)/len(points),
        "grad_norm_max_pre_clipping": max(p.grad_norm for p in points),
        "validation_loss_before": initial, "validation_loss_after": final,
        "validation_loss_delta": delta, "validation_loss_relative_pct": relative,
        "trend_interpretation": "single_step_no_trend" if len(points) == 1 else (
            "two_steps_insufficient" if len(points) == 2 else "trajectory_only_no_quality_claim"),
        "quality_improvement_measured": False,
        "benchmark_f4_03_used_in_training": False,
    }


def _f(value: float) -> str:
    return format(value, ".6g")


def _svg_shell(width: int, height: int, elements: Sequence[str], title: str, desc: str) -> bytes:
    text = '\n'.join(elements)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="svg-title svg-desc">\n'
            f'<title id="svg-title">{xml_escape(title)}</title><desc id="svg-desc">{xml_escape(desc)}</desc>\n'
            f'<rect width="100%" height="100%" fill="#ffffff" rx="12"/>\n{text}\n</svg>\n').encode("utf-8")


def _svg_line(name: str, values: Sequence[float], *, width: int, height: int,
              color: str, y_min_zero: bool = False) -> bytes:
    """График наблюдений. При одном шаге рисуется только точка, НЕ линия."""
    if not values or any(not math.isfinite(x) for x in values):
        raise VisualizationError("Невозможно отобразить нечисловую кривую")
    left, right, top, bottom = 75, width - 30, 53, height - 64
    low, high = min(values), max(values)
    if y_min_zero:
        low = 0.0
    elif high == low:
        low = max(0.0, low - max(0.1, 0.1 * abs(low)))
    span = high - low
    high = high + max(span * 0.1, 1e-8)
    if high <= low:
        high = low + 1.0
    x_pos = [left + (right-left)*(idx/(len(values)-1)) if len(values)>1 else (left+right)/2
             for idx in range(len(values))]
    y_pos = [bottom - (value-low)/(high-low)*(bottom-top) for value in values]
    pieces = [f'<text x="{left}" y="29" font-family="Arial,sans-serif" font-size="17" font-weight="600" fill="#14253e">{xml_escape(name)}</text>']
    for k in range(5):
        y = top + (bottom-top)*k/4
        v = high - (high-low)*k/4
        pieces.extend((f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                       f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#475569">{_f(v)}</text>'))
    pieces.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#64748b"/>')
    if len(values) > 1:
        xy = ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(x_pos, y_pos))
        pieces.append(f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="2.8" stroke-linejoin="round"/>')
    for x, y in zip(x_pos, y_pos):
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
    ticks = sorted({1, len(values), *([1 + (len(values)-1)*i//4 for i in range(1, 4)] if len(values)>2 else [])})
    for tick in ticks:
        x = x_pos[tick-1]
        pieces.append(f'<text x="{x:.1f}" y="{bottom+20}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#475569">{tick}</text>')
    pieces.append(f'<text x="{(left+right)/2:.1f}" y="{height-12}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#334155">Optimizer step</text>')
    if len(values) == 1:
        pieces.append(f'<text x="{right}" y="40" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#9a3412">1 point: no trend · {_f(values[0])}</text>')
    return _svg_shell(width, height, pieces, name, f'{len(values)} actual F4.5 measurements; one point does not establish a trend')


def _svg_validation(before: float, after: float, steps: int, *, width: int, height: int) -> bytes:
    title = 'Validation loss: two measured endpoints'
    top, bottom = 74, height-55
    ymax = max(before, after, 0.05)*1.2
    pts = [(width*0.36, before, 'Before'), (width*0.68, after, f'After {steps} steps')]
    body = [f'<text x="56" y="31" font-size="17" font-weight="600" fill="#14253e" font-family="Arial,sans-serif">{xml_escape(title)}</text>',
            f'<line x1="100" y1="{bottom}" x2="{width-50}" y2="{bottom}" stroke="#64748b"/>']
    for x, value, label in pts:
        bh = (bottom-top)*value/ymax
        y = bottom-bh
        body.extend([f'<rect x="{x-62:.1f}" y="{y:.1f}" width="124" height="{bh:.1f}" rx="6" fill="#0f766e"/>',
                     f'<text x="{x:.1f}" y="{max(55,y-10):.1f}" text-anchor="middle" font-size="16" font-weight="600" fill="#0f172a" font-family="Arial,sans-serif">{_f(value)}</text>',
                     f'<text x="{x:.1f}" y="{height-25}" text-anchor="middle" font-size="13" fill="#334155" font-family="Arial,sans-serif">{xml_escape(label)}</text>'])
    body.append(f'<text x="{width-30}" y="54" text-anchor="end" font-size="12" fill="#9a3412" font-family="Arial,sans-serif">Only 2 evaluations, not a learning curve</text>')
    return _svg_shell(width, height, body, title, 'Two F4.5 validation-loss evaluations: before and after training')


def _svg_lr(lr: float, *, width: int, height: int) -> bytes:
    body = [f'<text x="56" y="34" font-size="17" font-weight="600" fill="#14253e" font-family="Arial,sans-serif">Configured learning rate</text>',
            f'<rect x="56" y="70" width="{width-112}" height="{height-131}" rx="14" fill="#f0fdfa" stroke="#99f6e4"/>',
            f'<text x="{width/2:.1f}" y="{height/2+8:.1f}" text-anchor="middle" font-size="39" font-weight="700" fill="#0f766e" font-family="Arial,sans-serif">{_f(lr)}</text>',
            f'<text x="{width/2:.1f}" y="{height-39}" text-anchor="middle" font-size="13" fill="#475569" font-family="Arial,sans-serif">Set in F4.5 config; NO per-step LR telemetry</text>']
    return _svg_shell(width, height, body, 'Configured learning rate (not observed)', 'The optimizer learning rate is read from frozen F4.5 configuration, not step logs')


def _csv(points: Sequence[Point], lr: float) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("optimizer_step", "train_loss_step_mean", "grad_norm_pre_clip", "micro_batches", "learning_rate_configured_not_observed"))
    for p in points:
        writer.writerow((p.step, repr(p.train_loss), repr(p.grad_norm), p.micro_batches, repr(lr)))
    return output.getvalue().encode("utf-8")


def _html(summary: Mapping[str, Any], profile: str) -> bytes:
    title = f'F4.6 — {profile.upper()} — Training Visualization'
    warning = ('Только один optimizer step: динамику обучения, устойчивость снижения loss '
               'и улучшение модели по этому эксперименту оценить нельзя.' if summary['optimizer_steps'] == 1
               else ('Всего два optimizer step: надёжную динамику обучения оценить нельзя.' if summary['optimizer_steps'] == 2
                     else 'График показывает наблюдаемые шаги, но сам по себе не доказывает улучшения качества ответов.'))
    metrics = [
        ('Модель', summary['model_id']),
        ('Шаги оптимизатора', str(summary['optimizer_steps'])),
        ('Train / validation examples', f"{summary['train_records_used']} / {summary['validation_records_used']}"),
        ('Train loss: первый → последний', f"{_f(summary['train_loss_first'])} → {_f(summary['train_loss_last'])}"),
        ('Validation loss: до → после', f"{_f(summary['validation_loss_before'])} → {_f(summary['validation_loss_after'])}"),
        ('Δ validation loss (после − до)', _f(summary['validation_loss_delta'])),
        ('Max grad norm (до clipping)', _f(summary['grad_norm_max_pre_clipping'])),
        ('Заданный LR (не измерялся по шагам)', _f(summary['learning_rate_configured'])),
        ('Обучаемые параметры', str(summary['trainable_parameters'])),
    ]
    trs = '\n'.join(f'<tr><th>{html_escape(label)}</th><td>{html_escape(val)}</td></tr>' for label, val in metrics)
    charts = [('train_loss.svg', 'Train loss — наблюдаемые шаги'),
              ('grad_norm.svg', 'Grad norm — до clipping'),
              ('validation_loss.svg', 'Validation loss — измерения до и после'),
              ('learning_rate.svg', 'Learning rate — конфигурация, не телеметрия')]
    cards = '\n'.join(f'<section class="chart"><h2>{html_escape(h)}</h2><img src="{src}" alt="{html_escape(h)}"></section>'
                      for src, h in charts)
    doc = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html_escape(title)}</title><style>
:root{{color-scheme:light}}body{{margin:0;background:#f1f5f9;color:#14253e;font:16px/1.55 system-ui,Arial,sans-serif}}
main{{max-width:1050px;margin:0 auto;padding:32px 20px 48px}}h1{{margin:0;font-size:30px}}.meta{{color:#475569}}
.warning{{border-left:4px solid #d97706;padding:14px 18px;background:#fff7ed;margin:22px 0;border-radius:6px}}
.panel,.chart{{background:white;border:1px solid #e2e8f0;border-radius:14px;margin:18px 0;padding:18px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:10px 8px;border-bottom:1px solid #e2e8f0;text-align:left}}
th{{font-weight:500;color:#475569;width:55%}}td{{font-variant-numeric:tabular-nums}}h2{{font-size:19px}}
.chart img{{max-width:100%;height:auto;display:block;margin:auto}}.small{{color:#64748b;font-size:14px}}
</style></head><body><main>
<h1>F4.6 · Training Visualization</h1><p class="meta">Профиль {html_escape(profile.upper())} · Qwen2.5-0.5B Base + LoRA · данные подтверждены F4.5 VERIFY</p>
<div class="warning"><strong>Ограничение интерпретации.</strong> {html_escape(warning)}</div>
<section class="panel"><h2>Зафиксированные показатели</h2><table>{trs}</table></section>
{cards}
<section class="panel"><h2>Методика и ограничения</h2>
<p>Train loss взят из каждого optimizer step журнала F4.5. Это среднее по micro-batches одного шага, а не среднее по всему датасету. Gradient norm сохранена <strong>до</strong> clipping; F4.5 использует границу clipping 1.0.</p>
<p>Validation loss измерена F4.5 дважды: до обучения и после последнего шага. Это два значения, <strong>не</strong> кривая валидации. LR взят из зафиксированной конфигурации и <strong>не</strong> записывался в metrics.jsonl; изменения расписания LR не измерены.</p>
<p>Показатели не измеряют правильность технических ответов. Независимый benchmark F4.3 и ручная оценка относятся к F4.7. Уменьшение loss на малой выборке не доказывает повышение качества ответов.</p>
<p class="small">Отчёт создан локально без обращения к сети и загрузки весов. Источник: F4.5 run_manifest.json + metrics.jsonl. Машинные значения: <code>summary.json</code> и <code>metrics.csv</code>; контрольные SHA-256: <code>visualization_manifest.json</code>.</p>
</section></main></body></html>\n'''
    return doc.encode('utf-8')


def render_files(summary: Mapping[str, Any], points: Sequence[Point], conf: Config) -> dict[str, bytes]:
    """Построить детерминированные графики/таблицы без внешних зависимостей."""
    w, h = conf.chart_width, conf.chart_height
    lr = summary["learning_rate_configured"]
    return {
        "summary.json": _canonical(summary),
        "metrics.csv": _csv(points, lr),
        "report.html": _html(summary, summary["profile"]),
        "train_loss.svg": _svg_line('Training loss per optimizer step', [p.train_loss for p in points], width=w, height=h, color="#0f766e"),
        "grad_norm.svg": _svg_line('Gradient norm before clipping', [p.grad_norm for p in points], width=w, height=h, color="#2563eb", y_min_zero=True),
        "validation_loss.svg": _svg_validation(summary["validation_loss_before"], summary["validation_loss_after"], summary["optimizer_steps"], width=w, height=h),
        "learning_rate.svg": _svg_lr(lr, width=w, height=h),
    }


def _source(conf: Config, *, root: Path, profile: str) -> tuple[dict[str, Any], tuple[Point, ...], dict[str, str]]:
    if profile not in ("cpu", "gpu"):
        raise VisualizationError("Допустимы только профили cpu и gpu")
    sft = get_sft(conf, root=root)
    try:
        verified = verify_sft_run(sft, root=root, profile=profile)
    except (SFTError, OSError) as exc:
        raise VisualizationError(f"F4.5 VERIFY {profile.upper()} не пройден: {exc}") from exc
    folder = sft.output_dir / profile
    source_metrics = _read(folder / "metrics.jsonl")
    source_manifest = _read(folder / "run_manifest.json")
    # Проверяем неизменность логов после verify_run, иначе возможна гонка файлов.
    if sha256(source_metrics).hexdigest() != verified.get("metrics_sha256"):
        raise VisualizationError("Метрики изменились после F4.5 VERIFY")
    if _decode_json(source_manifest, "run_manifest.json") != verified:
        raise VisualizationError("Manifest изменился после F4.5 VERIFY")
    expected_steps = verified.get("optimizer_steps")
    if type(expected_steps) is not int:
        raise VisualizationError("В manifest отсутствует целое optimizer_steps")
    points = parse_metrics(source_metrics, expected_steps=expected_steps, max_steps=conf.max_steps)
    summary = validate_source(verified, points, profile=profile, sft=sft)
    fingerprints = {"f45_run_manifest_sha256": sha256(source_manifest).hexdigest(),
                    "f45_metrics_sha256": sha256(source_metrics).hexdigest(),
                    "f46_config_sha256": conf.config_sha256}
    summary.update(fingerprints)
    return summary, points, fingerprints


def build_report(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    """F4.5 verify => SVG/HTML/CSV/JSON; отказ от перезаписи существующего профиля."""
    summary, points, fingerprints = _source(conf, root=root, profile=profile)
    files = render_files(summary, points, conf)
    if set(files) != set(GENERATED):
        raise VisualizationError("Неполный набор выходных файлов")
    manifest = {"schema_version": 1, "stage": "F4.6", "profile": profile,
                "source_fingerprints": fingerprints,
                "generated_files_sha256": {name: sha256(data).hexdigest() for name, data in files.items()}}
    destination = conf.output_dir / profile
    if destination.exists() or destination.is_symlink():
        raise VisualizationError(f"Каталог уже существует (не перезаписываем): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".f46_", dir=destination.parent) as tmp:
        folder = Path(tmp)
        for name, payload in files.items():
            (folder / name).write_bytes(payload)
        (folder / "visualization_manifest.json").write_bytes(_canonical(manifest))
        if destination.exists() or destination.is_symlink():
            raise VisualizationError("Каталог появился во время построения; отказ от перезаписи")
        folder.rename(destination)
    return summary


def verify_report(conf: Config, *, root: Path, profile: str) -> dict[str, Any]:
    """Проверить реальный F4.5 снова и побайтово пересобрать ожидаемые результаты."""
    summary, points, fingerprints = _source(conf, root=root, profile=profile)
    destination = conf.output_dir / profile
    manifest = _decode_json(_read(destination / "visualization_manifest.json"), "visualization_manifest")
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "stage", "profile", "source_fingerprints", "generated_files_sha256"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["stage"] != "F4.6"
            or manifest["profile"] != profile or manifest["source_fingerprints"] != fingerprints
            or not isinstance(manifest["generated_files_sha256"], dict)
            or set(manifest["generated_files_sha256"]) != set(GENERATED)):
        raise VisualizationError("Manifest визуализации не соответствует текущим входам")
    expected = render_files(summary, points, conf)
    if {p.name for p in destination.iterdir()} != set(GENERATED) | {"visualization_manifest.json"}:
        raise VisualizationError("В отчёте отсутствуют файлы или есть посторонние")
    for name, body in expected.items():
        digest = sha256(_read(destination / name)).hexdigest()
        if digest != sha256(body).hexdigest() or digest != manifest["generated_files_sha256"].get(name):
            raise VisualizationError(f"Изменён файл визуализации: {name}")
    return summary
