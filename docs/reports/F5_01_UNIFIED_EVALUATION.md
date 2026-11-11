# F5.1 — Unified Evaluation / Report

**Статус сборки:** `PASS`. Данные считываются из сохранённых JSON; обучение и inference не выполняются.

## 1. Охват источников

- Источников в протоколе: **4**; с извлечёнными метриками: **4**.
- Извлечено метрик: **19**.
- Фазы без подходящих метрик: **нет**.

## 2. Метрики по независимым протоколам

### F1 — Классический ML

**Источник:** `reports/classical.json` · `EXTRACTED` · split `test` · dataset `f1_qa_source_labels` · protocol `f1_07_classical`.

SHA-256: `d1779ed698707ea98c13d687f8dd0052614fd8e8726005e77c83d301482b74a5`.

| Метрика | Значение | Единица | JSON selector |
|---|---:|---|---|
| `accuracy` | 0.8799342105 | `fraction` | `test.accuracy` |
| `macro_f1` | 0.8838453043 | `fraction` | `test.macro_f1` |
| `macro_precision` | 0.884620128 | `fraction` | `test.macro_precision` |
| `macro_recall` | 0.8831810397 | `fraction` | `test.macro_recall` |

### F2 — PyTorch

**Источник:** `outputs/f2_6_diagnostics/metrics.json` · `EXTRACTED` · split `validation` · dataset `f2_qa_source_labels` · protocol `f2_06_diagnostics`.

SHA-256: `db1fba67c244694376b3478c4b522e8fc9a37ec7e5d6d7389e5600aedd277dc8`.

| Метрика | Значение | Единица | JSON selector |
|---|---:|---|---|
| `macro_f1` | 0.78225166 | `fraction` | `best_val_macro_f1` |
| `best_epoch` | 5 | `count` | `best_epoch` |
| `epochs` | 5 | `count` | `epochs` |

### F3 — Retrieval

**Источник:** `reports/public/f3_07/F3_07_comparison.json` · `EXTRACTED` · split `frozen_test` · dataset `f3_06_synthetic_80_queries` · protocol `f3_07_frozen_retrieval`.

SHA-256: `b1351e7448e52bed8e3476155c4e227d843815092c3a8061edee209633639557`.

| Метрика | Значение | Единица | JSON selector |
|---|---:|---|---|
| `bm25_recall_at_5` | 0.6375 | `fraction` | `metrics.bm25.5.recall_at_k` |
| `vector_recall_at_5` | 0.7875 | `fraction` | `metrics.vector.5.recall_at_k` |
| `hybrid_recall_at_5` | 0.76875 | `fraction` | `metrics.hybrid.5.recall_at_k` |
| `bm25_mrr_at_5` | 0.7222916667 | `fraction` | `metrics.bm25.5.mrr_at_k` |
| `vector_mrr_at_5` | 0.8502083333 | `fraction` | `metrics.vector.5.mrr_at_k` |
| `hybrid_mrr_at_5` | 0.8310416667 | `fraction` | `metrics.hybrid.5.mrr_at_k` |

### F4 — LLM / Fine-Tuning

**Источник:** `data/processed/f4_06/cpu/summary.json` · `EXTRACTED` · split `train_validation` · dataset `f4_01_instruction_dataset` · protocol `f4_05_cpu_single_step_lora`.

SHA-256: `75121ed2c9eed55f6ee6ccca91bb21c79e3cf26739f1cc12cdc9157e9ce2eeb9`.

| Метрика | Значение | Единица | JSON selector |
|---|---:|---|---|
| `optimizer_steps` | 1 | `count` | `optimizer_steps` |
| `train_loss` | 2.39262414 | `loss` | `train_loss_last` |
| `validation_loss_before` | 2.288156748 | `loss` | `validation_loss_before` |
| `validation_loss_after` | 2.281612158 | `loss` | `validation_loss_after` |
| `trainable_parameters` | 540672 | `count` | `trainable_parameters` |
| `learning_rate_configured` | 0.0002 | `learning_rate` | `learning_rate_configured` |

## 3. Интерпретация и ограничения

Метрики разных задач **не суммируются и не усредняются**. Доля корректных классификаций, доля релевантных документов в top-k и validation loss имеют разные знаменатели и разные направления интерпретации.

`PASS` означает только полноту извлечения **по данному конфигурационному контракту**, не повторную независимую оценку модели, не production readiness и не качество генерации.

- SHA256 подтверждает целостность прочитанных файлов, но не подлинность исходного эксперимента.
- Метрики относятся к собственным протоколам и разным наборам данных; общего score нет.
- F4.5/F4.6: один optimizer step не доказывает тенденцию обучения или улучшение качества ответов.
- Автоматическая выжимка не заменяет F1/F3/F4 аудит исходных отчётов.
- Отсутствующие источники и неподходящие selectors отражаются как PARTIAL, значения не подставляются.

## 4. Метод воспроизведения

`python -m scripts.f5_unified_report --check` сверяет текущие файлы с сохранённым отчётом. Несовпадение SHA-256 или значений требует повторной сборки.
