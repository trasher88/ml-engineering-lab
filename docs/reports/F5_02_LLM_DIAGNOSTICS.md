# F5.2 — LLM Evaluation & Diagnostics

**Статус доказательств:** `PARTIAL`. Автоматические метрики — proxy; они не устанавливают factual correctness.

## Источник и протокол

- Исходный файл: `data/processed/f4_07/full/human_review.md`.
- SHA-256: `62be65d8bd2809598ab1f05e2de0dfb4cc573c7b0b5d726a2c3f7f011ee0869b`.
- Заданий: 32; проверяемых фактов: 64.
- Токенизация: Unicode-слова/пунктуация, casefold, multiset overlap.

## Автоматические proxy-метрики

| Показатель | Base | LoRA |
|---|---:|---:|
| Средний token F1 | 0.185899 | 0.184882 |
| Изменённых ответов (из 32) | — | 12 |

Разница token F1 (LoRA − Base): **-0.001017**. Положительная разница не означает улучшения correctness.

## Диагностические сигналы

| Flag | Base | LoRA |
|---|---:|---:|
| `low_reference_overlap` | 25 | 26 |
| `possible_prompt_echo` | 5 | 5 |
| `prompt_template_marker` | 7 | 7 |
| `repeated_fourgram` | 16 | 16 |

## Экспертная оценка по рубрике

- Полностью оценённых случаев: **0/32**.
- Неоценённых/частично оценённых случаев: **32**.
- Оценок без reviewer_id, исключённых из подсчёта: **0**.
- Средняя оценка Base (0–2): **NOT RATED**.
- Средняя оценка LoRA (0–2): **NOT RATED**.

Отсутствующие экспертные баллы остаются `null`; система не выставляет их по сходству текста с эталоном.

## Покейсный аудит (без публикации текста)

| Case | Домен | Изменился текст | F1 Base | F1 LoRA | Флаги Base | Флаги LoRA |
|---|---|---|---:|---:|---|---|
| `ev_py_01` | `python` | нет | 0.2574 | 0.2574 | prompt_template_marker, possible_prompt_echo, repeated_fourgram | prompt_template_marker, possible_prompt_echo, repeated_fourgram |
| `ev_py_02` | `python` | да | 0.2069 | 0.2069 | low_reference_overlap | low_reference_overlap |
| `ev_py_03` | `python` | да | 0.1250 | 0.1250 | low_reference_overlap | low_reference_overlap |
| `ev_py_04` | `python` | нет | 0.2200 | 0.2200 | low_reference_overlap, repeated_fourgram | low_reference_overlap, repeated_fourgram |
| `ev_py_05` | `python` | да | 0.1905 | 0.1923 | low_reference_overlap | low_reference_overlap |
| `ev_py_06` | `python` | нет | 0.1856 | 0.1856 | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram |
| `ev_py_07` | `python` | да | 0.2353 | 0.2353 | low_reference_overlap | low_reference_overlap |
| `ev_py_08` | `python` | нет | 0.1043 | 0.1043 | low_reference_overlap, repeated_fourgram | low_reference_overlap, repeated_fourgram |
| `ev_lx_01` | `linux` | да | 0.3025 | 0.4000 | repeated_fourgram | repeated_fourgram |
| `ev_lx_02` | `linux` | нет | 0.2913 | 0.2913 | repeated_fourgram | repeated_fourgram |
| `ev_lx_03` | `linux` | нет | 0.1731 | 0.1731 | low_reference_overlap | low_reference_overlap |
| `ev_lx_04` | `linux` | да | 0.1474 | 0.2128 | low_reference_overlap, repeated_fourgram | low_reference_overlap, repeated_fourgram |
| `ev_lx_05` | `linux` | нет | 0.1573 | 0.1573 | low_reference_overlap, repeated_fourgram | low_reference_overlap, repeated_fourgram |
| `ev_lx_06` | `linux` | нет | 0.1818 | 0.1818 | low_reference_overlap | low_reference_overlap |
| `ev_lx_07` | `linux` | да | 0.2791 | 0.2340 | — | low_reference_overlap, repeated_fourgram |
| `ev_lx_08` | `linux` | нет | 0.3077 | 0.3077 | — | — |
| `ev_sq_01` | `sql` | нет | 0.2069 | 0.2069 | low_reference_overlap | low_reference_overlap |
| `ev_sq_02` | `sql` | нет | 0.1149 | 0.1149 | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram |
| `ev_sq_03` | `sql` | да | 0.1978 | 0.2045 | low_reference_overlap | low_reference_overlap |
| `ev_sq_04` | `sql` | нет | 0.0723 | 0.0723 | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram |
| `ev_sq_05` | `sql` | нет | 0.1458 | 0.1458 | low_reference_overlap | low_reference_overlap |
| `ev_sq_06` | `sql` | да | 0.1778 | 0.1758 | low_reference_overlap | low_reference_overlap |
| `ev_sq_07` | `sql` | нет | 0.2524 | 0.2524 | repeated_fourgram | repeated_fourgram |
| `ev_sq_08` | `sql` | да | 0.2151 | 0.0635 | low_reference_overlap, repeated_fourgram | low_reference_overlap |
| `ev_ml_01` | `ml` | нет | 0.2353 | 0.2353 | low_reference_overlap | low_reference_overlap |
| `ev_ml_02` | `ml` | нет | 0.0625 | 0.0625 | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram | low_reference_overlap, prompt_template_marker, possible_prompt_echo, repeated_fourgram |
| `ev_ml_03` | `ml` | да | 0.0000 | 0.0000 | low_reference_overlap, prompt_template_marker, repeated_fourgram | low_reference_overlap, prompt_template_marker, repeated_fourgram |
| `ev_ml_04` | `ml` | нет | 0.2143 | 0.2143 | low_reference_overlap, repeated_fourgram | low_reference_overlap, repeated_fourgram |
| `ev_ml_05` | `ml` | да | 0.1837 | 0.1782 | low_reference_overlap | low_reference_overlap |
| `ev_ml_06` | `ml` | нет | 0.2931 | 0.2931 | — | — |
| `ev_ml_07` | `ml` | нет | 0.2118 | 0.2118 | low_reference_overlap | low_reference_overlap |
| `ev_ml_08` | `ml` | нет | 0.0000 | 0.0000 | low_reference_overlap, prompt_template_marker, repeated_fourgram | low_reference_overlap, prompt_template_marker, repeated_fourgram |

## Ограничения

- Token overlap измеряет сходство токенов, а не фактическую корректность ответа.
- Flags — эвристические сигналы для разбора, не диагноз hallucination и не подтверждённые ошибки.
- Критерии рубрики перечислены для учёта покрытия; оценивание фактов требует человека.
- Оценки без reviewer_id игнорируются; указанное reviewer_id — декларация источника, не доказательство квалификации эксперта.
- Сравнение Base/LoRA по одному frozen набору без экспертного анализа не доказывает улучшение качества.
- Результаты не являются LLM-as-a-judge и не предназначены для вывода о безопасности/production readiness.
