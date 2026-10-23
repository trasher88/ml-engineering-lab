# Происхождение данных

Source: [flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl).
Revision: `32151f5480872e6db89ae147e1d727266f574606`. Obtained UTC: 2026-10-12T10:15:25.265497+00:00.
License dataset card: **CC BY NC SA 4.0**, non-commercial/share-alike. Это metadata mirror, не юридическая проверка всех оригинальных posts.
Авторы/URL оригинальных posts отсутствуют в данном export; полноценная индивидуальная атрибуция не восстановлена. Перед внешним распространением или коммерческим применением требуется источник с оригинальной attribution и проверка условий.

| File | Available | Selected |
|---|---:|---:|
| unix.stackexchange.com.jsonl.gz | 6173 | 800 |
| dba.stackexchange.com.jsonl.gz | 2502 | 800 |
| serverfault.com.jsonl.gz | 7969 | 800 |
| codereview.stackexchange.com.jsonl.gz | 666 | 666 |

Raw поля физически представлены JSON-массивом [title_body, upvoted_answer, downvoted_answer], а viewer показывает названные columns.
Mapping: title_body→context; upvoted_answer→answer; downvoted_answer→rejected. Label=site→domain. ID=file + 0-based raw line. source_url ведёт к pinned gzip, source_row указывает строку.
language=en — metadata корпуса; отдельный language detector не запускался. quality_score не добавлен: калиброванной оценки качества нет.
Instruction — явно добавленный шаблон задачи, не поле источника. Group_id/split вычисляются preprocessing.

Seed=42; максимум 800/section, codereview доступно 666. Всего 3066 raw subset; после очистки 3040.
Sample — 60 real records/section, 240 всего, taken до cleaning. Он нужен для CI/smoke и не подменяет основной эксперимент.
SHA256 каждого raw file: [manifest](../data/raw/manifest.json). Subset SHA256: `fd4a3c1a8e160d89adb49efff9cf7429ca5df6f42682d4d65891dad4c81b12c3`.
Raw missing values и cleaning counters: локальном data_audit.json (будет опубликован вместе с этапом F1). Raw gzip не изменяется; interim и processed пересчитываются.

## F1.2 — публичный контроль происхождения

Metadata можно проверить без исходных текстов: `python -m src.data.provenance`.
Проверяются четыре pinned URL, 40-символьная ревизия, ожидаемые SHA-256
(формат), выбранное количество строк, пути и перечень полей.
Это **не** проверка реальных байтов: для неё после F1.3 нужен
`python -m src.data.provenance --verify-files`. Скачивание не запускается автоматически.

Отдельно: Stack Exchange использует лицензии CC BY-SA, зависящие от
даты конкретного поста/ревизии. Лицензия карточки HF CC BY-NC-SA 4.0
не устанавливает однозначно условия всех исходных Q&A. Если требуются
публичные текстовые примеры или коммерческое использование, сначала
восстановить исходные посты/авторов и проверить соответствующие условия.
До этого raw, subset, sample и processed тексты не коммитятся.

Справка: [Stack Exchange licensing](https://english.stackexchange.com/help/licensing),
[Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
