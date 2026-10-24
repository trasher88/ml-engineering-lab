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

## F1.3 — воспроизводимая загрузка

[`scripts/download_data.py`](../scripts/download_data.py) читает **закреплённый** manifest, не заменяет его новым состоянием HF, проверяет SHA-256 исходных gzip перед чтением и создаёт локальные `data/raw/subset.jsonl` и `data/sample/technical_qa.jsonl`. Режим `--offline` использует уже полученные архивы. Подробности и тесты: [F1.3](steps/F1_03_data_ingestion.md).

До реального запуска на новой машине подтверждены **лишь синтетические тесты**. Ожидаемые размеры subset=3066 и sample=240 происходят из исторических локальных отчётов, не из загрузки в CI. Все исходные тексты остаются исключены из публичного Git.

## F1.4 — локальная очистка (без публикации текстов)

[`src/data/cleaning.py`](../src/data/cleaning.py) воспроизводит правила нормализации оригинального F1 pipeline, а [`scripts/clean_data.py`](../scripts/clean_data.py) применяет их к полному subset F1.3. Пустые/некорректные обязательные поля, повторные ID и полностью совпадающие вопросы, а также записи с вопросом или ответом короче `min_words=8` исключаются. Подробности: [F1.4](steps/F1_04_cleaning.md).

Исторический `reports/data_audit.json`: 3066 входных, 26 слишком коротких, 3040 выходных записей. Число 535 — связи **почти** дубликатов при подготовке групп для split (F1.6), это **не 535 удалённых строк**. При повторном локальном запуске F1.4 на Windows получены те же 3066→3040; в публичном CI сохраняются только синтетические тесты. Очищенные тексты остаются локальными.


## F1.5 — разведочный анализ очищенного корпуса

`src/data/eda.py` и `scripts/eda_data.py` вычисляют **только агрегированные** статистики `domain`, длины `context` и `answer` (числа слов, медиана, p90/p95), количество пустых обязательных полей и SHA-256 очищенного локального JSONL. По историческому `data_audit.json` распределение: databases=800, linux=793, programming=666, systems=781 (в сумме 3040). Новая проверка должна самостоятельно подтвердить эти числа.

Созданные `reports/data_eda_audit.json` и `docs/visuals/f1/*.png` остаются локальными и исключены из Git. **F1.5 ещё не делает split**: сравнение по train/val/test и контроль группового пересечения относятся к F1.6. [Учебный разбор](steps/F1_05_eda.md).

## F1.6 — групповой split без прямой лексической утечки

`src/data/splitting.py` и `scripts/split_data.py` строят **лексические** группы из `context` и `answer` по char_wb TF-IDF (3–5), cosine >= 0.92 и union-find. Далее StratifiedGroupKFold (5 частей, seed=42) отделяет train/val/test. В локальном отчёте хранятся агрегаты, источник и его SHA-256, не исходные тексты.

Оригинальный исторический аудит: 3040 записей, 535 совпадающих парных связей по вопросам/ответам и train/val/test = 1824/608/608; group overlap между выборками — 0. Точные числа нужно перепроверить на локальном полном корпусе. Это не доказывает отсутствие **любых** семантических перефразировок. [F1.6](steps/F1_06_splits.md).

## F1.7 — обучение только на train

[`src/classical_ml/train.py`](../src/classical_ml/train.py) читает только локальные
результаты F1.6. Предварительно проверяет отсутствие дублирующихся ID и
пересечений `group_id`, затем `Pipeline.fit` вызывается **только с train**.
Validation и test используются для заранее определённой оценки, без изменения
TF-IDF и параметров классификатора. Отчёт и веса модели исключены из Git,
так как текст корпуса имеет не до конца разрешённые права публикации, а веса
содержат словарь обучающего корпуса. [Учебный разбор](steps/F1_07_classical_baseline.md).

## F1.8 — анализ ошибок без распространения текстов

После F1.7 скрипт `scripts/analyze_classical.py` проверяет SHA-256 test-выборки по зафиксированному `reports/classical.json`, сопоставляет ID/labels/метрики и строит confusion matrix, per-class Precision/Recall/F1 и частоты направленных ошибок. Агрегаты, PNG и ID ошибочных записей остаются локально, а исходные Q&A не копируются в артефакты. Стадия не выполняет повторное обучение или загрузку модели. [Учебный разбор](steps/F1_08_error_analysis.md).


## F1.9 — проверяемый агрегированный отчёт

`scripts/build_f1_report.py` сверяет F1.4–F1.8 по численности, class distribution, SHA-256 очищенного корпуса и train/val/test, confusion matrix и F1.8 error counts. В `docs/reports/F1_PHASE_REPORT.md` публикуются только согласованные агрегаты и хеши; оригинальные Q&A, ID ошибок, веса модели и локальные аудиты остаются вне Git. Детали: [F1.9](steps/F1_09_report.md).
