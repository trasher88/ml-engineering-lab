# F3.7 — BM25 vs Vector vs Hybrid: отчёт эксперимента

## Условия сравнения

- Benchmark: `f3.6-synthetic-retrieval-v1` (SHA-256 manifest: `a584868b8818552815151a85dfde714a762f14d959813e607bb4319cec70756c`).
- Конфигурация F3.7 (SHA-256): `6682562919c46b608823689a34dfb5983ed671ad21e3ec844fbcacdb4e315fb7`.
- Encoder: `sentence-transformers/paraphrase-MiniLM-L3-v2` @ `4ca70771034acceecb2e72475f72050fcdde4ddc`.
- Лицензия, заявленная в model card: `apache-2.0`.
- Manifest с SHA файлов модели: `59cfd5c59c7e2e95cc670c80469809778691875b8f4fcd093914346af13ad7f5`.
- Библиотеки: `huggingface_hub=2.4.0`, `torch=2.14.1+cpu`, `transformers=5.22.1`.
- Всего запросов: **80**; используется весь frozen набор, не выборка удачных случаев.
- Режим: **pretrained Transformer, CPU inference, без обучения и подбора параметров по qrels**.
- BM25: `k1=1.5`, `b=0.75`; exact cosine; 
  RRF: `k=60`, веса `1.0`/`1.0`, 
  candidate pool `20` на поисковик.

## Общие результаты

| Метод | K | Recall@K | MRR@K |
|---|---:|---:|---:|
| BM25 | 1 | 0.3250 | 0.6500 |
| BM25 | 5 | 0.6375 | 0.7223 |
| BM25 | 10 | 0.7312 | 0.7291 |
| VECTOR | 1 | 0.4000 | 0.8000 |
| VECTOR | 5 | 0.7875 | 0.8502 |
| VECTOR | 10 | 0.8938 | 0.8559 |
| HYBRID | 1 | 0.3812 | 0.7625 |
| HYBRID | 5 | 0.7688 | 0.8310 |
| HYBRID | 10 | 0.8750 | 0.8356 |

## По стилю запроса (K=5)

| Группа | Метод | N | Recall@5 | MRR@5 |
|---|---|---:|---:|---:|
| lexical | bm25 | 40 | 0.8875 | 0.9875 |
| lexical | vector | 40 | 0.9125 | 0.9833 |
| lexical | hybrid | 40 | 0.9125 | 1.0000 |
| paraphrase | bm25 | 40 | 0.3875 | 0.4571 |
| paraphrase | vector | 40 | 0.6625 | 0.7171 |
| paraphrase | hybrid | 40 | 0.6250 | 0.6621 |

## По техническому домену (K=5)

| Группа | Метод | N | Recall@5 | MRR@5 |
|---|---|---:|---:|---:|
| databases | bm25 | 20 | 0.6250 | 0.6975 |
| databases | vector | 20 | 0.6500 | 0.7667 |
| databases | hybrid | 20 | 0.7750 | 0.8417 |
| linux | bm25 | 20 | 0.6000 | 0.6750 |
| linux | vector | 20 | 0.9500 | 0.9350 |
| linux | hybrid | 20 | 0.7750 | 0.8142 |
| programming | bm25 | 20 | 0.7250 | 0.7917 |
| programming | vector | 20 | 0.8250 | 0.9100 |
| programming | hybrid | 20 | 0.8000 | 0.8767 |
| systems | bm25 | 20 | 0.6000 | 0.7250 |
| systems | vector | 20 | 0.7250 | 0.7892 |
| systems | hybrid | 20 | 0.7250 | 0.7917 |

## Диагностика разногласий (K=5)

Дельта Recall@5 считается относительно BM25; примеры иллюстрируют различия, а не являются отдельной оценкой качества.

### VECTOR относительно BM25: улучшений 28, ухудшений 8, без изменений 44

**Примеры улучшений:**
- `qry-0084d212f624e8dd` (paraphrase, linux): ΔRecall@5=+1.00; BM25 top-5=['doc-34ab3040f462c408', 'doc-dfb217e507caae6d', 'doc-4ae3286cc237299f', 'doc-c3fba9998af841fe', 'doc-340478b40c305b7d']; vector top-5=['doc-5357c812caf1ca82', 'doc-1ff70b08c91060b6', 'doc-c3fba9998af841fe', 'doc-ffe83e9d8feb3573', 'doc-42e3259e8e775b5e'].
- `qry-1d17963df556c0f8` (paraphrase, programming): ΔRecall@5=+1.00; BM25 top-5=['doc-ccae3a305aabfa5f', 'doc-42e3259e8e775b5e', 'doc-b456380aa5e6b979', 'doc-24b462aced1f90b3', 'doc-783b98988963895b']; vector top-5=['doc-ea8ed763832e8996', 'doc-ccae3a305aabfa5f', 'doc-634066890c1c26f0', 'doc-61db7a457fede1c9', 'doc-c3fba9998af841fe'].
- `qry-56784c758ffd0baa` (paraphrase, linux): ΔRecall@5=+1.00; BM25 top-5=['doc-34ab3040f462c408', 'doc-dfb217e507caae6d', 'doc-fa19d9cf439cb167', 'doc-a923b740b67980a6', 'doc-bd4e63d0e38c9f23']; vector top-5=['doc-3584a0561b43c2cf', 'doc-4ac973ea7a68014f', 'doc-783b98988963895b', 'doc-80d0c66a887e8b51', 'doc-7d718f047f419750'].

**Примеры ухудшений:**
- `qry-10f7ebe134d2fbc6` (lexical, databases): ΔRecall@5=-0.50; BM25 top-5=['doc-3cc7d48d20343742', 'doc-b450c62ed37e0501', 'doc-a923b740b67980a6', 'doc-584a530dcf759bbd', 'doc-ea8ed763832e8996']; vector top-5=['doc-3cc7d48d20343742', 'doc-a6108eb1a5387a98', 'doc-a00d46e97dfdefbb', 'doc-a748b69d3c78798b', 'doc-a923b740b67980a6'].
- `qry-4a0d8c0a5798d0e3` (lexical, databases): ΔRecall@5=-0.50; BM25 top-5=['doc-a748b69d3c78798b', 'doc-fd0e52e60a131c43', 'doc-a6108eb1a5387a98', 'doc-f92e10566a88e69b', 'doc-3cc7d48d20343742']; vector top-5=['doc-a748b69d3c78798b', 'doc-a6108eb1a5387a98', 'doc-3cc7d48d20343742', 'doc-9e523df11037d5d2', 'doc-b18669db0ef5018c'].
- `qry-6432cc4725e3945d` (paraphrase, programming): ΔRecall@5=-0.50; BM25 top-5=['doc-b04fe615d8313e6f', 'doc-cdb635d4dca69998', 'doc-647412681155459a', 'doc-9e523df11037d5d2', 'doc-634066890c1c26f0']; vector top-5=['doc-ea8ed763832e8996', 'doc-e9883cb73bfa06d0', 'doc-634066890c1c26f0', 'doc-994893baa31daa52', 'doc-bd4e63d0e38c9f23'].

### HYBRID относительно BM25: улучшений 21, ухудшений 2, без изменений 57

**Примеры улучшений:**
- `qry-223d26e01a005177` (paraphrase, systems): ΔRecall@5=+1.00; BM25 top-5=['doc-dfb217e507caae6d', 'doc-ea8ed763832e8996', 'doc-ccae3a305aabfa5f', 'doc-34ab3040f462c408', 'doc-d55df7cf08c0b6d8']; hybrid top-5=['doc-ccae3a305aabfa5f', 'doc-c3fba9998af841fe', 'doc-42e3259e8e775b5e', 'doc-994893baa31daa52', 'doc-bd4e63d0e38c9f23'].
- `qry-66d5eb6c6dd5109c` (paraphrase, linux): ΔRecall@5=+1.00; BM25 top-5=['doc-a923b740b67980a6', 'doc-b04fe615d8313e6f', 'doc-b456380aa5e6b979', 'doc-364eb0468195fad7', 'doc-634066890c1c26f0']; hybrid top-5=['doc-a923b740b67980a6', 'doc-5bb0841304bf9371', 'doc-e873d7c858862d55', 'doc-b04fe615d8313e6f', 'doc-bd53a7b604614ea4'].
- `qry-0084d212f624e8dd` (paraphrase, linux): ΔRecall@5=+0.50; BM25 top-5=['doc-34ab3040f462c408', 'doc-dfb217e507caae6d', 'doc-4ae3286cc237299f', 'doc-c3fba9998af841fe', 'doc-340478b40c305b7d']; hybrid top-5=['doc-c3fba9998af841fe', 'doc-1ff70b08c91060b6', 'doc-340478b40c305b7d', 'doc-34ab3040f462c408', 'doc-5357c812caf1ca82'].

**Примеры ухудшений:**
- `qry-7530c5084955c673` (paraphrase, databases): ΔRecall@5=-0.50; BM25 top-5=['doc-ea8ed763832e8996', 'doc-994893baa31daa52', 'doc-bd4e63d0e38c9f23', 'doc-dfb217e507caae6d', 'doc-a00d46e97dfdefbb']; hybrid top-5=['doc-141059be14b642cf', 'doc-b962d55f8fa219ae', 'doc-994893baa31daa52', 'doc-d55df7cf08c0b6d8', 'doc-671125a54cecbc3b'].
- `qry-c99755a165a64851` (lexical, programming): ΔRecall@5=-0.50; BM25 top-5=['doc-364eb0468195fad7', 'doc-a4dedfefa055135a', 'doc-6e72e83fe2331727', 'doc-34ab3040f462c408', 'doc-634066890c1c26f0']; hybrid top-5=['doc-364eb0468195fad7', 'doc-634066890c1c26f0', 'doc-ccae3a305aabfa5f', 'doc-a4dedfefa055135a', 'doc-ea8ed763832e8996'].

## Ограничения

- Это **авторский синтетический** benchmark из F3.6, а не независимое измерение производственного поиска.
- Qrels двоичные и не охватывают все потенциально полезные документы; отсутствие qrel означает 0 только по контракту набора.
- Модель F3.2 не обучалась на этом benchmark; конфигурация F3.7 была зафиксирована до оценки.
- Vector использует exact cosine, BM25 использует простую токенизацию без морфологии, Hybrid объединяет РАНГИ, а не вероятности.
- Изменение frozen-разметки, версии encoder либо гиперпараметров требует отдельного эксперимента, а не перезаписи данного результата.
- Если hybrid оказался слабее одного из методов, результат остаётся фактическим: сочетание рангов не гарантирует прироста.
- Полные 80 ранжированных списков каждого метода и метаданные доступны в сопутствующем JSON-артефакте.
