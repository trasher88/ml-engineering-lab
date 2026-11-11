# ML Engineering Lab

**Практический учебный проект по Machine Learning, NLP, LLM и ML Engineering:** от подготовки текстовых данных и построения классификатора до поиска информации, экспериментов с дообучением языковой модели, оценки и демонстрационного API.

Проект построен вокруг технических вопросов и ответов из Stack Exchange. Его цель — последовательно реализовать основные компоненты ML/AI-системы, научиться проверять результаты и понимать ограничения каждого подхода. Это **инженерная лаборатория**, а не готовый промышленный AI-продукт.

**Масштаб:** 5 фаз · 43 этапа · Python · локальные эксперименты преимущественно на CPU.

## Архитектура проекта

```mermaid
flowchart LR
    A[Технические Q&A] --> B[F1: данные и baseline]
    B --> C[F2: нейросеть PyTorch]
    C --> D[F3: поиск и RAG]
    D --> E[F4: SFT / LoRA / DPO]
    E --> F[F5: оценка и ML-инфраструктура]
```

Фазы изучают связанные задачи, но их результаты **не следует считать одним непрерывно обученным production-пайплайном**: часть проверок F5 использует синтетический backend, а не реальные модели предыдущих фаз.

## Фазы

| Фаза | Этапы | Что реализовано | Краткие итоги |
|---|---:|---|---|
| [F1 — Data Engineering & Classical ML](docs/phases/F1.md) | 9 | Подготовка данных, разделение выборки, TF-IDF + Logistic Regression | Test Macro F1 **0.8838** |
| [F2 — PyTorch & Neural Networks](docs/phases/F2.md) | 8 | Autograd, embedding-классификатор, обучение, validation, диагностика | Test Macro F1 **0.7813** |
| [F3 — Transformers & Retrieval](docs/phases/F3.md) | 9 | Attention, embeddings, BM25, vector search, Hybrid RRF, RAG с provenance | Лучший Silver Recall@5: **0.8500** (vector) |
| [F4 — LLM Fine-Tuning](docs/phases/F4.md) | 9 | Instruction dataset, Qwen2.5-0.5B, SFT + LoRA, учебный DPO | Реальные CPU-шаги обучения выполнены; рост качества ответов **не доказан** |
| [F5 — Evaluation & ML Engineering](docs/phases/F5.md) | 8 | Сводная оценка, диагностика, safety, API, Docker, тесты, MLflow, CLI-demo | Инфраструктурные проверки пройдены на учебных сценариях |

Подробное техническое описание отдельных этапов — в [`docs/steps/`](docs/steps/). Отчёты и проверочные материалы — в [`docs/reports/`](docs/reports/) и [`reports/public/`](reports/public/).

## Ключевые результаты

| Направление | Результат | Как интерпретировать |
|---|---|---|
| Классический ML | 3 040 очищенных записей; 608 test-примеров; Accuracy **0.8799**, Macro F1 **0.8838** | Сильный baseline для учебной классификации |
| Нейросеть PyTorch | Test Macro F1 **0.7813** | На этом эксперименте уступила классическому baseline |
| Retrieval, 80 запросов | Silver Recall@5: BM25 **0.4500**, Vector **0.8500**, Hybrid RRF **0.5500** | Vector оказался лучше Hybrid на зафиксированном benchmark |
| Дообучение LLM | SFT + LoRA: один реальный optimizer step; validation loss **2.2882 → 2.2816** | Изменение loss за один шаг не доказывает улучшения ответов |
| ML Engineering | FastAPI и Docker HTTP smoke: **8/8**; MLflow: **6/6** запусков; CLI demo: **10/10** сценариев | Проверена учебная инфраструктура; demo/API используют синтетический backend |

**Принцип проекта:** отрицательный или неоднозначный результат — тоже результат. Мы отделяем воспроизводимую метрику от предположений о качестве, а техническую проверку сервиса — от качества реальной LLM.

## Технологии

- **Данные и классический ML:** Python, pandas, scikit-learn, TF-IDF, Logistic Regression.
- **Нейросети:** PyTorch, autograd, `Dataset`/`DataLoader`, training loop, checkpointing.
- **Поиск и RAG:** Transformer/Attention, sentence-transformers, BM25, cosine similarity, Reciprocal Rank Fusion, provenance.
- **LLM:** Hugging Face Transformers, Qwen2.5-0.5B, SFT, LoRA, учебный DPO.
- **Инфраструктура:** FastAPI, Docker, pytest, MLflow, CLI, GitHub Actions (конфигурация).

## Структура репозитория

```text
ml-engineering-lab/
├── README.md                # Обзор проекта
├── configs/                 # Конфигурации экспериментов
├── data/                    # Манифесты и метаданные данных
├── docs/
│   ├── phases/              # Краткие итоги F1–F5
│   ├── steps/               # Подробная документация этапов
│   └── reports/             # Отчёты по фазам
├── models/                  # Манифесты моделей и метаданные
├── reports/public/          # Публикуемые результаты
├── scripts/                 # Запуск и проверка экспериментов
├── src/                     # Реализации алгоритмов и сервисов
├── tests/                   # Тесты
└── .github/workflows/       # Конфигурации CI
```


## Статус публикации
### Данные и классический ML
1. [F1.1 - Основа ML-проекта](docs/steps/F1_01_project_foundation.md)
2. [F1.2 - Выбор датасета](docs/steps/F1_02_dataset_selection.md)
3. [F1.3 - Получение данных](docs/steps/F1_03_data_ingestion.md)
4. [F1.4 - Очистка](docs/steps/F1_04_cleaning.md)
5. [F1.5 - Разведочный анализ](docs/steps/F1_05_eda.md)
6. [F1.6 - Split без утечки](docs/steps/F1_06_splits.md)
7. [F1.7 - TF-IDF + Logistic Regression](docs/steps/F1_07_classical_baseline.md)
8. [F1.8 - Анализ ошибок](docs/steps/F1_08_error_analysis.md)
9. [F1.9 - Итоговый отчёт F1](docs/steps/F1_09_report.md)
10. [**F1 - Итоги Фазы 1**](docs/reports/F1_PHASE_REPORT.md)

### PyTorch и обучение нейросети
1. [F2.1 - PyTorch: Tensor and autograd (Тензоры, вычислительный граф и градиенты)](docs/steps/F2_01_autograd.md)
2. [F2.2 - Dataset, DataLoader и токенизация](docs/steps/F2_02_dataset_dataloader.md)
3. [F2.3 - Embedding Classifier: Первая маленькая нейросетевая архитектура](docs/steps/F2_03_embedding_classifier.md)
4. [F2.4 - Training Loop: обучение Embedding Classifier в PyTorch](docs/steps/F2_04_training_loop.md)
5. [F2.5 - Validation & Checkpointing: Оценка и выбор лучшей нейросети](docs/steps/F2_05_validation_checkpointing.md)
6. [F2.6 - Training Diagnostics & Visualization: Графики обучения](docs/steps/F2_06_training_diagnostics.md)
7. [F2.7 - Classical ML vs Neural Network: Сравнение](docs/steps/F2_07_classical_vs_neural.md)
8. [F2.8 - Final Phase Report: Сведение проверяемых результатов](docs/steps/F2_08_final_phase_report.md)
9. [**F2 - Итоги Фазы 2**](docs/reports/F2_PHASE_REPORT.md)

## Воспроизводимость и ограничения

- Данные разделяются на train/validation/test; версии внешних ресурсов и контрольные суммы фиксируются в манифестах: [`data/raw/manifest.json`](data/raw/manifest.json), [`models/external_manifest.json`](models/external_manifest.json).
- Retrieval-метрики рассчитаны по **silver-разметке**; это не экспертная проверка корректности сформированных ответов.
- F4 включает минимальные реальные CPU-эксперименты SFT/LoRA и DPO, но **не подтверждает прирост качества LLM**. Экспертная оценка ответов не завершена.
- F5 проверяет API, контейнер, MLflow и CLI в учебных режимах. **Реальная интеграция F1–F4/Qwen в inference-backend не доказана**, как и оценка safety на реальной LLM. Конфигурация GitHub Actions есть, но результат выполнения CI отдельно не подтверждён.
- CUDA/GPU-профиль не проверен. Сервис и safety-проверки не следует трактовать как production-ready.
- Исходные тексты и обработанный корпус не публикуются до правовой проверки: используемый источник обозначен как **CC BY-NC-SA 4.0**, а индивидуальная атрибуция авторов неполна. Лицензия на исходный код репозитория отдельно не установлена.

## Навигация

**Краткие итоги:** [F1](docs/phases/F1.md) · [F2](docs/phases/F2.md) · [F3](docs/phases/F3.md) · [F4](docs/phases/F4.md) · [F5](docs/phases/F5.md)

**Подробности:** [`docs/steps/`](docs/steps/) · [`docs/reports/`](docs/reports/) · [`reports/public/`](reports/public/)
