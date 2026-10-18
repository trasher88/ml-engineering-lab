# Поэтапная публикация ML Engineering Lab

## Принцип

Проект был реализован локально до публикации на GitHub. История Git отражает **постепенный аудит и публикацию готовой работы**, а не даты первоначальной разработки. Не публиковать сразу все 43 этапа и не добавлять фиктивные коммиты.

| Пакет | Содержание | Проверка перед commit |
|---|---|---|
| P0 | README, архитектура, .gitignore, provenance, манифесты, структурный CI | `python scripts/check_repository.py` |
| F1 — Data | Данные: pipeline, download script, конфигурации, схема | Синтаксис, прогон препроцессинга, leakage test на разрешённых локальных данных |
| F1 — Classical | baseline, метрики, анализ ошибок, отчёт F1 | Macro F1 и test split воспроизведены или отмечены как исторический результат |
| F2 | PyTorch train, validation, графики, сравнение с F1 | Форма тензоров, train loop, unit tests |
| F3 | SearchIndex, BM25/vector/RRF, benchmark, RAG | Recall@k, MRR, source provenance |
| F4 | SFT/DPO, frozen eval, отчёт и ограничения tiny LLM | Проверить код и артефакты, не обещать улучшение качества |
| F5 | FastAPI, safety, MLflow, Docker, CI, full demo | Unit/API/Integration отдельно, Docker только после реального запуска |
| v1.0 | Итоговая витрина, interview guide, финальный отчёт | Актуальные инструкции запуска, ссылки, результаты |

## Правила приёма шага

1. Разобрать назначение и зависимости кода.
2. Проверить, что никакие секреты, объёмные модели и фрагменты чужого контента не попадают в git.
3. Запустить релевантные проверки в поддерживаемом окружении, отделив реальные PASS от NOT RUN.
4. Обновить README/документацию и зафиксировать ограничения.
5. Создать отдельный осмысленный commit.

## Публикация P0 (PowerShell, Windows)

Создай пустой публичный репозиторий `ml-engineering-lab` на GitHub **без** автоматического README, `.gitignore` и LICENSE. Распакуй папку `ml-engineering-lab` из архива P0 на диск.

```powershell
cd "C:\Projects\ml-engineering-lab"  # замени на реальный путь
python scripts/check_repository.py
git init -b main
git add .
git status --short
git commit -m "chore: initialize ML Engineering Lab portfolio"
git remote add origin https://github.com/YOUR_USERNAME/ml-engineering-lab.git
git push -u origin main
```

`YOUR_USERNAME` обязательно замени на свой GitHub login. Git должен быть установлен; авторизацию GitHub выполняй через менеджер учётных данных / браузер. Не вставляй токены в README или код.

После push дождись запуска **Repository integrity** в Actions. Локальный PASS не равен удалённому PASS.

### Следующее действие

После первого push проводим F1.1: ревью foundation/config/CLI и первые проверенные исходники. Полный ZIP с исходниками предназначен для **локального разбора, а не загрузки целиком**.
