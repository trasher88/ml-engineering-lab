# syntax=docker/dockerfile:1
# F5.5: контейнер только для F5.4 synthetic CPU demo, без моделей F1–F4.
# Изменение тега базового образа может изменить окружение; для production нужен digest.
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Сначала зависимости — их слой не перестраивается при изменении исходников.
COPY requirements-f5_05.txt ./requirements-f5_05.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r requirements-f5_05.txt \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app

# Только утверждённые F5.4 runtime assets: whitelist дополнительно в .dockerignore.
COPY src/api_f5/__init__.py src/api_f5/app.py src/api_f5/demo_service.py ./src/api_f5/
COPY configs/f5_04_api.json ./configs/f5_04_api.json
COPY tests/fixtures/f5_04_api_demo_v1.json ./tests/fixtures/f5_04_api_demo_v1.json

# F5.4 load_demo() независимо проверяет SHA-256 fixture при запуске.
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200"]

# Uvicorn слушает 0.0.0.0 внутри namespace контейнера;
# на хосте порт публикуется только через 127.0.0.1 (см. F5.5 smoke).
CMD ["python", "-m", "uvicorn", "src.api_f5.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
