# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm

# uv as a static binary, version-pinned. Keeps the official Python base image
# and gives the same resolver/installer you use locally.
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /submission

# Dependency layer first: rebuilds only when requirements.txt changes.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv "$VIRTUAL_ENV" --python 3.12 && \
    uv pip install --python "$VIRTUAL_ENV/bin/python" -r requirements.txt

# Source (and checkpoint.pt, if you bake it in rather than mount it).
COPY . .

ENTRYPOINT ["python", "predict.py"]
CMD ["--input_dir", "/data/input", \
     "--output_file", "/output/predictions.csv", \
     "--checkpoint", "/submission/checkpoint.pt"]