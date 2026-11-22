FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY app ./app
COPY scripts ./scripts
COPY knowledge_base ./knowledge_base
COPY evals ./evals
COPY prompts ./prompts
RUN useradd --create-home pingo && chown -R pingo:pingo /app
USER pingo
EXPOSE 8000
# Render supplies PORT. Disable access logs to keep OAuth callback queries out of logs.
CMD ["sh", "-c", "exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
