FROM python:3.12-slim

RUN pip install --no-cache-dir uv
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY config ./config
COPY src ./src
RUN uv sync --frozen --no-dev

ENV AUTOALPHA_HOST=0.0.0.0
ENV AUTOALPHA_PORT=8788
EXPOSE 8788
ENTRYPOINT ["/workspace/.venv/bin/autoalpha-service"]
