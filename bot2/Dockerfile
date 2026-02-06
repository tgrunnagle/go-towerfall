FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/ ./src/
RUN uv pip install -e .

ENV GAME_SERVER_HTTP_URL="http://backend-service:4000"
ENV GAME_SERVER_WS_URL="ws://backend-service:4000/ws"

CMD ["uv", "run", "python", "-m", "bot"]
