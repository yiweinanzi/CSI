FROM ghcr.io/astral-sh/uv:debian
WORKDIR /Wi-GATr
COPY . /Wi-GATr
RUN uv venv && uv sync

# Ensure the venv is active by default
ENV PATH="/Wi-GATr/.venv/bin:$PATH"
