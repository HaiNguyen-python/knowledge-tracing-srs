# Pinned to Debian bookworm: plain "-slim" tags now follow Debian trixie.
FROM python:3.12-slim-bookworm

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train the HLR bundle on synthetic data at build time, so the image starts serving at once.
RUN ktrace train --model /app/artifacts/model.json

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["ktrace", "serve", "--model", "/app/artifacts/model.json", "--host", "0.0.0.0", "--port", "8000"]
