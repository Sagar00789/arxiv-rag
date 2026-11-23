FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# CPU-only torch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[all]"

# Bake the embedding model into the image so cold starts don't download it
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

# The index is built beforehand (`arxiv-rag ingest && arxiv-rag index`) and copied in
COPY qdrant_data ./qdrant_data

EXPOSE 8000
CMD ["uvicorn", "arxiv_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
