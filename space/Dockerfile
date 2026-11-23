# Hugging Face Spaces image (CPU). Deployed by .github/workflows/deploy-space.yml.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir ".[all,ui]"

COPY --chown=user eval/paper_ids.txt ./eval/paper_ids.txt
COPY --chown=user space/start.sh ./start.sh

USER user
# Bake the embedding model into the image so the first question doesn't wait for a download
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

EXPOSE 7860
CMD ["bash", "start.sh"]
