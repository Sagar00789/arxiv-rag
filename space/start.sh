#!/usr/bin/env bash
# Builds the corpus on first boot (papers are fetched from arXiv, not shipped in the repo),
# then starts the web demo. Takes ~5 minutes the first time a Space wakes up.
set -euo pipefail

if [ ! -s data/chunks.jsonl ]; then
  echo "First start: downloading the 49 papers from arXiv and building the index..."
  arxiv-rag ingest --ids-file eval/paper_ids.txt
  arxiv-rag index
fi

exec arxiv-rag ui --host 0.0.0.0 --port 7860
