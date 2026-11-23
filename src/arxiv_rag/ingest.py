"""Download papers from arXiv, extract their text and chunk them."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path

from .chunking import Chunk, chunk_paper, save_chunks

log = logging.getLogger(__name__)

USER_AGENT = "arxiv-rag/0.1 (https://github.com/Sagar00789/arxiv-rag)"


def download_pdf(url: str, dest: Path, retries: int = 3, timeout: int = 60) -> None:
    """Download a PDF with retries. Rejects non-PDF responses (e.g. HTML error pages).

    We do this ourselves because recent versions of the `arxiv` package removed
    `Result.download_pdf()`.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(data)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)


def fetch_papers(
    query: str, max_results: int, pdf_dir: Path, id_list: list[str] | None = None
) -> list[dict]:
    """Search arXiv (or fetch exact IDs) and download each PDF. Returns paper metadata."""
    import arxiv

    pdf_dir.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    if id_list:
        search = arxiv.Search(id_list=id_list, max_results=len(id_list))
        max_results = len(id_list)
    else:
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
    papers = []
    for r in client.results(search):
        paper_id = r.get_short_id()
        filename = f"{paper_id.replace('/', '_')}.pdf"
        pdf_path = pdf_dir / filename
        if not pdf_path.exists():
            url = getattr(r, "pdf_url", None) or f"https://arxiv.org/pdf/{paper_id}"
            try:
                download_pdf(url, pdf_path)
            except Exception as e:  # one bad PDF shouldn't stop the run
                log.warning("Skipping %s: %s", paper_id, e)
                continue
            time.sleep(1)  # be polite to arxiv.org
            log.info("Downloaded %s (%d/%d)", paper_id, len(papers) + 1, max_results)
        papers.append(
            {
                "paper_id": paper_id,
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "published": r.published.date().isoformat(),
                "abstract": r.summary,
                "url": r.entry_id,
                "pdf_path": str(pdf_path),
            }
        )
    return papers


def pdf_to_text(path: Path) -> str:
    import pymupdf

    pymupdf.TOOLS.mupdf_display_errors(False)  # harmless drawing-syntax warnings
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def read_id_list(path: Path) -> list[str]:
    """One arXiv ID per line; blank lines and # comments ignored."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()]


def build_corpus(
    query: str,
    max_results: int,
    data_dir: Path,
    size: int,
    overlap: int,
    id_list: list[str] | None = None,
) -> tuple[int, int]:
    """Fetch papers and write data/papers.jsonl and data/chunks.jsonl."""
    papers = fetch_papers(query, max_results, data_dir / "pdfs", id_list=id_list)
    if not papers:
        raise RuntimeError("No papers were downloaded. Check the warnings above.")
    all_chunks: list[Chunk] = []
    for p in papers:
        try:
            text = pdf_to_text(Path(p["pdf_path"]))
        except Exception as e:
            log.warning("Could not parse %s: %s", p["paper_id"], e)
            continue
        all_chunks.extend(chunk_paper(p["paper_id"], p["title"], text, size, overlap))

    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "papers.jsonl").open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_chunks = save_chunks(all_chunks, data_dir / "chunks.jsonl")
    return len(papers), n_chunks
