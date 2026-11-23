import io

import pytest

from arxiv_rag import ingest


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_pdf_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ingest.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"%PDF-1.7 x")
    )
    dest = tmp_path / "p.pdf"
    ingest.download_pdf("https://arxiv.org/pdf/1", dest)
    assert dest.read_bytes().startswith(b"%PDF")


def test_download_pdf_rejects_html(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest.urllib.request, "urlopen", lambda *a, **k: FakeResponse(b"<html>"))
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)
    with pytest.raises(ValueError):
        ingest.download_pdf("https://arxiv.org/pdf/1", tmp_path / "p.pdf", retries=2)
    assert not (tmp_path / "p.pdf").exists()


def test_build_corpus_fails_loudly_with_no_papers(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "fetch_papers", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="No papers"):
        ingest.build_corpus("q", 5, tmp_path, 300, 50)
