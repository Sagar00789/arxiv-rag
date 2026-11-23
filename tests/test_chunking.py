import pytest

from arxiv_rag.chunking import chunk_paper, chunk_text, clean_text, load_chunks, save_chunks


def test_chunks_overlap_and_cover_everything():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), size=10, overlap=3)
    assert chunks[0].split() == words[:10]
    assert chunks[1].split()[:3] == words[7:10]  # 3-word overlap
    assert chunks[-1].split()[-1] == "w24"  # nothing dropped at the end


def test_short_text_is_one_chunk_and_empty_is_none():
    assert chunk_text("just a few words", size=10, overlap=2) == ["just a few words"]
    assert chunk_text("   ", size=10, overlap=2) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_params(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size=size, overlap=overlap)


def test_clean_text_drops_references_and_fixes_hyphens():
    body = "Our retri-\never works well. " * 20
    text = body + "\nReferences\n[1] Someone et al. 2020."
    cleaned = clean_text(text)
    assert "retriever" in cleaned
    assert "Someone et al" not in cleaned


def test_chunk_ids_and_roundtrip(tmp_path):
    chunks = chunk_paper("2401.00001", "A Paper", "word " * 700, size=300, overlap=50)
    assert [c.chunk_id for c in chunks] == [f"2401.00001::{i}" for i in range(len(chunks))]
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    assert load_chunks(path) == chunks
