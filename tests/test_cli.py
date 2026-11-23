from typer.testing import CliRunner

from arxiv_rag import generate
from arxiv_rag.cli import app

runner = CliRunner()

PROVIDER_ERROR = (
    'litellm.NotFoundError: GeminiException - {\n  "error": {\n    "code": 404,\n'
    '    "message": "This model models/old-model is no longer available.",\n  }\n}'
)


def test_short_message_extracts_provider_explanation():
    msg = generate._short_message(Exception(PROVIDER_ERROR))
    assert msg == "This model models/old-model is no longer available."


def test_llm_errors_are_one_clean_line(monkeypatch):
    def boom(model):
        raise generate.LLMError("Request to 'x' was rejected: gone")

    monkeypatch.setattr(generate, "check_llm", boom)
    result = runner.invoke(app, ["check-llm"])
    assert result.exit_code == 1
    assert "LLM error: Request to 'x' was rejected: gone" in result.output
    assert "Traceback" not in result.output


def test_eval_refuses_empty_testset(tmp_path):
    empty = tmp_path / "testset.jsonl"
    empty.write_text("")
    result = runner.invoke(app, ["eval", "--testset", str(empty)])
    assert result.exit_code == 1
    assert "make-testset" in result.output
    assert not (tmp_path / "RESULTS.md").exists()  # no bogus row recorded
