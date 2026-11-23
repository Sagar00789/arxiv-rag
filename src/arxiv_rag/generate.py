"""Answer generation from retrieved chunks, with numbered citations."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .retrieve import Hit

SYSTEM_PROMPT = """You answer questions about research papers using ONLY the numbered context \
passages provided.
- Cite every claim with the passage number in square brackets, e.g. [2].
- If the passages do not contain the answer, say "I couldn't find this in the indexed papers." \
Do not use outside knowledge.
- Be concise and technical."""


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.title}, arXiv:{h.chunk.paper_id})\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    )


def build_messages(question: str, hits: Sequence[Hit]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"},
    ]


class LLMError(RuntimeError):
    """A configuration problem with the LLM (bad model name, missing/invalid key)."""


def _short_message(e: Exception) -> str:
    text = str(e)
    m = re.search(r'"message":\s*"([^"]+)"', text)  # provider's own explanation, if any
    return (m.group(1) if m else text.splitlines()[0])[:400]


def complete(messages: list[dict], model: str, temperature: float = 0.0) -> str:
    import litellm

    litellm.suppress_debug_info = True
    try:
        # num_retries backs off on rate limits, which free tiers hit quickly
        resp = litellm.completion(
            model=model, messages=messages, temperature=temperature, num_retries=5
        )
    except (litellm.NotFoundError, litellm.BadRequestError) as e:
        raise LLMError(
            f"Model '{model}' was rejected: {_short_message(e)}\n"
            "Set LLM_MODEL to a model your provider currently offers."
        ) from None
    except litellm.AuthenticationError as e:
        raise LLMError(
            f"API key rejected for '{model}': {_short_message(e)}\n"
            "Check the key for this provider (e.g. GEMINI_API_KEY)."
        ) from None
    return resp.choices[0].message.content or ""


def check_llm(model: str) -> str:
    """One tiny call, so a bad model or key fails in seconds rather than mid-run."""
    return complete([{"role": "user", "content": "Reply with the word OK."}], model=model)


def generate_answer(question: str, hits: Sequence[Hit], model: str) -> str:
    if not hits:
        return "I couldn't find this in the indexed papers."
    return complete(build_messages(question, hits), model=model)
