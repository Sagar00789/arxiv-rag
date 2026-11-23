"""Answer generation from retrieved chunks, with numbered citations."""

from __future__ import annotations

import logging
import re
import time
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
    """An LLM problem the user has to fix: bad model name, bad key, or quota used up."""


log = logging.getLogger(__name__)

_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_TRANSIENT = (
    "RateLimitError",
    "APIConnectionError",
    "Timeout",
    "InternalServerError",
    "ServiceUnavailableError",
)

# Client-side pacing: minimum seconds between calls (0 = no pacing).
_min_interval = 0.0
_last_call = 0.0


def set_rate_limit(requests_per_minute: float) -> None:
    """Pace calls to stay under a provider's per-minute limit (e.g. 15 RPM on free tiers)."""
    global _min_interval
    _min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0


def _pace() -> None:
    global _last_call
    wait = _last_call + _min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _short_message(e: Exception) -> str:
    text = str(e)
    m = re.search(r'"message":\s*"([^"]+)"', text)  # provider's own explanation, if any
    return (m.group(1) if m else text.splitlines()[0])[:400]


def _retry_delay(e: Exception, attempt: int) -> float:
    """Use the provider's suggested wait if it gives one, else exponential backoff."""
    m = _RETRY_IN.search(str(e))
    return float(m.group(1)) + 1.0 if m else min(60.0, 5.0 * 2**attempt)


def _is_daily_quota(e: Exception) -> bool:
    text = str(e).lower()
    return "perday" in text or "per day" in text


def complete(
    messages: list[dict], model: str, temperature: float = 0.0, max_retries: int = 8
) -> str:
    import litellm

    litellm.suppress_debug_info = True
    # Some models (e.g. newer Gemini) reject settings like `temperature`; drop them
    # instead of failing, so any LLM_MODEL works without code changes.
    litellm.drop_params = True
    transient = tuple(t for n in _TRANSIENT if (t := getattr(litellm, n, None)))

    for attempt in range(max_retries + 1):
        _pace()
        try:
            resp = litellm.completion(
                model=model, messages=messages, temperature=temperature, num_retries=0
            )
            return resp.choices[0].message.content or ""
        except (litellm.NotFoundError, litellm.BadRequestError) as e:
            raise LLMError(
                f"Request to '{model}' was rejected: {_short_message(e)}\n"
                "If the model name is wrong, set LLM_MODEL to one your provider currently offers."
            ) from None
        except litellm.AuthenticationError as e:
            raise LLMError(
                f"API key rejected for '{model}': {_short_message(e)}\n"
                "Check the key for this provider (e.g. GEMINI_API_KEY)."
            ) from None
        except transient as e:
            if _is_daily_quota(e):
                raise LLMError(
                    f"Daily free-tier quota for '{model}' is used up. "
                    "Try again tomorrow, or set LLM_MODEL to another model."
                ) from None
            if attempt == max_retries:
                raise LLMError(
                    f"'{model}' still failing after {max_retries} retries: {_short_message(e)}"
                ) from None
            delay = _retry_delay(e, attempt)
            log.info("%s; waiting %.0fs before retrying", type(e).__name__, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def check_llm(model: str) -> str:
    """One tiny call, so a bad model or key fails in seconds rather than mid-run."""
    return complete([{"role": "user", "content": "Reply with the word OK."}], model=model)


def generate_answer(question: str, hits: Sequence[Hit], model: str) -> str:
    if not hits:
        return "I couldn't find this in the indexed papers."
    return complete(build_messages(question, hits), model=model)
