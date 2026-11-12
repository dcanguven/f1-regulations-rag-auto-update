from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE = (
    "I could not find sufficient support for this answer in the indexed FIA regulations."
)


def generate_structured(
    system_prompt: str,
    user_message: str,
    schema: dict[str, Any],
    timeout: float | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "keep_alive": settings.llm_keep_alive,
        "options": {
            "temperature": 0,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_max_tokens,
        },
    }
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=timeout or settings.llm_timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        if meta is not None:
            meta.update(
                prompt_tokens=body.get("prompt_eval_count"),
                output_tokens=body.get("eval_count"),
                load_ms=round(body.get("load_duration", 0) / 1e6, 1),
                prompt_ms=round(body.get("prompt_eval_duration", 0) / 1e6, 1),
                decode_ms=round(body.get("eval_duration", 0) / 1e6, 1),
            )
        return json.loads(body["message"]["content"])
    except httpx.HTTPError as e:
        logger.error("Ollama request failed: %s", e)
        raise RuntimeError(f"LLM unavailable: {e}") from e
    except (KeyError, json.JSONDecodeError) as e:
        logger.error("Unexpected Ollama response: %s", e)
        raise RuntimeError("Malformed LLM response") from e
