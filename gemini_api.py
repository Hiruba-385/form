
"""Wrapper around google-genai Gemini client."""

from __future__ import annotations

import logging
import os

from google import genai

logger = logging.getLogger(__name__)

API_KEY_ENV = "GEMINI_API_KEY"
MODEL_NAME = "gemini-3.8-flash"

_client = None


def _configure_client() -> genai.Client:
    api_key = os.getenv(API_KEY_ENV)

    if not api_key:
        raise RuntimeError(
            f"Environment variable {API_KEY_ENV} is not set"
        )

    return genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    global _client

    if _client is None:
        _client = _configure_client()

    return _client


def call_gemini(
    prompt: str,
    model: str = MODEL_NAME,
) -> str:
    try:
        response = get_client().models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 800,
                "response_mime_type": "application/json",
            },
        )

        text = response.text

        if not text or not text.strip():
            logger.error(
                "Gemini returned an empty response. "
                "Finish reason: %s",
                getattr(response, "candidates", None),
            )
            raise RuntimeError(
                "Gemini returned an empty response. "
                "Check the model response and API logs."
            )

        logger.info(
            "Gemini returned a response containing %d characters",
            len(text),
        )

        return text.strip()

    except Exception:
        logger.exception("Gemini API call failed")
        raise
