"""Abstractive text summarization using HuggingFace transformers (BART/T5)."""

from __future__ import annotations

import logging
from typing import Any

from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_MAP = {
    "bart": "facebook/bart-large-cnn",
    "t5": "t5-small",
    "pegasus": "google/pegasus-xsum",
    "distilbart": "sshleifer/distilbart-cnn-12-6",
}


class TextSummarizer:
    """Transformer-based abstractive text summarizer."""

    def __init__(self, model_name: str = "bart", device: int = -1) -> None:
        self.model_key = model_name
        self.model_id = MODEL_MAP.get(model_name, MODEL_MAP["bart"])
        self.device = device
        self._pipe: Any = None
        self._load()

    def _load(self) -> None:
        logger.info("Loading summarization model: %s", self.model_id)
        self._pipe = pipeline(
            "summarization",
            model=self.model_id,
            device=self.device,
        )

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        do_sample: bool = False,
    ) -> str:
        if self._pipe is None:
            raise RuntimeError("Pipeline not loaded")
        result = self._pipe(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
        )
        return result[0]["summary_text"]

    def summarize_bullets(
        self,
        text: str,
        num_bullets: int = 3,
        max_length: int = 200,
    ) -> list[str]:
        prompt = (
            f"Summarize the following text into {num_bullets} bullet points:\n\n{text}"
        )
        summary = self.summarize(prompt, max_length=max_length, min_length=30)
        bullets = [b.strip("-• ").strip() for b in summary.split("\n") if b.strip()]
        return bullets[:num_bullets]

    def summarize_long(
        self,
        text: str,
        chunk_size: int = 1024,
        overlap: int = 100,
        max_length: int = 150,
    ) -> str:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        summaries = [self.summarize(c, max_length=max_length) for c in chunks]
        if len(summaries) > 1:
            combined = " ".join(summaries)
            return self.summarize(combined, max_length=max_length)
        return summaries[0]
