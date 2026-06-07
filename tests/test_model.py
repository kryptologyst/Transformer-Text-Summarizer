"""Tests for transformer text summarizer."""

from __future__ import annotations

import pytest

from src.data import SAMPLE_TEXTS, load_sample_text, load_text_file
from src.model import MODEL_MAP, TextSummarizer


class TestTextSummarizer:
    @pytest.fixture
    def sample_text(self) -> str:
        return load_sample_text("ai_overview")

    def test_init_loads_model(self) -> None:
        summarizer = TextSummarizer(model_name="bart")
        assert summarizer._pipe is not None

    def test_summarize_returns_string(self, sample_text: str) -> None:
        summarizer = TextSummarizer(model_name="bart")
        result = summarizer.summarize(sample_text, max_length=80, min_length=20)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summarize_shorter_than_original(self, sample_text: str) -> None:
        summarizer = TextSummarizer(model_name="bart")
        result = summarizer.summarize(sample_text, max_length=80, min_length=20)
        assert len(result.split()) < len(sample_text.split())

    def test_summarize_bullets_returns_list(self, sample_text: str) -> None:
        summarizer = TextSummarizer(model_name="bart")
        bullets = summarizer.summarize_bullets(sample_text, num_bullets=3)
        assert isinstance(bullets, list)
        assert len(bullets) <= 3

    def test_model_map_has_expected_keys(self) -> None:
        for key in ["bart", "t5", "pegasus", "distilbart"]:
            assert key in MODEL_MAP

    def test_load_sample_text_returns_string(self) -> None:
        for topic in SAMPLE_TEXTS:
            assert isinstance(load_sample_text(topic), str)

    def test_invalid_model_falls_back_to_bart(self) -> None:
        summarizer = TextSummarizer(model_name="nonexistent")
        assert summarizer.model_id == MODEL_MAP["bart"]
