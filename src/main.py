"""CLI for transformer-based text summarization."""

from __future__ import annotations

import logging

import typer

from src.data import load_sample_text, load_text_file
from src.model import TextSummarizer

app = typer.Typer(help="Transformer Text Summarizer CLI")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def summarize(
    text: str | None = typer.Option(None, help="Text to summarize"),
    file: str | None = typer.Option(None, help="File path to summarize"),
    topic: str = typer.Option("ai_overview", help="Sample topic: ai_overview, climate_change, quantum_computing"),
    model: str = typer.Option("bart", help="Model: bart, t5, pegasus, distilbart"),
    max_length: int = typer.Option(150, help="Max summary length in tokens"),
    min_length: int = typer.Option(40, help="Min summary length in tokens"),
    bullets: bool = typer.Option(False, help="Output as bullet points"),
) -> None:
    if text:
        source = text
    elif file:
        source = load_text_file(file)
    else:
        source = load_sample_text(topic)

    summarizer = TextSummarizer(model_name=model)
    logger.info("Original length: %d words", len(source.split()))

    if bullets:
        points = summarizer.summarize_bullets(source)
        for i, p in enumerate(points, 1):
            typer.echo(f"{i}. {p}")
    else:
        result = summarizer.summarize(source, max_length=max_length, min_length=min_length)
        typer.echo(result)


@app.command()
def compare(
    text: str | None = typer.Option(None, help="Text to summarize"),
    topic: str = typer.Option("ai_overview", help="Sample topic"),
) -> None:
    source = text or load_sample_text(topic)
    models = ["bart", "t5", "distilbart"]
    for m in models:
        summarizer = TextSummarizer(model_name=m)
        result = summarizer.summarize(source)
        typer.echo(f"\n--- {m} ---")
        typer.echo(result)


if __name__ == "__main__":
    app()
