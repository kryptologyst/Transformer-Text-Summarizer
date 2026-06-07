"""Visualization utilities for text summarization."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_summary_comparison(
    original: str,
    summaries: dict[str, str],
    save_path: str | None = None,
) -> None:
    models = list(summaries.keys())
    orig_words = len(original.split())
    summary_words = [len(s.split()) for s in summaries.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh(["Original"] + models, [orig_words] + summary_words, color=["gray"] + plt.cm.Set2.colors[:len(models)])
    ax1.set_xlabel("Word Count")
    ax1.set_title("Text Length Comparison")

    ratios = [orig_words / max(w, 1) for w in summary_words]
    ax2.bar(models, ratios, color=plt.cm.Set2.colors[:len(models)])
    ax2.set_ylabel("Compression Ratio")
    ax2.set_title("Compression Ratio (Original / Summary)")
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Summary comparison plot saved to %s", save_path)
    plt.close(fig)


def plot_word_frequencies(
    text: str,
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    from collections import Counter
    import re

    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop_words = {"the", "and", "that", "for", "are", "with", "this", "from", "have", "been"}
    filtered = [w for w in words if w not in stop_words]
    counts = Counter(filtered).most_common(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    labels, values = zip(*counts)
    ax.barh(list(labels), list(values), color="teal")
    ax.set_xlabel("Frequency")
    ax.set_title("Top Word Frequencies")
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Word frequency plot saved to %s", save_path)
    plt.close(fig)
