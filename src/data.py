"""Data loading utilities for text summarization."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_TEXTS = {
    "ai_overview": (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "in contrast to the natural intelligence displayed by humans and animals. "
        "Leading AI textbooks define the field as the study of intelligent agents: "
        "any device that perceives its environment and takes actions that maximize "
        "its chance of successfully achieving its goals. Colloquially, the term "
        "artificial intelligence is often used to describe machines that mimic "
        "cognitive functions that humans associate with the human mind, such as "
        "learning and problem solving. As machines become increasingly capable, "
        "tasks considered to require intelligence are often removed from the "
        "definition of AI, a phenomenon known as the AI effect."
    ),
    "climate_change": (
        "Climate change includes both global warming driven by human-induced "
        "emissions of greenhouse gases and the resulting large-scale shifts in "
        "weather patterns. Though there have been previous periods of climatic "
        "change, since the mid-20th century humans have had an unprecedented "
        "impact on Earth's climate system and caused change on a global scale. "
        "The largest driver of warming is the emission of gases that create a "
        "greenhouse effect, of which more than 90% are carbon dioxide and methane. "
        "Fossil fuel burning for energy consumption is the main source of these "
        "emissions, with additional contributions from agriculture, deforestation, "
        "and manufacturing."
    ),
    "quantum_computing": (
        "Quantum computing is the use of quantum phenomena such as superposition "
        "and entanglement to perform computation. Computers that perform quantum "
        "computations are known as quantum computers. Quantum computers are believed "
        "to be able to solve certain computational problems, such as integer "
        "factorization, substantially faster than classical computers. The study "
        "of quantum computing is a subfield of quantum information science. "
        "Quantum computing began in the early 1980s when physicist Paul Benioff "
        "proposed a quantum mechanical model of the Turing machine. Richard Feynman "
        "and Yuri Manin later suggested that a quantum computer had the potential "
        "to simulate things a classical computer could not feasibly do."
    ),
}


def load_sample_text(topic: str = "ai_overview") -> str:
    return SAMPLE_TEXTS.get(topic, SAMPLE_TEXTS["ai_overview"])


def load_text_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")


def load_text_from_url(url: str) -> str:
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())
    except ImportError:
        logger.warning("requests/bs4 not installed; returning placeholder")
        return f"Content from {url} (install requests and beautifulsoup4 to scrape)"
