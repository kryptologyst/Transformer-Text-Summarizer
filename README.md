# Transformer Text Summarizer

**Abstractive text summarization** using HuggingFace BART, T5, and Pegasus models.

## Overview

- Multiple transformer models: BART, T5, Pegasus, DistilBART
- Bullet-point summarization mode
- Long text chunking with overlap
- **Streamlit dashboard** with model comparison
- CLI for batch summarization

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main summarize --topic ai_overview
# CLI: python -m src.main compare
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# Transformer-Text-Summarizer
