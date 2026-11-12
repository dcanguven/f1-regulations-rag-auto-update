# F1 Regulations AI | RAG Assistant with Automated FIA Updates

Ask questions about Formula 1 regulations and get answers based on official FIA documents, with references to the original rules.

The application can monitor new FIA regulation releases and update its knowledge base as the rules change.

![F1 Regulations AI Demo](https://github.com/user-attachments/assets/5e23b5f1-9ca2-4141-bf77-99bdf4e65af7)

## What can it do?

- **Ask about F1 rules:** Get answers to questions about technical and sporting regulations.
- **Check the sources:** See which FIA documents and articles were used to answer your question.
- **Explore future changes:** Find rules scheduled to take effect in upcoming seasons.
- **Stay up to date:** Detect new FIA document versions and update the knowledge base.
- **Run locally:** Use the application without paid AI APIs.

The system uses a local language model and combines semantic search with keyword search to find relevant information.

## Built with

Python, RAG, Ollama, Qwen3.5, Sentence Transformers, Qdrant, BM25, FastAPI, Streamlit and PyMuPDF.

All regulation documents come from the [official FIA website](https://www.fia.com/regulation/category/110).

## Getting started

You'll need Python 3.12+, [Ollama](https://ollama.com) and an internet connection for the initial setup.

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/dcanguven/f1-regulations-rag-auto-update.git
cd f1-regulations-rag-auto-update

conda create -n f1rag python=3.12
conda activate f1rag

pip install -e ".[frontend]"
cp .env.example .env
ollama pull qwen3.5:9b
```

**2. Download the latest FIA regulations**

```bash
python -m app.ingestion.sync
```

This downloads the official documents and builds the local search index. The first run may take a few minutes.

**3. Start the application**

Start Ollama if it isn't already running:

```bash
ollama serve
```

In another terminal, with the `f1rag` environment active:

```bash
scripts/start.sh
```

Open **http://localhost:8501** to use the chatbot.

## Keeping the regulations updated

The application can check the FIA website for new regulation versions and update only the documents that have changed.

To check manually:

```bash
python -m app.ingestion.sync
```

For automatic scheduled checks on macOS:

```bash
scripts/update_schedule.sh install
```

Scheduled updates are optional and are not enabled by default.

## More details

For a closer look at the architecture and technical implementation, see [Architecture](docs/architecture.md) and [Project Details](docs/PORTFOLIO.md).

*This is an independent project and is not affiliated with or endorsed by the FIA or Formula 1.*
