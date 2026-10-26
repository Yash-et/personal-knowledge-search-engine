# PKS Engine - Personal Knowledge Search

**PKS = Personal Knowledge Search**

A local-first personal knowledge search engine that lets you index your documents, search their content semantically, and ask questions using retrieval-augmented generation (RAG).

The project combines document processing, local embeddings, persistent vector storage, semantic retrieval, and a local large language model in a Streamlit interface.

## Application Screenshots

### Main Application

![PKS Engine — Streamlit application](https://github.com/user-attachments/assets/295d0946-04ea-4834-8df7-45bce420e174)

### Semantic Search

![PKS Engine — Semantic Search](https://github.com/user-attachments/assets/6a171ede-2564-4084-ad09-3a52db96fcf4)

### Document Library

![PKS Engine - Document Library](https://github.com/user-attachments/assets/dc9d1180-ef77-4bca-9aae-eb8799322a10)

## Features

- **Document ingestion:** Load supported PDF, TXT, and Markdown documents.
- **Text chunking:** Split documents into smaller passages suitable for retrieval.
- **Local embeddings:** Convert text into numerical representations using Sentence Transformers.
- **Persistent vector storage:** Store and retrieve document passages with ChromaDB.
- **Semantic search:** Find relevant passages based on meaning rather than exact keyword matches.
- **Ask Your Documents:** Generate answers using retrieved document passages through Ollama.
- **Source citations:** Display source filenames and page information with generated answers when available.
- **Document Library:** View indexed documents and their passage counts.
- **Document removal:** Remove a selected document's passages from the index, with confirmation.
- **Re-indexing:** Replace previously indexed passages when a document changes.
- **Local model inference:** Run the language model locally using Ollama.
- **Automated tests:** Validate core functionality with pytest.

## Technology Stack

| Component | Technology |
|---|---|
| User interface | Streamlit |
| Programming language | Python |
| Document parsing | pypdf |
| Text embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Local language model runtime | Ollama |
| Language model | `llama3.2:3b` |
| Testing | pytest |

Dependencies are pinned in `requirements.txt` to make environment setup more reproducible.

## How It Works

The application follows a retrieval-augmented generation workflow.

1. **Load:** Read a supported document and extract its text and metadata.
2. **Chunk:** Divide the extracted text into smaller passages while preserving source and page information.
3. **Embed:** Generate embeddings for the passages.
4. **Index:** Store the passages, metadata, and embeddings in a persistent ChromaDB collection.
5. **Retrieve:** Convert a search query into an embedding and retrieve semantically relevant passages.
6. **Generate:** For document questions, provide retrieved passages to the local language model through Ollama.
7. **Cite:** Present source information alongside the generated answer when available.

```text
PDF / TXT / Markdown
        |
        v
  Document Loader
        |
        v
    Text Chunker
        |
        v
 Sentence Transformers
        |
        v
 Persistent ChromaDB
        |
        +----------------------+
        |                      |
        v                      v
  Semantic Search       Ask Your Documents
                               |
                               v
                         Ollama LLM
                               |
                               v
                      Answer with Sources
```

## Requirements

Before running the project, install or prepare:

- Python 3.11
- Git
- Ollama
- The Ollama model `llama3.2:3b`

The project has been developed and tested in a Python virtual environment on Windows. Other operating systems may require small adjustments to the commands.

## Installation on Windows

### 1. Clone the repository

```powershell
git clone https://github.com/Yash-et/personal-knowledge-search-engine
cd personal-knowledge-search-engine
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, you can run the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Install dependencies

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify dependency consistency:

```powershell
python -m pip check
```

### 4. Install and prepare Ollama

Install Ollama from the official website:

https://ollama.com/download

Download the model used by the project:

```powershell
ollama pull llama3.2:3b
```

Confirm that the model is available:

```powershell
ollama list
```

Ensure the Ollama service is running before using the document-question-answering feature. Depending on your installation, Ollama may already be running in the background.

### 5. Start the application

From the repository root, with the virtual environment activated:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## Using the Application

### Document Library

Use the Document Library to inspect indexed sources and their passage counts.

- Review which documents have been indexed.
- Check whether the original local file is available.
- Remove a document from the index after confirming the action.
- Re-index a document after changing its contents.

Re-indexing replaces the existing passages for that source instead of intentionally adding a second copy of the same source's passages.

### Semantic Search

Enter a query to retrieve passages that are semantically relevant to the query. Review the returned text and source metadata to find the information you need.

### Ask Your Documents

Ask a question about your indexed documents. The application retrieves relevant passages and uses the local Ollama model to generate an answer based on the retrieved context.

Check the displayed source citations against the original documents when accuracy is important. Retrieval-based answers can still be incomplete or incorrect.

## Project Structure

```text
personal-knowledge-search-engine/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── documents/              # Local source documents
│   ├── vector_db/              # Application vector database
│   └── evaluation_db/          # Benchmark vector database
├── scripts/
│   └── evaluate_retrieval.py
├── src/
│   ├── evaluation/             # Retrieval metrics and evaluation
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── storage/
│   ├── indexing/
│   ├── search/
│   ├── rag/
│   └── ui_helpers.py
└── tests/
    ├── fixtures/
    │   ├── retrieval_eval_dataset.json
    │   └── retrieval_eval_dataset_challenging.json
    ├── test_retrieval_metrics.py
    └── test_retrieval_evaluator.py
```

The `data/` directories are used for local application data. They are not intended to contain personal documents or generated vector database files in the public Git repository.

## Running Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

For verbose test output:

```powershell
python -m pytest -v
```

Check installed dependency consistency:

```powershell
python -m pip check
```

The latest verified test run completed with **109 passing tests**. Test results may change as the project evolves.

## Retrieval Evaluation

The project includes a retrieval evaluation CLI for measuring semantic search performance against a labeled JSON dataset.

The evaluation code is located in `src/evaluation/`, and the command-line entry point is `scripts/evaluate_retrieval.py`.

### Metrics

- **Recall@k:** Measures the proportion of known relevant passages retrieved within the top *k* results.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant result is ranked across evaluation queries.

Recall@k does not penalize irrelevant passages returned alongside relevant ones, so it should not be interpreted as a precision metric.

### Run the evaluation

Activate the project's Python virtual environment and run these commands from the repository root:

```powershell
python scripts/evaluate_retrieval.py --dataset tests/fixtures/retrieval_eval_dataset.json --k 3
```

Run the more challenging benchmark:

```powershell
python scripts/evaluate_retrieval.py --dataset tests/fixtures/retrieval_eval_dataset_challenging.json --k 1
python scripts/evaluate_retrieval.py --dataset tests/fixtures/retrieval_eval_dataset_challenging.json --k 3
```

The challenging dataset contains 10 benchmark passages and 8 queries, including paraphrased questions and queries with multiple relevant passages.

### Challenging benchmark results

Observed results from the local embedding model:

| Metric | k = 1 | k = 3 |
|---|---:|---:|
| Recall@k | 0.812 | 1.000 |
| MRR | 1.000 | 1.000 |

At k = 1, some queries have multiple relevant passages, so retrieving only one of them results in less than 100% recall. At k = 3, all labeled relevant passages were retrieved within the top three results for this dataset.

These results are a small, synthetic benchmark intended to validate retrieval behavior and evaluation tooling. They do not establish general retrieval quality across arbitrary documents or real-world queries. Results may vary with changes to the dataset, embedding model, indexed passages, or retrieval implementation.

### Evaluation data

The benchmark datasets are stored in `tests/fixtures/`. Evaluation uses a separate persistent vector database under `data/evaluation_db/`, distinct from the application's document index.

When changing a benchmark dataset, ensure that the evaluation index contains only the intended benchmark passages. Persistent indexes may retain passages from earlier runs if they are not explicitly removed or reset.

## Local Data and Privacy

The project is designed around local document processing, local vector storage, and local language-model inference.

- Source documents are stored under `data/documents/`.
- The persistent vector database is stored under `data/vector_db/`.
- Ollama runs the configured language model locally.
- `.gitignore` excludes local documents, vector database files, virtual environments, and environment files from version control.

Local processing reduces the need to send document content to an external AI service, but it is not a guarantee of complete privacy. Review your environment, application configuration, and any future integrations before indexing sensitive information.

**Important:** Do not commit private documents, credentials, `.env` files, or generated vector database files.

## Current Status

The application currently supports document ingestion and indexing, persistent semantic search, RAG-based question answering, source citations, and Document Library management, including source removal and re-indexing.

Development is ongoing. Future improvements may include additional document formats, richer retrieval controls, improved evaluation of answer quality, and further UI refinements.

## License

A license has not yet been specified in this README. Add the appropriate license information once the repository's licensing choice is confirmed.
