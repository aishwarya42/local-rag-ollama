# local-rag-ollama
# RAG Document Summariser

Summarise PDF, DOCX and TXT documents using Retrieval-Augmented Generation (RAG). The project offers four ways to run it, from a hosted API to a fully offline setup:

| Entry point | Interface | Models run | Needs internet / token |
|---|---|---|---|
| `app.py` | Streamlit web app | Hugging Face Inference API | Yes, HF token |
| `app_ollama.py` | Streamlit web app | Local, through [Ollama](https://ollama.com) | No |
| `ollama_rag.py` | Command line | Local, through Ollama | No |
| `local-rag.py` | Script | Local, through Hugging Face `transformers` | Only to download models |

## Features

- Upload **PDF, DOCX or TXT** files (Streamlit apps) or pass a file path (command-line scripts)
- **Focused summaries**: ask about a topic, retrieve the most relevant chunks, and summarise them with `[chunk N]` citations
- **Whole-document summaries**: summarise a document too long for one prompt
  - *Map-reduce*: summarise sections, then merge the summaries in groups
  - *RAG themes* (`ollama_rag.py`, default): cluster chunk embeddings into themes, retrieve the most representative chunks per theme, and summarise those
- Adjustable chunk size, overlap and number of retrieved chunks
- Search with plain NumPy, with no FAISS in the local scripts

## How it works

```text
Document -> extract text -> split into overlapping chunks -> embed each chunk
                                                                   |
        question ------------------------------------> embed -> find nearest chunks
                                                                   |
                                          chunks + instructions -> language model -> summary
```

1. **Extract** text with `pypdf` or `python-docx`.
2. **Chunk** the text into overlapping pieces so ideas that cross a boundary are kept together.
3. **Embed** each chunk into a vector with an embedding model. Similar meanings give similar vectors.
4. **Retrieve** the chunks closest to your topic by cosine similarity (or, for whole-document RAG mode, the chunks closest to each theme).
5. **Generate** the summary from only the retrieved text, with instructions to use nothing else and to cite chunk numbers.

For whole-document summaries, retrieval alone is not enough, because a summary must cover the whole document and not just the parts closest to one question. That is why those modes use map-reduce or theme clustering.

## Quick start

Requires Python 3.10 or newer. Using a virtual environment is recommended:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Option A: Local with Ollama (recommended for private documents)

Nothing is sent over the internet.

```bash
# 1. Install Ollama from https://ollama.com and open the app
ollama pull nomic-embed-text      # embedding model
ollama pull llama3.2:3b           # chat model (use llama3.1:8b if you have 16 GB RAM or more)

# 2. Install dependencies
pip install -r requirements_ollama.txt

# 3a. Web app
streamlit run app_ollama.py

# 3b. or command line
python ollama_rag.py path/to/document.pdf
```

Command-line options for `ollama_rag.py`:

| Option | Default | Purpose |
|---|---|---|
| `--topic "..."` | none | Focused summary on a topic or question |
| `--method` | `rag` | Whole-document method: `rag` (themes plus retrieval) or `mapreduce` |
| `--themes` | automatic | Number of themes for `--method rag` |
| `--per-theme` | 4 | Chunks retrieved per theme |
| `--chat-model` | `llama3.2:3b` | Any chat model from `ollama list` |
| `--embed-model` | `nomic-embed-text` | Embedding model |
| `--chunk-size`, `--overlap` | 1500, 200 | Characters per chunk and overlap |
| `--top-k` | 5 | Chunks retrieved for `--topic` |
| `--out` | none | Also save the summary to a file |

### Option B: Hugging Face Inference API

Needs a [Hugging Face token](https://huggingface.co/settings/tokens) and available Inference Providers credits.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste your token into the sidebar and choose a chat model. Model availability and free usage limits change over time, so check the model page on the Hub if you get an error.

### Option C: Local with Hugging Face `transformers`

Runs a small model directly with PyTorch.

```bash
pip install transformers sentence-transformers pypdf torch accelerate numpy
python local-rag.py path/to/document.pdf
```

Models are downloaded from the Hub on first run (a few GB for the generator). Do not use `streamlit run` with this file, because it is a plain script.

## Project structure

```text
.
├── app.py                    # Streamlit app using the Hugging Face API
├── app_ollama.py             # Streamlit app using local Ollama models
├── ollama_rag.py             # Command-line summariser using Ollama
├── local-rag.py              # Script using local transformers models
├── requirements.txt          # Dependencies for app.py
├── requirements_ollama.txt   # Dependencies for app_ollama.py and ollama_rag.py
└── README.md
```

## Configuration

- **Hugging Face token**: for `app.py`, type it into the sidebar. For other scripts that download models, you can set `HF_TOKEN` as an environment variable or in a `.env` file (load it with `python-dotenv`). Never commit tokens.
- **Chunk size and overlap**: larger chunks keep more context but retrieve less precisely.
- **Model choice**: bigger models write better summaries but need more memory. A 1.5B to 3B model is fine for testing, and 7B to 8B is noticeably better.

## Privacy

| Setup | Where your document goes |
|---|---|
| Ollama (`app_ollama.py`, `ollama_rag.py`) | Stays on your machine |
| `local-rag.py` | Stays on your machine (models are downloaded once) |
| `app.py` | Every chunk is sent to Hugging Face and to whichever inference provider serves the chat model |

For confidential or regulated documents, use one of the fully local options.

## Troubleshooting

- **`Could not reach Ollama`**: open the Ollama app or run `ollama serve`, then retry.
- **No chat or embedding model in the dropdown**: run the `ollama pull` commands above.
- **`402 Payment Required` (Hugging Face API)**: the monthly free Inference Providers credits are used up. Wait for the reset, add credits, or switch to a local option.
- **`NoneType has no attribute 'strip'` or empty replies (Hugging Face API)**: reasoning models can spend the whole token budget on hidden thinking. Use a non-reasoning instruct model.
- **Segmentation fault on macOS**: this can happen when FAISS and PyTorch are loaded together. The local scripts avoid it by using NumPy for search.
- **`ModuleNotFoundError: docx`**: install `python-docx`, not `docx`.
- **Scanned PDFs give no text**: the file needs OCR first.
- **Slow whole-document summaries**: they make many model calls. Use `--method rag` in `ollama_rag.py`, a smaller document, or a faster model.

## Limitations

- Small local models give rough summaries and can omit or invent details. Check important points against the source, and use the `[chunk N]` citations in focused summaries to do so.
- Whole-document summaries lose detail at each merge or selection step.
- Text extraction from PDFs with complex layouts, tables or columns may be imperfect.

## License

Add a license of your choice (for example MIT) before publishing.
