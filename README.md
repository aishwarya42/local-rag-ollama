# Local RAG Document Summariser (Streamlit + Ollama)

A web app that summarises PDF, DOCX and TXT documents using Retrieval-Augmented Generation (RAG). Everything runs on your own machine through [Ollama](https://ollama.com), so there are no API keys, no usage credits, and your documents never leave your computer.

## Features

- **Upload** a PDF, DOCX or TXT file in the browser
- **Whole-document summary**: summarises the document section by section, then merges the section summaries into one overview with key points
- **Focused summary**: type a topic or question, and the app retrieves the most relevant chunks and summarises only those, with `[chunk N]` citations and a viewer for the source text
- **Model picker**: chat and embedding models are read from your Ollama install, with no model names to type
- **Adjustable settings**: chunk size, overlap and number of retrieved chunks
- **Download** the final summary as a `.txt` file
- **Private by design**: all processing is local

## How it works

```text
Upload -> extract text -> split into overlapping chunks -> embed each chunk (Ollama)
                                                                   |
 topic ------------------------------------> embed -> nearest chunks (cosine similarity)
                                                                   |
                                   chunks + instructions -> chat model (Ollama) -> summary
```

1. The text is extracted with `pypdf` or `python-docx`.
2. It is split into overlapping chunks, so ideas that cross a boundary stay together.
3. An embedding model (default suggestion: `nomic-embed-text`) turns each chunk into a vector. Chunks with similar meaning get similar vectors.
4. For a **focused summary**, your topic is embedded the same way, and the closest chunks are found with a NumPy dot product. The chat model summarises only those chunks and is told to use nothing else.
5. For a **whole-document summary**, retrieval is not enough, because a summary has to cover the whole text. The app uses **map-reduce** instead: it summarises small groups of chunks, then merges those summaries in groups until one remains.

The Streamlit page is only the interface. The Python script talks to the Ollama server on your machine (`http://localhost:11434`) to create embeddings and generate text.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running
- Enough memory for your chosen model (roughly: 8 GB RAM for a 3B model, 16 GB or more for an 7B to 8B model)

## Setup

```bash
# 1. Get the code
git clone <your-repo-url>
cd <your-repo-folder>

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements_ollama.txt

# 4. Start Ollama (open the Ollama app, or run: ollama serve)
#    then download one embedding model and one chat model
ollama pull nomic-embed-text       # embedding model (small)
ollama pull llama3.2:3b            # chat model for ~8 GB RAM
# ollama pull llama3.1:8b          # better quality if you have 16 GB RAM or more
```

Check that the models are installed:

```bash
ollama list
```

## Run

```bash
streamlit run app_ollama.py
```

Your browser opens at `http://localhost:8501`. If port 8501 is busy, use `streamlit run app_ollama.py --server.port 8502`.

## Usage

1. Pick a **chat model** and an **embedding model** in the sidebar.
2. Upload a document and wait for the "Indexed ..." message.
3. Choose a summary type:
   - **Whole document**: choose how many chunks go into each section summary, then click *Summarise document*. A progress bar shows each section as it finishes, and the section summaries can be opened in an expander.
   - **Focused on a topic**: type a question and click *Summarise topic*. Open *Source chunks used* to check what the answer is based on.
4. Download the summary with the *Download summary* button.

Whole-document summaries make many model calls (a full book can mean around 45), so they take a while on a laptop. Test with a short file first.

## Settings

| Setting | Range | Effect |
|---|---|---|
| Chunk size | 500 to 3000 characters | Larger chunks keep more context but retrieve less precisely |
| Chunk overlap | 0 to 500 characters | Repeated text between neighbouring chunks, so ideas are not cut in half |
| Chunks to retrieve | 2 to 10 | How many chunks a focused summary reads |
| Chunks per section | 1 to 8 | How many chunks go into each section summary (whole document) |

The app sets Ollama's context window to 8,192 tokens for each request, because the default can be too small and silently cut long prompts.

## Project structure

```text
.
├── app_ollama.py              # the Streamlit app
├── requirements_ollama.txt    # streamlit, ollama, pypdf, python-docx, numpy
└── README.md
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not reach Ollama` | Open the Ollama app or run `ollama serve`, then refresh the page. Check with `curl http://localhost:11434` (it should say "Ollama is running"). |
| `No embedding model installed` | Run `ollama pull nomic-embed-text`, then refresh the page. |
| `No chat model installed` | Run `ollama pull llama3.2:3b`, then refresh the page. |
| A model I installed does not appear | The app sorts models into chat and embedding lists by name. Embedding models are recognised by `embed`, `bge`, `minilm`, `arctic`, `gte` or `e5` in the name. Everything else is treated as a chat model. |
| Very slow or empty replies | Reasoning models (such as `qwen3` or `deepseek-r1`) spend time and tokens on hidden thinking. Use a standard instruct model such as Llama 3.2 or Qwen2.5. |
| "No text could be extracted" | The PDF is probably scanned images. It needs OCR first. |
| `ModuleNotFoundError: docx` | Install `python-docx` (not `docx`). |
| Slow first request after a pause | Ollama unloads idle models, and the first call reloads them. |

## Limitations

- Small local models give rough summaries and can leave out or invent details. Use the source-chunk viewer in focused mode to check claims, and verify anything important against the original.
- Whole-document summaries lose detail at each merge step.
- PDFs with complex layouts, columns or tables may extract imperfectly.
- The index is rebuilt each time you upload a file or change the chunk settings or embedding model. Nothing is saved to disk.

## License

Add a license of your choice (for example MIT) before publishing.
