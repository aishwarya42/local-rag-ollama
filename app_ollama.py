import numpy as np
import streamlit as st
import ollama
from pypdf import PdfReader
from docx import Document

st.set_page_config(page_title="Local RAG Summariser", page_icon="📄", layout="wide")
st.title("📄 Local RAG Document Summariser (Ollama)")
st.caption("Everything runs on your Mac through Ollama. No token, no credits, nothing leaves your machine.")

EMBED_HINTS = ("embed", "bge", "minilm", "arctic", "gte", "e5")


# ---------------- Helpers ----------------
def model_name(m):
    for key in ("model", "name"):
        try:
            v = m[key]
            if v:
                return v
        except (KeyError, TypeError, AttributeError):
            pass
    return str(m)


def installed_models():
    resp = ollama.list()
    return sorted(model_name(m) for m in resp["models"])


def is_embed(name):
    return any(h in name.lower() for h in EMBED_HINTS)


def extract_text(uploaded) -> str:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(uploaded).pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(uploaded).paragraphs)
    return uploaded.read().decode("utf-8", errors="ignore")


def front_matter(uploaded, text, chars=2000):
    """Title page info that retrieval would miss: PDF metadata (title/author) plus the start of the text."""
    lines = []
    if uploaded.name.lower().endswith(".pdf"):
        try:
            meta = PdfReader(uploaded).metadata
            if meta:
                if meta.title:
                    lines.append(f"Title (file metadata): {meta.title}")
                if meta.author:
                    lines.append(f"Author (file metadata): {meta.author}")
        except Exception:
            pass
    lines.append("Start of document:\n" + text[:chars].strip())
    return "\n".join(lines)


def chunk_text(text, size, overlap):
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def embed(texts, model, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        res = ollama.embed(model=model, input=texts[i:i + batch])
        out.append(np.asarray(res["embeddings"], dtype="float32"))
    v = np.vstack(out)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12  # unit length -> dot product = cosine similarity
    return v


def llm(prompt, model, max_tokens=500):
    r = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise summariser. Use only the text provided."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.2, "num_ctx": 8192, "num_predict": max_tokens},
    )
    content = (r["message"]["content"] or "").strip()
    if "</think>" in content:  # reasoning models may include a thinking block
        content = content.split("</think>", 1)[1].strip()
    return content


def kmeans(x, k, iters=50, seed=0):
    """Spherical k-means in plain NumPy (vectors are unit length, so dot product = cosine similarity)."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):  # k-means++ style start: pick far-apart points
        d = np.clip(1 - np.max(x @ np.array(centers).T, axis=1), 0, None) ** 2
        p = d / d.sum() if d.sum() > 0 else None
        centers.append(x[rng.choice(len(x), p=p)])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmax(x @ centers.T, axis=1)
        new = np.array([x[labels == c].mean(axis=0) if (labels == c).any() else centers[c] for c in range(k)])
        new /= np.linalg.norm(new, axis=1, keepdims=True) + 1e-12
        if np.allclose(new, centers):
            break
        centers = new
    return np.argmax(x @ centers.T, axis=1), centers


def pick_theme_chunks(vecs, themes=0, per_theme=4):
    """Cluster the chunk embeddings into themes, then RETRIEVE the chunks closest to each theme's centre.
    Returns one list of chunk numbers per theme, themes ordered by where they first appear."""
    n = len(vecs)
    k = themes or int(np.clip(round(n / 15), 4, 12))
    k = max(1, min(k, n))
    labels, centers = kmeans(vecs, k)
    groups = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sims = vecs[idx] @ centers[c]
        groups.append(sorted(int(i) for i in idx[np.argsort(-sims)[:per_theme]]))
    groups.sort(key=lambda g: g[0])
    return groups


# ---------------- Connect to Ollama ----------------
try:
    names = installed_models()
except Exception as e:
    st.error(
        "Could not reach Ollama. Open the Ollama app (menu-bar icon) or run `ollama serve` in a terminal, "
        f"then refresh this page.\n\nDetails: {e}"
    )
    st.stop()

chat_models = [n for n in names if not is_embed(n)]
embed_models = [n for n in names if is_embed(n)]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    if not chat_models:
        st.error("No chat model installed. Run: `ollama pull llama3.2:3b`")
        st.stop()
    if not embed_models:
        st.error("No embedding model installed. Run: `ollama pull nomic-embed-text`")
        st.stop()
    chat_model = st.selectbox("Chat model", chat_models)
    embed_model = st.selectbox("Embedding model", embed_models)
    chunk_size = st.slider("Chunk size (characters)", 500, 3000, 1500, 100)
    overlap = st.slider("Chunk overlap (characters)", 0, 500, 200, 50)
    top_k = st.slider("Chunks to retrieve (topic summary)", 2, 10, 5)

doc_prefix = "search_document: " if "nomic" in embed_model.lower() else ""
query_prefix = "search_query: " if "nomic" in embed_model.lower() else ""

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
if uploaded is None:
    st.info("Upload a PDF, DOCX or TXT file to get started.")
    st.stop()

# ---------------- Build index (once per file + settings) ----------------
doc_key = (uploaded.name, uploaded.size, chunk_size, overlap, embed_model)
if st.session_state.get("doc_key") != doc_key:
    with st.spinner("Reading and indexing document..."):
        text = extract_text(uploaded)
        if not text.strip():
            st.error("No text could be extracted. If this is a scanned PDF, it needs OCR first.")
            st.stop()
        chunks = chunk_text(text, chunk_size, overlap)
        try:
            vecs = embed([doc_prefix + c for c in chunks], embed_model)
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()
        st.session_state.update(doc_key=doc_key, chunks=chunks, vecs=vecs, text=text,
                                front=front_matter(uploaded, text))

chunks = st.session_state["chunks"]
vecs = st.session_state["vecs"]
st.success(f"Indexed **{uploaded.name}**: {len(st.session_state['text']):,} characters in {len(chunks)} chunks.")

# ---------------- Summarise ----------------
mode = st.radio(
    "Summary type",
    ["Whole document (RAG themes)", "Whole document (map-reduce)", "Focused on a topic"],
    horizontal=True,
)

if mode == "Focused on a topic":
    query = st.text_input("Topic or question", placeholder="e.g. key risks and mitigations")
    if st.button("Summarise topic", type="primary") and query:
        with st.spinner("Retrieving and summarising..."):
            q = embed([query_prefix + query], embed_model)[0]
            hits = np.argsort(-(vecs @ q))[:min(top_k, len(chunks))]
            context = "\n\n".join(f"[chunk {i}]\n{chunks[i]}" for i in hits)
            answer = llm(
                "Answer the topic below using ONLY the front matter and the excerpts. Cite chunk numbers like "
                "[chunk 3], and cite the front matter as [front matter]. If neither covers it, say so plainly "
                "and do not guess.\n\n"
                f"Topic: {query}\n\nFront matter (title page and file metadata):\n{st.session_state['front']}"
                f"\n\nExcerpts:\n{context}",
                chat_model,
            )
        st.subheader("Summary")
        st.write(answer)
        with st.expander("Source chunks used"):
            for i in hits:
                st.markdown(f"**chunk {i}**")
                st.text(chunks[i])
elif mode == "Whole document (RAG themes)":
    st.caption(
        "Groups similar chunks into themes, retrieves the most representative chunks for each theme, and "
        "summarises only those. Needs far fewer model calls than map-reduce, but reads less of the text."
    )
    c1, c2 = st.columns(2)
    themes = c1.slider("Number of themes (0 = automatic)", 0, 12, 0)
    per_theme = c2.slider("Chunks retrieved per theme", 2, 8, 4)
    if st.button("Summarise document", type="primary", key="rag_summary"):
        groups = pick_theme_chunks(vecs, themes, per_theme)
        theme_summaries = []
        progress = st.progress(0.0, text="Summarising themes...")
        for n, g in enumerate(groups, 1):
            context = "\n\n".join(f"[chunk {i}]\n{chunks[i]}" for i in g)
            theme_summaries.append(
                llm("These excerpts all belong to one theme of a document. Give the theme a short heading, then "
                    "summarise it in 3-4 bullet points using ONLY the excerpts. Cite chunk numbers like [chunk 3].\n\n"
                    + context, chat_model, 300)
            )
            progress.progress(n / len(groups), text=f"Summarised theme {n} of {len(groups)}")
        with st.spinner("Combining..."):
            final = llm(
                "Combine these theme summaries into one clear summary of the whole document: a short overview, "
                "then key points. Keep the [chunk N] citations.\n\n" + "\n\n".join(theme_summaries),
                chat_model, 700,
            )
        progress.empty()
        st.subheader("Summary")
        st.write(final)
        st.download_button("Download summary (.txt)", final, file_name="summary.txt", on_click="ignore")
        with st.expander("Theme summaries and the chunks they came from"):
            for n, (g, s) in enumerate(zip(groups, theme_summaries), 1):
                st.markdown(f"**Theme {n}** (chunks {', '.join(str(i) for i in g)})")
                st.write(s)
else:
    batch = st.slider("Chunks per section summary", 1, 8, 3)
    if st.button("Summarise document", type="primary"):
        starts = list(range(0, len(chunks), batch))
        partials = []
        progress = st.progress(0.0, text="Summarising sections...")
        for n, i in enumerate(starts):
            partials.append(
                llm("Summarise this section in 4-5 bullet points:\n\n" + "\n\n".join(chunks[i:i + batch]),
                    chat_model, 300)
            )
            progress.progress((n + 1) / len(starts), text=f"Summarised section {n + 1} of {len(starts)}")

        section_summaries = list(partials)
        group = 6
        with st.spinner("Combining..."):
            # Merge in groups so the final prompt never outgrows a small model's context window.
            while len(partials) > 1:
                partials = [
                    llm("Merge these summaries into one shorter summary, keeping the key ideas:\n\n"
                        + "\n\n".join(partials[i:i + group]), chat_model, 400)
                    for i in range(0, len(partials), group)
                ]
            final = llm(
                "Rewrite this as a clear summary with a short overview followed by key points:\n\n" + partials[0],
                chat_model, 600,
            )
        progress.empty()
        st.subheader("Summary")
        st.write(final)
        st.download_button("Download summary (.txt)", final, file_name="summary.txt", on_click="ignore")
        with st.expander("Section summaries"):
            for n, p in enumerate(section_summaries, 1):
                st.markdown(f"**Section {n}**")
                st.write(p)
