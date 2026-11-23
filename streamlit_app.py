"""arxiv-rag web demo. Hosted on Streamlit Community Cloud; run locally with `arxiv-rag ui`.

Thin UI layer: everything it calls lives in src/arxiv_rag (demo.py, ui.py) and is tested.
"""

import os
from pathlib import Path

import streamlit as st

# Demo defaults (free hosting: little RAM, no GPU). Set before arxiv_rag reads its settings.
os.environ.setdefault("EMBED_BACKEND", "fastembed")
os.environ.setdefault("DATA_DIR", "demo_data")
try:  # Streamlit Cloud secrets -> environment (the LLM library reads keys from the environment)
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except FileNotFoundError:
    pass

from arxiv_rag.config import Settings  # noqa: E402
from arxiv_rag.demo import build_demo  # noqa: E402
from arxiv_rag.ui import (  # noqa: E402
    EXAMPLES,
    REPO_URL,
    RETRIEVER_CHOICES,
    answer_question,
    results_markdown,
)

ROOT = Path(__file__).parent

st.set_page_config(page_title="arxiv-rag: cited answers from AI papers", page_icon="📄")


@st.cache_resource(show_spinner=False)
def load_demo():
    with st.status("Preparing the demo (first visit after a restart takes ~5 min)…") as status:
        demo = build_demo(Settings(), ROOT / "eval" / "paper_ids.txt", progress=status.write)
        status.update(label=f"Ready: {len(demo.chunks):,} passages indexed", state="complete")
    return demo


st.title("arxiv-rag")
st.markdown(
    "Ask about **retrieval-augmented generation and LLM agents**. Answers come only from "
    "49 arXiv papers and cite the passages they use. "
    f"[Code, tests and evaluation on GitHub]({REPO_URL})."
)

ask_tab, results_tab = st.tabs(["Ask", "How well does it work?"])

with ask_tab:
    demo = load_demo()
    labels = {label: value for label, value in RETRIEVER_CHOICES}
    with st.form("ask"):
        question = st.text_input("Question", placeholder=EXAMPLES[0])
        retriever = st.selectbox("Retriever", list(labels))
        submitted = st.form_submit_button("Ask", type="primary")
    st.caption("Try: " + " · ".join(f"*{q}*" for q in EXAMPLES))

    if submitted:
        with st.spinner("Retrieving passages and writing a cited answer…"):
            answer_md, sources_md = answer_question(question, labels[retriever], demo.pipeline)
        st.markdown(answer_md, unsafe_allow_html=True)
        if sources_md:
            st.subheader("Sources")
            st.markdown(sources_md, unsafe_allow_html=True)

with results_tab:
    st.markdown(results_markdown(ROOT / "README.md"))
