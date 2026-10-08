"""
RAG Assistant — NexaForge
ChromaDB + sentence-transformers (all-MiniLM-L6-v2) + LangChain RetrievalQA.
Answers questions about NexaForge features using the knowledge base docs.

Resume skills demonstrated:
  - RAG pipeline
  - Embeddings (sentence-transformers)
  - Vector Search (ChromaDB HNSW index)
  - LangChain (RetrievalQA chain)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

os.environ.setdefault("ANONYMOUS_TELEMETRY", "False")

logger = logging.getLogger("nexaforge.rag")

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

_SYSTEM_PROMPT = """\
You are NexaForge Assistant, a specialized AI for the NexaForge AI Media Processing Engine.

CRITICAL INSTRUCTIONS:
1. MULTILINGUAL RESPONSE: Automatically detect the language and dialect of the user's question (such as Hinglish, English, Swedish, Hindi, Spanish, French, German, etc.) and reply in the EXACT SAME language and conversational tone.
   - If the user asks in Hinglish (e.g. "tum kya kr sakte ho?"), respond in natural, friendly Hinglish explaining what NexaForge can do.
   - If the user asks in Swedish (e.g. "vad kan du göra?"), respond in Swedish.
   - If the user asks in English, respond in English.
2. NO RAW MARKDOWN HEADINGS OR UNFORMATTED SYMBOLS: Do NOT output raw heading symbols like '##', '###', or markdown separators '***'. Use clean, clear paragraphs, neat bullet points (- item), or plain readable text.
3. CONTEXT USAGE: Answer accurately based on the context below. If a feature or question is outside NexaForge, politely inform the user in their language.

Context:
{context}

Question: {question}

Answer:"""


def _load_documents() -> list[dict]:
    """Load all markdown files from the knowledge base."""
    docs = []
    if not KNOWLEDGE_DIR.exists():
        logger.warning(f"Knowledge base directory not found: {KNOWLEDGE_DIR}")
        return docs

    for md_file in KNOWLEDGE_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        # Split into chunks at double-newlines (≤500 chars each)
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]
        for chunk in chunks:
            docs.append({"text": chunk, "source": md_file.stem})

    logger.info(f"Loaded {len(docs)} chunks from {KNOWLEDGE_DIR}")
    return docs


@lru_cache(maxsize=1)
def get_assistant() -> RetrievalQA:
    """
    Build and cache the RAG chain.
    Called once at startup (lru_cache ensures single load).
    """
    logger.info("Building RAG assistant …")

    # 1. Embeddings — all-MiniLM-L6-v2 (~80MB RAM, downloads on first run)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. Load knowledge base documents
    docs = _load_documents()
    if not docs:
        raise RuntimeError("Knowledge base is empty — check backend/rag/knowledge/")

    texts  = [d["text"]   for d in docs]
    metas  = [{"source": d["source"]} for d in docs]

    # 3. ChromaDB in-memory vector store
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metas,
        collection_name="nexaforge_kb",
    )

    # 4. Retriever — top-3 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # 5. Groq LLM for answer generation
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    llm = ChatGroq(
        model=model_name,
        temperature=0.2,
        max_tokens=512,
    )

    # 6. LangChain RetrievalQA chain
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=_SYSTEM_PROMPT,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    logger.info("RAG assistant built successfully")
    return chain


def _clean_reply_text(text: str) -> str:
    """Strip raw markdown heading hashes and dividers that clutter bot replies."""
    import re
    # Remove raw markdown heading hashes at beginning of lines
    text = re.sub(r'^(?:#{1,6}\s*)(.*)$', r'\1', text, flags=re.MULTILINE)
    # Remove standalone *** or --- separator lines
    text = re.sub(r'^\s*[\*\-_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Clean redundant blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def ask_assistant(question: str) -> dict:
    """
    Query the RAG assistant.
    Returns {"answer": str, "sources": list[str]}
    """
    chain = get_assistant()
    try:
        result = chain.invoke({"query": question})
        raw_ans = result["result"]
        clean_ans = _clean_reply_text(raw_ans)
        sources = list({
            doc.metadata.get("source", "")
            for doc in result.get("source_documents", [])
        })
        return {
            "answer": clean_ans,
            "sources": sources,
        }
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return {
            "answer": "Sorry, I ran into an error. Please try again.",
            "sources": [],
        }
