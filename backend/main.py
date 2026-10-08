"""
NexaForge — AI Media Processing Engine
FastAPI Application Entry Point
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from routers import image, pdf, ai

load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nexaforge")

# ── LangSmith Observability (auto-instruments all LangChain/LangGraph calls) ─
# Just set these 3 env vars in Railway → full trace dashboard, zero extra code
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "nexaforge"))


# ── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NexaForge starting up ...")
    # Pre-warm RAG assistant (loads embedding model once — ~120MB RAM)
    try:
        from rag.assistant import get_assistant
        get_assistant()
        logger.info("RAG assistant initialized")
    except Exception as e:
        logger.warning(f"RAG assistant warm-up failed (will retry on first request): {e}")
    yield
    logger.info("NexaForge shutting down")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NexaForge — AI Media Processing Engine",
    description=(
        "AI-powered image and PDF processing. "
        "Describe what you want in plain English — NexaForge does the rest."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow all origins (Cloudflare Workers, Pages, Custom Domains, Dev) ──
ALLOWED_ORIGINS = [
    "https://nexaforge.sayanmandal.in",
    "https://nexaforge-ai.pages.dev",
    "https://nexaforge-ai.hakerworld309.workers.dev",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",   # VS Code Live Server
    "http://127.0.0.1:3000",
    "null",                     # file:// opened locally
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(image.router, prefix="/api/image", tags=["Image Processing"])
app.include_router(pdf.router,   prefix="/api/pdf",   tags=["PDF Processing"])
app.include_router(ai.router,    prefix="/api",        tags=["AI Agent & Assistant"])


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "NexaForge",
        "version": "1.0.0",
        "tracing": os.getenv("LANGCHAIN_TRACING_V2", "false"),
    }


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"message": "NexaForge API — visit /docs for Swagger UI"})
