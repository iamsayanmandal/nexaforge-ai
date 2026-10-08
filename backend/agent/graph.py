"""
LangGraph Agent — NexaForge
Parses natural-language instructions and routes to deterministic tool operations.

Graph:
  START → parse_instruction → route → [image_tool | pdf_tool | error_node] → END

Intelligence layer strictly handles parameter parsing and routing.
File bytes are never injected into the LLM context.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logger = logging.getLogger("nexaforge.agent")


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    instruction: str
    file_type: str          # "image" | "pdf"
    file_name: str
    operation: str          # parsed operation name
    params: dict            # parsed parameters
    error: Optional[str]


# ── LLM Setup ────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=512,
    )


# ── System Prompt ─────────────────────────────────────────────────────────────

_IMAGE_OPS = """
Operations for images:
- compress   -> params: {"target_kb": <int>}  (use when user asks to reduce, compress, shrink under a size)
- increase   -> params: {"target_kb": <int>}  (use when user asks to increase, pad, expand file size, e.g. "increase to 100kb", "increase image size")
- resize     -> params: {"width": <int>, "height": <int>} OR {"scale_percent": <int>}
- convert    -> params: {"format": "JPEG"|"PNG"|"WEBP"}
- rotate     -> params: {"degrees": 90|180|270}
"""

_PDF_OPS = """
Operations for PDFs:
- compress       -> params: {"target_kb": <int>}  (use when user asks to reduce, compress, shrink under a size)
- increase       -> params: {"target_kb": <int>}  (use when user asks to increase, expand, pad size, e.g. "increase pdf", "increase to 200kb")
- merge          -> params: {}   (requires multiple files)
- split          -> params: {"start_page": <int>, "end_page": <int>}
- extract_pages  -> params: {"pages": [<int>, ...]}
- images_to_pdf  -> params: {}   (requires multiple image files)
"""

_SYSTEM_PROMPT = """\
You are NexaForge's natural language instruction parser. Your ONLY job is to read \
a user's instruction and return a valid JSON object.

File type: {file_type}
File name: {file_name}

{ops_block}

Rules:
1. Return ONLY raw JSON. No markdown backticks, no explanatory text.
2. Select the single best matching operation.
3. MULTILINGUAL SUPPORT: The user can provide instructions in ANY language or dialect (English, Hinglish, Swedish, Hindi, Spanish, French, German, etc.). Parse the semantic intent regardless of language!
4. If user says "increase", "increase size", "make bigger", "expand", "size badhao", "öka storlek", use operation "increase".
5. If target_kb is not specified for "increase", default to 200.
6. If target_kb is not specified for "compress", default to 100 for images, 300 for PDFs.
7. Extract numeric values from the instruction (e.g., "50KB" -> 50, "50 kb se kam" -> 50).
8. If the instruction is completely unintelligible, return {{"operation": "unknown", "params": {{}}}}.

Examples in multiple languages:
- English: "compress this image under 50KB" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- Hinglish: "isko 50 kb se kam karo" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- Hinglish: "pdf ka size badhao 150kb tak" -> {{"operation": "increase", "params": {{"target_kb": 150}}}}
- Hinglish: "isko ghuma do 90 degree" -> {{"operation": "rotate", "params": {{"degrees": 90}}}}
- Swedish: "minska denna bild under 50kb" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- Swedish: "rotera 90 grader medurs" -> {{"operation": "rotate", "params": {{"degrees": 90}}}}
- Spanish: "reducir el tamaño a menos de 50kb" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- English: "make it 1200 by 800" -> {{"operation": "resize", "params": {{"width": 1200, "height": 800}}}}
- English: "convert to webp" -> {{"operation": "convert", "params": {{"format": "WEBP"}}}}
- English: "split pages 1 to 5" -> {{"operation": "split", "params": {{"start_page": 1, "end_page": 5}}}}
- English: "extract pages 2, 4, 6" -> {{"operation": "extract_pages", "params": {{"pages": [2, 4, 6]}}}}
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def parse_instruction(state: AgentState) -> AgentState:
    """LLM node: parse natural language instruction into operation and params."""
    file_type = state["file_type"]
    ops_block = _IMAGE_OPS if file_type == "image" else _PDF_OPS

    prompt = _SYSTEM_PROMPT.format(
        file_type=file_type,
        file_name=state["file_name"],
        ops_block=ops_block,
    )

    llm = _get_llm()
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["instruction"]),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Remove markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()

        parsed = json.loads(raw)
        operation = parsed.get("operation", "unknown")
        params = parsed.get("params", {})
        logger.info(f"Parsed instruction: operation={operation}, params={params}")
        return {**state, "operation": operation, "params": params}

    except Exception as e:
        logger.error(f"Instruction parsing failed: {e}")
        return {**state, "operation": "unknown", "params": {}, "error": str(e)}


def route(state: AgentState) -> Literal["image_tool", "pdf_tool", "error_node"]:
    """Conditional edge: route based on file type and parsed operation."""
    if state.get("error") or state.get("operation") == "unknown":
        return "error_node"
    if state["file_type"] == "image":
        return "image_tool"
    return "pdf_tool"


def image_tool(state: AgentState) -> AgentState:
    """Validate image operation is supported."""
    valid = {"compress", "increase", "resize", "convert", "rotate"}
    if state["operation"] not in valid:
        return {**state, "error": f"Unsupported image operation: '{state['operation']}'"}
    return state


def pdf_tool(state: AgentState) -> AgentState:
    """Validate PDF operation is supported."""
    valid = {"compress", "increase", "merge", "split", "extract_pages", "images_to_pdf"}
    if state["operation"] not in valid:
        return {**state, "error": f"Unsupported PDF operation: '{state['operation']}'"}
    return state


def error_node(state: AgentState) -> AgentState:
    """Return helpful error message without emojis."""
    if not state.get("error"):
        state = {
            **state,
            "error": (
                "Unable to determine the intended operation. "
                "Supported commands include: 'compress under 50KB', 'increase size to 150KB', "
                "'resize to 1200x800', 'convert to WEBP', 'rotate 90 degrees', "
                "'compress PDF under 300KB', 'increase PDF to 200KB', 'split pages 1 to 5', "
                "or 'extract pages 2, 4, 6'."
            ),
        }
    return state


# ── Build Graph ───────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("parse_instruction", parse_instruction)
    g.add_node("image_tool", image_tool)
    g.add_node("pdf_tool", pdf_tool)
    g.add_node("error_node", error_node)

    g.set_entry_point("parse_instruction")
    g.add_conditional_edges("parse_instruction", route, {
        "image_tool": "image_tool",
        "pdf_tool":   "pdf_tool",
        "error_node": "error_node",
    })
    g.add_edge("image_tool",  END)
    g.add_edge("pdf_tool",    END)
    g.add_edge("error_node",  END)

    return g.compile()


# Singleton graph instance
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(instruction: str, file_type: str, file_name: str) -> dict:
    """
    Run the LangGraph agent.
    Returns {"operation": str, "params": dict, "error": str | None}
    """
    graph = get_graph()
    result = graph.invoke({
        "instruction": instruction,
        "file_type": file_type,
        "file_name": file_name,
        "operation": "",
        "params": {},
        "error": None,
    })
    return {
        "operation": result.get("operation", "unknown"),
        "params":    result.get("params", {}),
        "error":     result.get("error"),
    }
