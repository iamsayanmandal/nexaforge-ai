"""
LangGraph Agent — NexaForge
Parses natural-language instructions, validates parameter bounds via Pydantic,
and provides a cyclic self-repair loop for out-of-bounds or malformed requests.

Cyclic Graph:
  START → parse_instruction → route → [image_tool | pdf_tool]
                                          ↓ (if invalid params, retry_count < 2)
                                       repair_node ──┐
                                          ↑          │ (cycle back)
                                          └──────────┘
                                          ↓ (if valid or max retries reached)
                                       [END | error_node]
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
from pydantic import BaseModel, Field, field_validator, model_validator
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
    fallback_used: bool
    retry_count: int


# ── Pydantic Schemas for Parameter Validation ─────────────────────────────────

class CompressParams(BaseModel):
    target_kb: int = Field(default=100, ge=1, le=50000)

class IncreaseParams(BaseModel):
    target_kb: int = Field(default=200, ge=1, le=50000)

class ResizeParams(BaseModel):
    width: Optional[int] = Field(default=None, ge=1, le=10000)
    height: Optional[int] = Field(default=None, ge=1, le=10000)
    scale_percent: Optional[int] = Field(default=None, ge=1, le=500)

    @model_validator(mode="after")
    def check_dimensions(self):
        if not self.width and not self.height and not self.scale_percent:
            raise ValueError("Provide at least one of width, height, scale_percent")
        return self

class ConvertParams(BaseModel):
    format: Literal["JPEG", "PNG", "WEBP"] = "PNG"

class RotateParams(BaseModel):
    degrees: Literal[90, 180, 270] = 90

class SplitParams(BaseModel):
    start_page: int = Field(default=1, ge=1)
    end_page: Optional[int] = Field(default=None, ge=1)

class ExtractPagesParams(BaseModel):
    pages: list[int] = Field(default_factory=lambda: [1])

    @field_validator("pages")
    @classmethod
    def validate_pages(cls, v):
        if not v:
            raise ValueError("Pages list cannot be empty")
        if len(v) > 1000:
            raise ValueError("Too many pages requested (maximum 1000)")
        return [max(1, int(p)) for p in v]


def _validate_operation_params(file_type: str, operation: str, params: dict) -> tuple[dict, str | None]:
    """Validate and normalize parameters using Pydantic models."""
    try:
        if file_type == "image":
            if operation == "compress":
                p = CompressParams(**params)
                return p.model_dump(), None
            elif operation == "increase":
                p = IncreaseParams(**params)
                return p.model_dump(), None
            elif operation == "resize":
                p = ResizeParams(**params)
                return {k: v for k, v in p.model_dump().items() if v is not None}, None
            elif operation == "convert":
                fmt = str(params.get("format", "PNG")).upper().replace("JPG", "JPEG")
                p = ConvertParams(format=fmt)  # type: ignore
                return p.model_dump(), None
            elif operation == "rotate":
                deg = int(params.get("degrees", 90))
                p = RotateParams(degrees=deg)  # type: ignore
                return p.model_dump(), None
        else:
            # PDF operations
            if operation == "compress":
                p = CompressParams(target_kb=params.get("target_kb", 300))
                return p.model_dump(), None
            elif operation == "increase":
                p = IncreaseParams(target_kb=params.get("target_kb", 200))
                return p.model_dump(), None
            elif operation == "split":
                p = SplitParams(**params)
                return p.model_dump(), None
            elif operation == "extract_pages":
                p = ExtractPagesParams(**params)
                return p.model_dump(), None
            elif operation in ("merge", "images_to_pdf"):
                return {}, None

        return params, None
    except Exception as e:
        return params, f"Parameter validation error for '{operation}': {e}"


# ── LLM Singleton Setup ───────────────────────────────────────────────────────

_LLM_INSTANCE: Optional[ChatGroq] = None

def _get_llm() -> ChatGroq:
    """Build ChatGroq client once and cache instance across requests."""
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        _LLM_INSTANCE = ChatGroq(
            model=model_name,
            temperature=0,
            max_tokens=1024,      # Sufficient headroom for reasoning models without payload truncation
            request_timeout=20,   # Prevent stalling
            max_retries=2,
        )
    return _LLM_INSTANCE


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
7. Extract numeric values from the instruction (e.g., "50KB" -> 50, "1MB" -> 1000, "50 kb se kam" -> 50).
8. If the instruction is completely unintelligible, return {{"operation": "unknown", "params": {{}}}}.

Examples in multiple languages:
- English: "compress this image under 50KB" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- English: "under 1MB" -> {{"operation": "compress", "params": {{"target_kb": 1000}}}}
- Hinglish: "isko 50 kb se kam karo" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- Hinglish: "pdf ka size badhao 150kb tak" -> {{"operation": "increase", "params": {{"target_kb": 150}}}}
- Hinglish: "isko ghuma do 90 degree" -> {{"operation": "rotate", "params": {{"degrees": 90}}}}
- Swedish: "minska denna bild under 50kb" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- Swedish: "rotera 90 grader medurs" -> {{"operation": "rotate", "params": {{"degrees": 90}}}}
- Spanish: "reducir el tamaño a menos de 50kb" -> {{"operation": "compress", "params": {{"target_kb": 50}}}}
- English: "make it 1200 by 800" -> {{"operation": "resize", "params": {{"width": 1200, "height": 800}}}}
- English: "scale by 50%" -> {{"operation": "resize", "params": {{"scale_percent": 50}}}}
- English: "convert to webp" -> {{"operation": "convert", "params": {{"format": "WEBP"}}}}
- English: "split pages 1 to 5" -> {{"operation": "split", "params": {{"start_page": 1, "end_page": 5}}}}
- English: "extract pages 2-4" -> {{"operation": "extract_pages", "params": {{"pages": [2, 3, 4]}}}}
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _heuristic_fallback_parse(instruction: str, file_type: str) -> dict | None:
    """
    Deterministic fallback parser for CI environments, mock keys, and network timeouts.
    Handles KB, MB (including decimals like 0.5 MB), page intervals (2-4 -> [2,3,4]), and scaling.
    """
    text = instruction.lower().strip()

    # Match MB / M or KB / K with optional decimal support (e.g. 0.5 mb, 1.5mb, 100kb)
    mb_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mb|m\b)", text)
    kb_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kb|k\b)", text)

    if mb_match:
        kb_val = int(float(mb_match.group(1)) * 1000)
    elif kb_match:
        kb_val = int(float(kb_match.group(1)))
    else:
        kb_val = None

    # Increase / expand
    if any(k in text for k in ["increase", "expand", "badhao", "upscale"]):
        return {"operation": "increase", "params": {"target_kb": kb_val or 200}}

    # Compress / reduce
    if any(k in text for k in ["compress", "reduce", "minska", "kam", "chhota", "smaller", "under", "below"]):
        default_kb = 100 if file_type == "image" else 300
        return {"operation": "compress", "params": {"target_kb": kb_val or default_kb}}

    # Rotate
    if any(k in text for k in ["rotate", "rotera", "ghuma", "flip"]):
        deg = 90
        if "180" in text:
            deg = 180
        elif "270" in text:
            deg = 270
        return {"operation": "rotate", "params": {"degrees": deg}}

    # Convert
    if any(k in text for k in ["convert", "badlo", "format"]):
        fmt = "WEBP"
        if "png" in text:
            fmt = "PNG"
        elif "jpg" in text or "jpeg" in text:
            fmt = "JPEG"
        return {"operation": "convert", "params": {"format": fmt}}

    # Resize (exact dimensions or percentage scaling)
    if any(k in text for k in ["resize", "dimension", "width", "height", "scale", "%"]):
        pct_match = re.search(r"(\d+)\s*%", text)
        if pct_match:
            return {"operation": "resize", "params": {"scale_percent": int(pct_match.group(1))}}
        dim_match = re.search(r"(\d+)\s*(?:x|by)\s*(\d+)", text)
        if dim_match:
            return {"operation": "resize", "params": {"width": int(dim_match.group(1)), "height": int(dim_match.group(2))}}

    # Split (PDF)
    if file_type == "pdf" and "split" in text:
        nums = re.findall(r"\d+", text)
        if len(nums) >= 2:
            return {"operation": "split", "params": {"start_page": int(nums[0]), "end_page": int(nums[1])}}
        elif len(nums) == 1:
            return {"operation": "split", "params": {"start_page": int(nums[0]), "end_page": None}}

    # Extract Pages (PDF) — support discrete numbers and ranges e.g. "extract pages 2-4" -> [2, 3, 4]
    if file_type == "pdf" and "extract" in text:
        range_match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", text)
        if range_match:
            start_p = int(range_match.group(1))
            end_p = int(range_match.group(2))
            if start_p <= end_p and (end_p - start_p) < 1000:
                return {"operation": "extract_pages", "params": {"pages": list(range(start_p, end_p + 1))}}

        nums = [int(n) for n in re.findall(r"\d+", text)]
        if nums:
            return {"operation": "extract_pages", "params": {"pages": nums}}

    return None


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
        return {**state, "operation": operation, "params": params, "fallback_used": False}

    except Exception as e:
        logger.warning(f"LLM parsing error ({e}), executing deterministic fallback...")
        fallback = _heuristic_fallback_parse(state["instruction"], file_type)
        if fallback:
            logger.info(f"Fallback parse successful: operation={fallback['operation']}, params={fallback['params']}")
            return {
                **state,
                "operation": fallback["operation"],
                "params": fallback["params"],
                "error": None,
                "fallback_used": True,
            }
        logger.error(f"Instruction parsing completely failed: {e}")
        return {
            **state,
            "operation": "unknown",
            "params": {},
            "error": str(e),
            "fallback_used": True,
        }


def route(state: AgentState) -> Literal["image_tool", "pdf_tool", "error_node"]:
    """Conditional edge: route based on file type and parsed operation."""
    if state.get("error") or state.get("operation") == "unknown":
        return "error_node"
    if state["file_type"] == "image":
        return "image_tool"
    return "pdf_tool"


def image_tool(state: AgentState) -> AgentState:
    """Validate image operation and enforce parameter schema."""
    valid = {"compress", "increase", "resize", "convert", "rotate"}
    if state["operation"] not in valid:
        return {**state, "error": f"Unsupported image operation: '{state['operation']}'"}

    validated_params, err = _validate_operation_params("image", state["operation"], state.get("params", {}))
    if err:
        return {**state, "params": validated_params, "error": err}
    return {**state, "params": validated_params, "error": None}


def pdf_tool(state: AgentState) -> AgentState:
    """Validate PDF operation and enforce parameter schema."""
    valid = {"compress", "increase", "merge", "split", "extract_pages", "images_to_pdf"}
    if state["operation"] not in valid:
        return {**state, "error": f"Unsupported PDF operation: '{state['operation']}'"}

    validated_params, err = _validate_operation_params("pdf", state["operation"], state.get("params", {}))
    if err:
        return {**state, "params": validated_params, "error": err}
    return {**state, "params": validated_params, "error": None}


def repair_node(state: AgentState) -> AgentState:
    """
    Self-repair cycle: heals invalid, negative, or out-of-bounds parameters
    before re-validating in the cyclic state machine.
    """
    op = state.get("operation", "")
    params = dict(state.get("params", {}))
    file_type = state.get("file_type", "image")
    retries = state.get("retry_count", 0) + 1

    if op in ("compress", "increase"):
        tkb = params.get("target_kb")
        if not isinstance(tkb, int) or tkb <= 0:
            params["target_kb"] = 100 if (op == "compress" and file_type == "image") else 200
        elif tkb > 50000:
            params["target_kb"] = 50000
    elif op == "rotate":
        deg = params.get("degrees", 90)
        if deg not in (90, 180, 270):
            params["degrees"] = 90
    elif op == "convert":
        fmt = str(params.get("format", "PNG")).upper().replace("JPG", "JPEG")
        params["format"] = fmt if fmt in ("JPEG", "PNG", "WEBP") else "PNG"
    elif op == "resize":
        if "scale_percent" in params and params["scale_percent"] is not None:
            params["scale_percent"] = max(1, min(500, int(params["scale_percent"])))
        else:
            if params.get("width"):
                params["width"] = max(1, min(10000, int(params["width"])))
            if params.get("height"):
                params["height"] = max(1, min(10000, int(params["height"])))
            if not params.get("width") and not params.get("height"):
                params["scale_percent"] = 50
    elif op == "split":
        params["start_page"] = max(1, int(params.get("start_page", 1)))
        if params.get("end_page") is not None:
            params["end_page"] = max(params["start_page"], int(params["end_page"]))
    elif op == "extract_pages":
        pages = params.get("pages", [1])
        if isinstance(pages, list) and pages:
            params["pages"] = [max(1, int(p)) for p in pages[:1000]]
        else:
            params["pages"] = [1]

    logger.info(f"Self-repair node normalized parameters (attempt {retries}): {params}")
    return {**state, "params": params, "error": None, "retry_count": retries}


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


# ── Build Cyclic Graph ────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("parse_instruction", parse_instruction)
    g.add_node("image_tool", image_tool)
    g.add_node("pdf_tool", pdf_tool)
    g.add_node("repair_node", repair_node)
    g.add_node("error_node", error_node)

    g.set_entry_point("parse_instruction")
    g.add_conditional_edges("parse_instruction", route, {
        "image_tool": "image_tool",
        "pdf_tool":   "pdf_tool",
        "error_node": "error_node",
    })

    def check_tool_result(state: AgentState) -> str:
        if state.get("error"):
            # Trigger cyclic repair loop if retry threshold has not been reached
            if state.get("retry_count", 0) < 2 and state.get("operation") not in ("unknown", ""):
                return "repair_node"
            return "error_node"
        return "end"

    g.add_conditional_edges("image_tool", check_tool_result, {
        "repair_node": "repair_node",
        "error_node": "error_node",
        "end": END,
    })
    g.add_conditional_edges("pdf_tool", check_tool_result, {
        "repair_node": "repair_node",
        "error_node": "error_node",
        "end": END,
    })
    # Repair loops back through tool validation
    g.add_conditional_edges("repair_node", route, {
        "image_tool": "image_tool",
        "pdf_tool":   "pdf_tool",
        "error_node": "error_node",
    })
    g.add_edge("error_node", END)

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
    Returns {"operation": str, "params": dict, "error": str | None, "fallback_used": bool}
    """
    graph = get_graph()
    result = graph.invoke({
        "instruction": instruction,
        "file_type": file_type,
        "file_name": file_name,
        "operation": "",
        "params": {},
        "error": None,
        "fallback_used": False,
        "retry_count": 0,
    })
    return {
        "operation":     result.get("operation", "unknown"),
        "params":        result.get("params", {}),
        "error":         result.get("error"),
        "fallback_used": result.get("fallback_used", False),
    }
