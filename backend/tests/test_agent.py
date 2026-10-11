"""Tests for LangGraph agent router and state machine — NexaForge"""
import pytest
from agent.graph import route, image_tool, pdf_tool, error_node, AgentState


def test_agent_route_image_success():
    state: AgentState = {
        "instruction": "compress image under 100kb",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "compress",
        "params": {"target_kb": 100},
        "error": None,
    }
    assert route(state) == "image_tool"


def test_agent_route_image_increase():
    state: AgentState = {
        "instruction": "increase image to 200kb",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "increase",
        "params": {"target_kb": 200},
        "error": None,
    }
    assert route(state) == "image_tool"
    assert image_tool(state).get("error") is None


def test_agent_route_pdf_success():
    state: AgentState = {
        "instruction": "compress pdf under 300kb",
        "file_type": "pdf",
        "file_name": "doc.pdf",
        "operation": "compress",
        "params": {"target_kb": 300},
        "error": None,
    }
    assert route(state) == "pdf_tool"


def test_agent_route_pdf_increase():
    state: AgentState = {
        "instruction": "increase pdf to 300kb",
        "file_type": "pdf",
        "file_name": "doc.pdf",
        "operation": "increase",
        "params": {"target_kb": 300},
        "error": None,
    }
    assert route(state) == "pdf_tool"
    assert pdf_tool(state).get("error") is None


def test_agent_route_error_on_unknown_operation():
    state: AgentState = {
        "instruction": "do something unknown",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "unknown",
        "params": {},
        "error": None,
    }
    assert route(state) == "error_node"


def test_agent_route_error_on_existing_error():
    state: AgentState = {
        "instruction": "fail please",
        "file_type": "pdf",
        "file_name": "doc.pdf",
        "operation": "compress",
        "params": {},
        "error": "Failed to parse",
    }
    assert route(state) == "error_node"


def test_image_tool_node_valid():
    state: AgentState = {
        "instruction": "resize to 800x600",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "resize",
        "params": {"width": 800, "height": 600},
        "error": None,
    }
    res = image_tool(state)
    assert res.get("error") is None
    assert res["operation"] == "resize"


def test_image_tool_node_invalid_op():
    state: AgentState = {
        "instruction": "merge images",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "merge",
        "params": {},
        "error": None,
    }
    res = image_tool(state)
    assert res.get("error") is not None
    assert "Unsupported image operation" in res["error"]


def test_pdf_tool_node_valid():
    state: AgentState = {
        "instruction": "split pages 1 to 3",
        "file_type": "pdf",
        "file_name": "doc.pdf",
        "operation": "split",
        "params": {"start_page": 1, "end_page": 3},
        "error": None,
    }
    res = pdf_tool(state)
    assert res.get("error") is None


def test_pdf_tool_node_invalid_op():
    state: AgentState = {
        "instruction": "sharpen pdf",
        "file_type": "pdf",
        "file_name": "doc.pdf",
        "operation": "sharpen",
        "params": {},
        "error": None,
    }
    res = pdf_tool(state)
    assert res.get("error") is not None
    assert "Unsupported PDF operation" in res["error"]


def test_error_node_message_generation():
    state: AgentState = {
        "instruction": "gibberish",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "unknown",
        "params": {},
        "error": None,
    }
    res = error_node(state)
    assert res.get("error") is not None
    assert "Unable to determine the intended operation" in res["error"]


def test_fallback_mb_and_decimal_parsing():
    from agent.graph import _heuristic_fallback_parse
    # 1MB -> 1000 KB
    res1 = _heuristic_fallback_parse("compress image under 1MB", "image")
    assert res1 is not None
    assert res1["operation"] == "compress"
    assert res1["params"]["target_kb"] == 1000

    # 0.5 MB -> 500 KB
    res2 = _heuristic_fallback_parse("reduce photo under 0.5 MB", "image")
    assert res2 is not None
    assert res2["params"]["target_kb"] == 500


def test_fallback_extract_page_range_expansion():
    from agent.graph import _heuristic_fallback_parse
    res = _heuristic_fallback_parse("extract pages 2-4", "pdf")
    assert res is not None
    assert res["operation"] == "extract_pages"
    assert res["params"]["pages"] == [2, 3, 4]


def test_fallback_percentage_scale_parsing():
    from agent.graph import _heuristic_fallback_parse
    res = _heuristic_fallback_parse("scale by 50%", "image")
    assert res is not None
    assert res["operation"] == "resize"
    assert res["params"]["scale_percent"] == 50


def test_repair_node_normalizes_defective_params():
    from agent.graph import repair_node
    state: AgentState = {
        "instruction": "compress image",
        "file_type": "image",
        "file_name": "photo.jpg",
        "operation": "compress",
        "params": {"target_kb": -10},
        "error": "Target must be positive",
        "fallback_used": False,
        "retry_count": 0,
    }
    repaired = repair_node(state)
    assert repaired["params"]["target_kb"] == 100
    assert repaired["error"] is None
    assert repaired["retry_count"] == 1

