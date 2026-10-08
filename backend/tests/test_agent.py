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

