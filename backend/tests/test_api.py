"""Integration tests for FastAPI endpoints — NexaForge"""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import fitz

from main import app

client = TestClient(app)


def _make_dummy_image(fmt="JPEG", size=(400, 300)):
    img = Image.new("RGB", size, color=(120, 180, 240))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_dummy_pdf(pages=2):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Test page {i + 1}")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "NexaForge"


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    assert "NexaForge API" in res.json()["message"]


# ── Image Endpoints ───────────────────────────────────────────────────────────

def test_api_image_compress():
    data = _make_dummy_image()
    files = {"file": ("test.jpg", data, "image/jpeg")}
    res = client.post("/api/image/compress", files=files, data={"target_kb": 100})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 0


def test_api_image_resize():
    data = _make_dummy_image(size=(800, 600))
    files = {"file": ("test.jpg", data, "image/jpeg")}
    res = client.post("/api/image/resize", files=files, data={"width": 400, "height": 300})
    assert res.status_code == 200
    img = Image.open(io.BytesIO(res.content))
    assert img.size == (400, 300)


def test_api_image_convert():
    data = _make_dummy_image(fmt="JPEG")
    files = {"file": ("test.jpg", data, "image/jpeg")}
    res = client.post("/api/image/convert", files=files, data={"target_format": "PNG"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"


def test_api_image_rotate():
    data = _make_dummy_image(size=(400, 300))
    files = {"file": ("test.jpg", data, "image/jpeg")}
    res = client.post("/api/image/rotate", files=files, data={"degrees": 90})
    assert res.status_code == 200
    img = Image.open(io.BytesIO(res.content))
    assert img.size == (300, 400)


# ── PDF Endpoints ─────────────────────────────────────────────────────────────

def test_api_pdf_compress():
    data = _make_dummy_pdf()
    files = {"file": ("doc.pdf", data, "application/pdf")}
    res = client.post("/api/pdf/compress", files=files, data={"target_kb": 200})
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"


def test_api_pdf_merge():
    pdf1 = _make_dummy_pdf(2)
    pdf2 = _make_dummy_pdf(3)
    files = [
        ("files", ("doc1.pdf", pdf1, "application/pdf")),
        ("files", ("doc2.pdf", pdf2, "application/pdf")),
    ]
    res = client.post("/api/pdf/merge", files=files)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    doc = fitz.open(stream=res.content, filetype="pdf")
    assert len(doc) == 5
    doc.close()


def test_api_pdf_split():
    data = _make_dummy_pdf(4)
    files = {"file": ("doc.pdf", data, "application/pdf")}
    res = client.post("/api/pdf/split", files=files, data={"start_page": 2, "end_page": 3})
    assert res.status_code == 200
    doc = fitz.open(stream=res.content, filetype="pdf")
    assert len(doc) == 2
    doc.close()


def test_api_pdf_extract_pages():
    data = _make_dummy_pdf(5)
    files = {"file": ("doc.pdf", data, "application/pdf")}
    res = client.post("/api/pdf/extract-pages", files=files, data={"pages": "1,3,4"})
    assert res.status_code == 200
    doc = fitz.open(stream=res.content, filetype="pdf")
    assert len(doc) == 3
    doc.close()


def test_api_image_increase():
    data = _make_dummy_image(size=(100, 100))
    files = {"file": ("test.jpg", data, "image/jpeg")}
    res = client.post("/api/image/increase", files=files, data={"target_kb": 50})
    assert res.status_code == 200
    assert len(res.content) >= 50 * 1024


def test_api_pdf_images_to_pdf():
    img1 = _make_dummy_image()
    img2 = _make_dummy_image()
    files = [
        ("files", ("img1.jpg", img1, "image/jpeg")),
        ("files", ("img2.jpg", img2, "image/jpeg")),
    ]
    res = client.post("/api/pdf/images-to-pdf", files=files)
    assert res.status_code == 200
    doc = fitz.open(stream=res.content, filetype="pdf")
    assert len(doc) == 2
    doc.close()


def test_api_pdf_increase():
    data = _make_dummy_pdf(1)
    files = {"file": ("test.pdf", data, "application/pdf")}
    res = client.post("/api/pdf/increase", files=files, data={"target_kb": 60})
    assert res.status_code == 200
    assert len(res.content) >= 60 * 1024


def test_api_process_and_download():
    data = _make_dummy_pdf(2)
    files = {"file": ("document.pdf", data, "application/pdf")}
    res = client.post(
        "/api/process",
        files=files,
        data={"instruction": "increase this pdf to 100 kb"},
    )
    assert res.status_code == 200
    res_json = res.json()
    assert res_json["status"] == "success"
    assert "reply" in res_json
    assert "verification" in res_json
    assert res_json["verification"]["status"] == "PASSED"
    assert "download_url" in res_json

    # Test downloading the result
    download_res = client.get(res_json["download_url"])
    assert download_res.status_code == 200
    assert len(download_res.content) >= 100 * 1024


def test_system_status_endpoint():
    res = client.get("/api/system/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "components" in data
    assert "api_server" in data["components"]
    assert "llm_engine" in data["components"]
    assert "vector_rag" in data["components"]
    assert "observability" in data["components"]
    assert data["components"]["api_server"]["status"] == "online"


def test_chained_session_pipeline():
    # Step 1: Compress image with natural instruction
    img_data = _make_dummy_image(size=(300, 300))
    files = {"file": ("banner.jpg", img_data, "image/jpeg")}
    res1 = client.post(
        "/api/process",
        files=files,
        data={"instruction": "compress under 50 kb"},
    )
    assert res1.status_code == 200
    json1 = res1.json()
    assert json1["status"] == "success"
    assert "Here is your image reduced" in json1["reply"]
    assert "session_file_id" in json1
    session_id = json1["session_file_id"]

    # Step 2: Chain follow-up edit using file_id without uploading any file!
    res2 = client.post(
        "/api/process",
        data={
            "file_id": session_id,
            "instruction": "rotate 90 degrees",
        },
    )
    assert res2.status_code == 200
    json2 = res2.json()
    assert json2["status"] == "success"
    assert "Here is your image rotated 90 degrees" in json2["reply"]
    assert json2["verification"]["status"] == "PASSED"
    assert "session_file_id" in json2


