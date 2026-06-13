from fastapi.testclient import TestClient
from salesvision_ai.main import app

client = TestClient(app)


def test_upload_csv_success():
    csv = "date,sales\n2023-01-01,100\n2023-01-02,150\n"
    files = {"file": ("test.csv", csv, "text/csv")}
    resp = client.post("/api/v1/data/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "test.csv"
    assert "columns" in data
    assert "preview" in data
    assert data["columns"] == ["date", "sales"]
    assert len(data["preview"]) <= 5
