from fastapi.testclient import TestClient
import pandas as pd

from salesvision_ai.main import app
from salesvision_ai.services.csv_service import coerce_sales_column_values, parse_csv_bytes

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


def test_parse_csv_bytes_converts_sales_column_to_float():
    csv = "Monto_Venta,producto\n847.68,alpha\n$491.70,beta\n$1293.27,gamma\n"

    cols, preview = parse_csv_bytes(csv.encode("utf-8"))

    assert cols == ["Monto_Venta", "producto"]
    assert preview[0]["Monto_Venta"] == 847.68
    assert preview[1]["Monto_Venta"] == 491.70
    assert preview[2]["Monto_Venta"] == 1293.27


def test_coerce_sales_column_values_handles_currency_strings():
    series = pd.Series(["847.68", "$491.70", "bad-value", "1293.27"])

    result = coerce_sales_column_values(series)

    assert result.tolist() == [847.68, 491.70, pd.NA, 1293.27]
