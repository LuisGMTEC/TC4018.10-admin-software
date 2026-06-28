import io
import pandas as pd


def _looks_like_sales_column(column_name: str) -> bool:
    normalized = column_name.lower().replace(" ", "_")
    sales_tokens = ("venta", "sales", "sale", "revenue", "monto", "amount", "ingreso")
    return any(token in normalized for token in sales_tokens)


def coerce_sales_column_values(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"[$,]", "", regex=True)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_sales_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if not _looks_like_sales_column(column):
            continue
        df[column] = coerce_sales_column_values(df[column])

    return df


def parse_csv_bytes(content: bytes, n_preview: int = 5):
    bio = io.BytesIO(content)
    df = pd.read_csv(bio)
    df = _parse_sales_columns(df)
    cols = df.columns.tolist()
    preview = df.head(n_preview).to_dict(orient="records")
    return cols, preview
