import io
import pandas as pd


def parse_csv_bytes(content: bytes, n_preview: int = 5):
    bio = io.BytesIO(content)
    df = pd.read_csv(bio)
    cols = df.columns.tolist()
    preview = df.head(n_preview).to_dict(orient="records")
    return cols, preview
