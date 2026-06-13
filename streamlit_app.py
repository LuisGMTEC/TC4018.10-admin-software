import io
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/api/v1/data/upload"
CONFIG_PATH = Path("prediction_config.json")


def upload_csv_to_backend(file_bytes: bytes, filename: str, backend_url: str):
    files = {"file": (filename, file_bytes, "text/csv")}
    response = requests.post(backend_url, files=files)
    response.raise_for_status()
    return response.json()


def load_dataframe(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def save_prediction_config(config: dict) -> Path:
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return CONFIG_PATH


def main():
    st.set_page_config(page_title="SalesVision AI", layout="wide")
    st.title("SalesVision AI")
    st.markdown(
        "SalesVision AI permite cargar un CSV de ventas, configurar columnas clave, seleccionar categorías y definir el horizonte de pronóstico."
    )

    st.sidebar.header("Configuración del flujo")
    backend_url = st.sidebar.text_input("Backend URL", BACKEND_URL)

    uploaded_file = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    if uploaded_file is None:
        st.warning("Carga un archivo CSV para comenzar.")
        return

    file_bytes = uploaded_file.read()
    if len(file_bytes) == 0:
        st.error("El archivo está vacío.")
        return

    try:
        dataframe = load_dataframe(file_bytes)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        return

    all_columns = dataframe.columns.tolist()
    if len(all_columns) < 2:
        st.error("Se requieren al menos dos columnas en el CSV: una de fecha y otra de ventas.")
        return

    st.success("CSV cargado correctamente.")
    st.subheader("Vista previa de datos")
    st.dataframe(dataframe.head(5))

    all_columns = dataframe.columns.tolist()
    if not all_columns:
        st.error("El archivo CSV no contiene columnas detectables.")
        return

    st.subheader("Columnas detectadas")
    st.write(all_columns)

    st.subheader("Mapeo de columnas críticas")
    date_column = st.selectbox("Selecciona la columna de fecha", options=all_columns, index=0)
    sales_column = st.selectbox(
        "Selecciona la columna de ventas/cantidad",
        options=[c for c in all_columns if c != date_column],
        index=0,
    )

    possible_category_columns = [c for c in all_columns if c not in {date_column, sales_column}]
    category_column = None
    selected_category_value = None
    analysis_global = True

    if possible_category_columns:
        analysis_global = st.checkbox("Análisis global", value=True)
        if not analysis_global:
            category_column = st.selectbox("Selecciona la columna de categoría", options=possible_category_columns)
            if category_column:
                category_values = dataframe[category_column].astype(str).dropna().unique().tolist()
                selected_category_value = st.selectbox("Selecciona el valor de categoría", options=category_values)

    st.subheader("Horizonte de pronóstico")
    horizon_days = st.number_input(
        "Número de días a proyectar",
        min_value=1,
        max_value=364,
        value=30,
        step=1,
    )

    st.markdown("---")
    st.subheader("Resumen de configuración")
    st.write("**Archivo:**", uploaded_file.name)
    st.write("**Columna de fecha:**", date_column)
    st.write("**Columna de ventas:**", sales_column)
    st.write("**Horizonte (días):**", int(horizon_days))
    st.write("**Análisis global:**", analysis_global)
    if not analysis_global:
        st.write("**Columna de categoría:**", category_column)
        st.write("**Valor de categoría:**", selected_category_value)

    st.subheader("Validaciones de datos")
    if not pd.api.types.is_datetime64_any_dtype(dataframe[date_column]):
        st.info("La columna de fecha no está en formato datetime. Se intentará convertirla automáticamente.")
        try:
            dataframe[date_column] = pd.to_datetime(dataframe[date_column], errors="coerce")
        except Exception:
            pass

    if st.button("Validar y analizar CSV en el backend"):
        try:
            result = upload_csv_to_backend(file_bytes, uploaded_file.name, backend_url)
            st.success("El backend procesó el archivo correctamente.")
            st.json(result)
        except requests.exceptions.RequestException as exc:
            st.error(f"Error al enviar el archivo al backend: {exc}")
            st.write("Asegúrate de que el backend de FastAPI esté activo en", backend_url)

    st.markdown("---")
    st.subheader("Guardar configuración para predicción")
    st.write("Genera y guarda un archivo JSON con las selecciones actuales para su futura integración con el motor predictivo.")

    config = {
        "filename": uploaded_file.name,
        "backend_url": backend_url,
        "date_column": date_column,
        "sales_column": sales_column,
        "horizon_days": int(horizon_days),
        "analysis_global": analysis_global,
        "category_column": category_column,
        "category_value": selected_category_value,
        "available_columns": all_columns,
    }

    if st.button("Guardar configuración de predicción"):
        config_path = save_prediction_config(config)
        st.success(f"Configuración guardada en {config_path}")
        st.json(config)


if __name__ == "__main__":
    main()
