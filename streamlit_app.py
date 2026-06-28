import io
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import altair as alt

from salesvision_ai.services.csv_service import coerce_sales_column_values

BACKEND_URL = "http://localhost:8000/api/v1/data/upload"
DATA_DIR = Path("data")
CONFIG_PATH = DATA_DIR / "prediction_config.json"


def upload_csv_to_backend(file_bytes: bytes, filename: str, backend_url: str):
    files = {"file": (filename, file_bytes, "text/csv")}
    response = requests.post(backend_url, files=files)
    response.raise_for_status()
    return response.json()


def load_dataframe(file_bytes: bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


def save_prediction_config(config: dict) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
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

    st.markdown("---")
    st.subheader("Generar y visualizar pronóstico")
    if st.button("Generar pronóstico con motor predictivo"):
        # construir lista de history a partir del dataframe
        try:
            df = dataframe.copy()
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            if sales_column in df.columns:
                df[sales_column] = coerce_sales_column_values(df[sales_column])
            agg = df.groupby(date_column)[sales_column].sum().reset_index()
            agg = agg.sort_values(by=date_column)
            history_values = pd.to_numeric(agg[sales_column], errors="coerce").dropna().tolist()
            if not history_values:
                st.error("No se encontraron valores de ventas válidos para generar el historial.")
            else:
                # derivar la URL del endpoint predict a partir de backend_url
                if "/data/upload" in backend_url:
                    predict_url = backend_url.replace("/data/upload", "/predict/")
                elif backend_url.endswith("/upload"):
                    predict_url = backend_url.replace("/upload", "/predict/")
                else:
                    predict_url = backend_url.rstrip("/") + "/predict/"

                payload = {"history": history_values, "horizon_days": int(horizon_days)}
                try:
                    resp = requests.post(predict_url, json=payload)
                    resp.raise_for_status()
                    forecast = resp.json()
                    st.success("Pronóstico generado correctamente.")
                    # preparar dataframe para graficar
                    last_date = agg[date_column].max()
                    hist_df = agg[[date_column, sales_column]].rename(columns={date_column: "date", sales_column: "value"})
                    hist_df["type"] = "history"

                    fut_rows = []
                    for p in forecast.get("forecast", []):
                        idx = int(p.get("date_index", 0))
                        date = last_date + pd.Timedelta(days=idx)
                        fut_rows.append({"date": date, "value": p.get("estimated", 0.0), "ci_lower": p.get("ci_lower"), "ci_upper": p.get("ci_upper"), "type": "forecast"})
                    fut_df = pd.DataFrame(fut_rows)

                    history_chart = alt.Chart(hist_df).mark_line(strokeWidth=2, color="#2c3e50").encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("value:Q", title="Sales"),
                        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("value:Q", title="Histórico")],
                    )

                    trend_rows = []
                    for _, row in fut_df.iterrows():
                        trend_rows.append({"date": row["date"], "series": "estimated", "value": row["value"]})
                        if pd.notna(row.get("ci_lower")):
                            trend_rows.append({"date": row["date"], "series": "lower", "value": row["ci_lower"]})
                        if pd.notna(row.get("ci_upper")):
                            trend_rows.append({"date": row["date"], "series": "upper", "value": row["ci_upper"]})
                    trend_df = pd.DataFrame(trend_rows)

                    trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("value:Q", title="Sales"),
                        color=alt.Color("series:N", title="Serie", scale=alt.Scale(domain=["estimated", "lower", "upper"], range=["#1f77b4", "#d62728", "#2ca02c"])),
                        strokeDash=alt.StrokeDash("series:N", title="Serie", scale=alt.Scale(domain=["estimated", "lower", "upper"], range=[[1, 0], [4, 2], [2, 2]])),
                        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("value:Q", title="Valor"), alt.Tooltip("series:N", title="Serie")],
                    )

                    if not fut_df.empty and fut_df["ci_lower"].notnull().all() and fut_df["ci_upper"].notnull().all():
                        ci_df = fut_df[["date", "ci_lower", "ci_upper"]].copy().rename(columns={"ci_lower": "lower", "ci_upper": "upper"})
                        area = alt.Chart(ci_df).mark_area(opacity=0.2, color="#5ab4ac").encode(
                            x="date:T",
                            y="lower:Q",
                            y2="upper:Q",
                        )
                        chart = (area + trend_chart + history_chart).properties(width=800, height=300)
                    else:
                        chart = (trend_chart + history_chart).properties(width=800, height=300)

                    st.altair_chart(chart, use_container_width=True)

                    st.subheader("Tabla de resultados")
                    if not fut_df.empty:
                        display_df = fut_df.copy()
                        display_df = display_df[["date", "value", "ci_lower", "ci_upper"]]
                        display_df = display_df.rename(columns={"value": "estimated", "ci_lower": "ci_lower", "ci_upper": "ci_upper"})
                        st.dataframe(display_df)
                except requests.exceptions.RequestException as exc:
                    st.error(f"Error al solicitar pronóstico al backend: {exc}")
        except Exception as exc:
            st.error(f"Error preparando datos para pronóstico: {exc}")


if __name__ == "__main__":
    main()
