#!/usr/bin/env python3
"""
Genera imágenes PNG en `data/screenshots/` que sirven como evidencia para
las Historias de Usuario de la Etapa 3. Usa el CSV de prueba y el archivo
`data/prediction_output.json` producido por el script de integración.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCREEN_DIR = DATA_DIR / "screenshots"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "testing_data" / "ventas_retail_6_meses.csv"
PRED_PATH = DATA_DIR / "prediction_output.json"


def make_table_image(df: pd.DataFrame, out_path: Path, title: str = ""):
    # Simple table renderer using PIL
    font = ImageFont.load_default()
    padding = 10
    measure_img = Image.new('RGB', (10, 10))
    measure_draw = ImageDraw.Draw(measure_img)
    col_widths = []
    cols = list(df.columns)
    # estimate column widths
    for c in cols:
        bbox = measure_draw.textbbox((0, 0), str(c), font=font)
        maxw = bbox[2] - bbox[0]
        for v in df[c].astype(str).values:
            bbox_v = measure_draw.textbbox((0, 0), str(v), font=font)
            w = bbox_v[2] - bbox_v[0]
            if w > maxw:
                maxw = w
        col_widths.append(maxw + 20)

    bbox_h = measure_draw.textbbox((0, 0), 'Hg', font=font)
    row_height = (bbox_h[3] - bbox_h[1]) + 8
    header_height = row_height + 4
    img_w = sum(col_widths) + padding * 2
    img_h = header_height + row_height * len(df) + padding * 2 + 30
    img = Image.new('RGB', (img_w, img_h), color='white')
    draw = ImageDraw.Draw(img)
    # title
    draw.text((padding, 4), title, fill='black', font=font)
    y = padding + 20
    x = padding
    # header
    for i, c in enumerate(cols):
        draw.text((x + 4, y), str(c), fill='black', font=font)
        x += col_widths[i]
    y += header_height
    # rows
    for _, row in df.iterrows():
        x = padding
        for i, c in enumerate(cols):
            draw.text((x + 4, y), str(row[c]), fill='black', font=font)
            x += col_widths[i]
        y += row_height
    img.save(out_path)


def make_line_chart(history, forecast_rows, out_path: Path, title: str = "Forecast"):
    # Simple line chart renderer using PIL
    font = ImageFont.load_default()
    width = 1000
    height = 400
    margin = 60
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Prepare data
    hist_dates = pd.to_datetime([r[0] for r in history])
    hist_vals = [float(r[1]) for r in history]
    fut_dates = [pd.to_datetime(history[-1][0]) + pd.Timedelta(days=int(r['date_index'])) for r in forecast_rows]
    fut_vals = [float(r['estimated']) for r in forecast_rows]
    all_vals = hist_vals + fut_vals
    vmin = min(all_vals)
    vmax = max(all_vals)
    if vmax == vmin:
        vmax = vmin + 1

    def x_pos(idx, total):
        return margin + int((width - 2 * margin) * idx / max(1, total - 1))

    def y_pos(val):
        return margin + int((height - 2 * margin) * (1 - (val - vmin) / (vmax - vmin)))

    # draw axes
    draw.line((margin, margin, margin, height - margin), fill='black')
    draw.line((margin, height - margin, width - margin, height - margin), fill='black')
    draw.text((margin, 8), title, fill='black', font=font)

    # draw history
    total = len(hist_vals) + len(fut_vals)
    points_hist = []
    for i, v in enumerate(hist_vals):
        x = x_pos(i, total)
        y = y_pos(v)
        points_hist.append((x, y))
    if len(points_hist) > 1:
        draw.line(points_hist, fill='blue', width=2)

    # draw forecast
    points_fut = []
    for i, v in enumerate(fut_vals, start=len(hist_vals)):
        x = x_pos(i, total)
        y = y_pos(v)
        points_fut.append((x, y))
    if points_fut:
        draw.line([points_hist[-1]] + points_fut, fill='red', width=2)

    # draw CI as shaded area if present
    ci_lower = [r.get('ci_lower') for r in forecast_rows]
    ci_upper = [r.get('ci_upper') for r in forecast_rows]
    if ci_lower and all(v is not None for v in ci_lower):
        poly = []
        for i, v in enumerate(ci_lower, start=len(hist_vals)):
            poly.append((x_pos(i, total), y_pos(v)))
        for i, v in reversed(list(enumerate(ci_upper, start=len(hist_vals)))):
            poly.append((x_pos(i, total), y_pos(v)))
        draw.polygon(poly, fill=(200, 200, 200, 100))

    img.save(out_path)


def main():
    # HU-01: tabla preview (first 6 rows)
    df = pd.read_csv(CSV_PATH)
    preview = df.head(6).copy()
    make_table_image(preview, SCREEN_DIR / 'hu01_carga_csv.png', title='HU-01: Preview CSV (first 6 rows)')

    # HU-02: selection of columns (show column names)
    cols = pd.DataFrame({'columns': list(df.columns)})
    make_table_image(cols, SCREEN_DIR / 'hu02_seleccion_columnas.png', title='HU-02: Columnas detectadas')

    # HU-03: horizon configuration (show example)
    horizon_df = pd.DataFrame({'parameter': ['horizon_days', 'date_column', 'sales_column'], 'value': [7, 'Fecha', 'Monto_Venta']})
    make_table_image(horizon_df, SCREEN_DIR / 'hu03_horizonte.png', title='HU-03: Resumen de configuración')

    # HU-04: backend validation - show sample upload JSON (from integrate script output if available)
    if PRED_PATH.exists():
        with PRED_PATH.open('r', encoding='utf-8') as fh:
            pred = json.load(fh)
    else:
        pred = {'message': 'prediction_output.json not found'}
    # make a small dataframe with model name and counts
    model = pred.get('model', {})
    upload_df = pd.DataFrame([{'filename': CSV_PATH.name, 'model': model.get('name', ''), 'forecast_len': len(pred.get('forecast', []))}])
    make_table_image(upload_df, SCREEN_DIR / 'hu04_validacion_backend.png', title='HU-04: Resultado validación backend')

    # HU-05/06/07/08: charts and table from prediction
    if pred and 'forecast' in pred and len(pred['forecast']) > 0:
        # create history as list of (date, value) using CSV aggregation
        df['Monto_Clean'] = df['Monto_Venta'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        agg = df.groupby('Fecha')['Monto_Clean'].sum().reset_index()
        history = list(zip(agg['Fecha'].dt.strftime('%Y-%m-%d').tolist(), agg['Monto_Clean'].tolist()))
        make_line_chart(history, pred['forecast'], SCREEN_DIR / 'hu05_06_pronostico_escenarios.png', title='HU-05/06: Pronóstico y escenarios')

        # HUD-08: detailed table of forecast
        table_rows = []
        for f in pred['forecast']:
            table_rows.append({'date_index': f.get('date_index'), 'estimated': f.get('estimated'), 'ci_lower': f.get('ci_lower'), 'ci_upper': f.get('ci_upper')})
        table_df = pd.DataFrame(table_rows)
        make_table_image(table_df, SCREEN_DIR / 'hu08_tabla_resultados.png', title='HU-08: Tabla detallada de resultados')

    else:
        # create placeholders
        pd.DataFrame([{'note': 'No forecast available'}]).to_csv(SCREEN_DIR / 'hu05_06_pronostico_escenarios.png')

    # HU-09: error handling screenshot (simulated message)
    err_df = pd.DataFrame([{'error': 'Simulated: missing date column'}])
    make_table_image(err_df, SCREEN_DIR / 'hu09_manejo_errores.png', title='HU-09: Manejo de errores (simulado)')

    print('Screenshots generated in', SCREEN_DIR)


if __name__ == '__main__':
    main()
