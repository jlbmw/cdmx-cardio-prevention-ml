"""
processing_heart_disease v2.py
Autor: JLBM
Fecha: 17/09/2025
Descripción:
  - Detecta automáticamente el archivo CSV dentro de la carpeta 'data/'.
  - Aplica StandardScaler a variables numéricas.
  - Codifica variables categóricas:
      * Binarias (2 categorías): 0/1
      * 3+ categorías: one-hot-encoding
  - Exporta el resultado a 'processing_heart_disease v2.csv'
  - Exporta diccionario de mapeos binarios a 'processing_heart_disease_mappings.json'
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import sys

# === 1. Buscar automáticamente el CSV en la carpeta 'data/' ===
BASE_DIR = Path(__file__).parent  # carpeta donde está el script
DATA_DIR = BASE_DIR / "data"

if not DATA_DIR.exists():
    sys.exit(f"❌ No existe la carpeta {DATA_DIR}. Crea 'data/' y coloca allí tu archivo CSV.")

csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    sys.exit(f"❌ No se encontró ningún archivo CSV en {DATA_DIR}.")
elif len(csv_files) > 1:
    print(f"⚠️ Se encontraron varios CSV, se usará el primero: {csv_files[0].name}")

INPUT_CSV = csv_files[0]
OUTPUT_CSV = BASE_DIR / "processing_heart_disease v2.csv"
OUTPUT_JSON = BASE_DIR / "processing_heart_disease_mappings.json"

print(f"✔ Archivo detectado: {INPUT_CSV.name}")

# === 2. Función de procesamiento ===
def process(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    scaled_df = df.copy()
    if num_cols:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(scaled_df[num_cols])
        scaled_df[num_cols] = scaled_values

    out_parts = []
    if num_cols:
        out_parts.append(scaled_df[num_cols])
    others = [c for c in scaled_df.columns if c not in num_cols and c not in cat_cols]
    if others:
        out_parts.append(scaled_df[others])

    binary_maps = {}
    for c in cat_cols:
        col = df[c]
        if col.dtype == bool:
            col = col.astype(str)
        uniq = sorted(list(col.dropna().astype(str).unique()))
        k = len(uniq)
        if k == 2:
            mapping = {uniq[0]: 0, uniq[1]: 1}
            binary_maps[c] = mapping
            encoded = col.astype(str).map(mapping)
            encoded = encoded.where(~col.isna(), np.nan)
            out_parts.append(pd.Series(encoded, name=c))
        elif k >= 3:
            dummies = pd.get_dummies(col, prefix=c, prefix_sep="__", dummy_na=True)
            out_parts.append(dummies)

    processed = pd.concat(out_parts, axis=1)

    # Reordenar columnas
    cols_num = [c for c in processed.columns if c in num_cols]
    cols_bin = [c for c in processed.columns if c in cat_cols and processed[c].dtype != 'O']
    cols_ohe = [c for c in processed.columns if "__" in c]
    cols_other = [c for c in processed.columns if c not in cols_num + cols_bin + cols_ohe]
    final_cols = cols_num + cols_bin + cols_ohe + cols_other
    processed = processed[final_cols]

    return processed, binary_maps

# === 3. Ejecutar procesamiento ===
def main():
    df = pd.read_csv(INPUT_CSV)
    processed, binary_maps = process(df)

    processed.to_csv(OUTPUT_CSV, index=False)
    print(f"✔ Datos procesados guardados en: {OUTPUT_CSV}")

    if binary_maps:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(binary_maps, f, indent=2, ensure_ascii=False)
        print(f"✔ Mapeos binarios guardados en: {OUTPUT_JSON}")

    print("Dimensiones originales :", df.shape)
    print("Dimensiones procesadas :", processed.shape)

if __name__ == "__main__":
    main()
