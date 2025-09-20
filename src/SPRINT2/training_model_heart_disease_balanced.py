
"""
training_model_heart_disease_balanced.py
- Carga heart_2020_cleaned.csv
- Split 80/20 (estratificado)
- Preprocesa: StandardScaler (num) + OneHot (cat)
- Entrena Regresión Logística con class_weight='balanced'
- Reporta precision, recall, accuracy, f1-score (train y test)
- Guarda resultados en results_training_model_heart_disease_balanced.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

DATA_PATH = Path("/mnt/data/heart_2020_cleaned.csv")
OUT_TXT = Path("/mnt/data/results_training_model_heart_disease_balanced.txt")

def prepare_target(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        y_str = y.astype(str).str.strip().str.lower()
        mapping_yes = {"yes","1","true","si","sí"}
        mapping_no  = {"no","0","false"}
        y_bin = y_str.map(lambda v: 1 if v in mapping_yes else (0 if v in mapping_no else np.nan))
        if y_bin.isna().any():
            y_num = pd.to_numeric(y, errors="coerce")
            y_bin = y_bin.fillna(y_num)
        return y_bin.astype(int)
    return y.astype(int)

def main():
    df = pd.read_csv(DATA_PATH)
    if "HeartDisease" not in df.columns:
        raise ValueError("No se encontró la columna objetivo 'HeartDisease'.")

    y = prepare_target(df["HeartDisease"])
    X = df.drop(columns=["HeartDisease"])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category","bool"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    y_tr_pred = pipe.predict(X_train)
    y_te_pred = pipe.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1_score":  f1_score(y_true, y_pred, zero_division=0),
        }

    m_tr = metrics(y_train, y_tr_pred)
    m_te = metrics(y_test, y_te_pred)

    lines = []
    lines.append("=== training_model_heart_disease_balanced.py ===")
    lines.append(f"Archivo de datos: {DATA_PATH}")
    lines.append(f"Filas totales: {df.shape[0]}  |  Variables (incl. target): {df.shape[1]}")
    lines.append("División: 80% entrenamiento / 20% prueba (estratificado)")
    lines.append("Modelo: Regresión Logística (class_weight='balanced')")
    lines.append("Objetivo: HeartDisease")
    lines.append("")
    lines.append("— Métricas (Entrenamiento) —")
    for k,v in m_tr.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")
    lines.append("— Métricas (Prueba) —")
    for k,v in m_te.items():
        lines.append(f"{k}: {v:.4f}")
    lines.append("")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
