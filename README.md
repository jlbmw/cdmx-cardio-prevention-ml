# Prevención Cardiovascular (EDA & ML)

EDA reproducible, procesamiento y baseline de ML para factores asociados a enfermedades cardiovasculares.

## 🎯 Objetivos
- Preparar y transformar datos (scripts en `src/`)
- Ejecutar un EDA reproducible (notebook en `notebooks/eda_heart_disease.ipynb`)
- Generar reporte automatizado en Word con hallazgos principales
- Establecer baseline predictivo y base para monitoreo

## 🗂️ Estructura
```
.
├─ src/
│  ├─ processing_heart_disease_v1.py
│  └─ processing_heart_disease_v2.py   # autodetecta CSV en la carpeta data/
├─ notebooks/
│  └─ eda_heart_disease.ipynb
├─ data/
│  ├─ raw/        # datos crudos (no versionados)
│  └─ processed/  # salidas procesadas
├─ reports/
│  ├─ report_eda_heart_disease_v1.docx
│  └─ figs/
├─ models/
├─ outputs/
├─ docs/
└─ requirements.txt
```

## ⚙️ Requisitos
- Python 3.10+ (Codespaces ya trae Python)
- (Opcional) entorno virtual

Instalar dependencias:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Cómo ejecutar

### 1) Procesamiento de datos
- **v1 (ruta fija):**
```bash
python src/processing_heart_disease_v1.py
```
Salida:
- `processing_heart_disease v1.csv` en la raíz
- `processing_heart_disease_mappings.json` (mapeos binarios)

- **v2 (autodetección en `data/`):** coloca tu CSV (p. ej., `data/heart_2020_cleaned.csv`) y:
```bash
python src/processing_heart_disease_v2.py
```

### 2) EDA
Abre y ejecuta el notebook:
```
notebooks/eda_heart_disease.ipynb
```
- Ajusta `CSV_PATH` en la primera celda.
- El notebook guarda figuras en `outputs/` y genera:
  - `reports/report_eda_heart_disease_v1.docx`

## 🔐 Datos y privacidad
- No subas datos sensibles a repos públicos.
- `data/raw/` está ignorado por defecto.

## 👥 Colaboración y permisos
- Settings → Collaborators para invitar usuarios (o Teams si es una organización).
- Recomendado usar Pull Requests hacia `main` con protección de rama.

## 🧭 Roadmap (resumen)
- Fase 1: Validación de datos
- Fase 2: EDA + reporte
- Fase 3: Baseline ML + métricas
- Fase 4: Integración con comunicación y monitoreo

## 📄 Licencia
MIT (ajústala según tu organización).
