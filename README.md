# SCSE3040 MLOps Mini Project — MP089 Bakery Bread Sales

Predict daily `loaves_sold` for Nagpur bakery outlets. This is a regression task evaluated using MAE.

> Read `STUDENT_HANDBOOK.md` (not yet supplied) before finalizing layout, seven data-problem checks, and submission details. This scaffold is based on the supplied problem statement only. Do not fabricate findings or scores.

## Data
Place the supplied files in `data/`:
- `train.csv`
- `later.csv`
- `holdout_features.csv`

Keep originals unchanged. Use Python for inspection; do not open/save the original CSVs in Excel.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Current scaffold
- `src/audit_data.py`: non-destructive data audit (run after data is placed)
- `src/train.py`: training/MLflow scaffold
- `src/monitor.py`: PSI and later-period evaluation scaffold
- `src/predict.py`: holdout prediction scaffold
- `src/api.py`: FastAPI API scaffold
- `tests/`: tests
- `k8s/`: Kubernetes deployment and service templates

## Important
The API accepts exactly eight features: `price_rs`, `temp_c`, `footfall_index`, `competitor_shops`, `shelf_hours`, `orders_previous_day`, `day_type`, `outlet`. The schema rejects `orders_previous_day > 850` and `price_rs < 10`.
