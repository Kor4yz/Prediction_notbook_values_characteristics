PY=python

setup:
\t$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pre-commit install

train:
\t$(PY) -m src.train --input data/raw/laptop_price.csv --out models/best_model.joblib --report reports/metrics.json

evaluate:
\t$(PY) -m src.evaluate --model models/best_model.joblib --input data/raw/laptop_price.csv --outdir reports/figures
