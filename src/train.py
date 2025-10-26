import argparse, json
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from src.features import make_features, Columns

def main(args):
    df = pd.read_csv(args.input)
    y = df[Columns.target]
    X = df.drop(columns=[Columns.target])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = make_features()
    candidates = {
        "linreg": (LinearRegression(), {}),
        "rf": (RandomForestRegressor(random_state=42), {"n_estimators":[200,400]}),
        "gbr": (GradientBoostingRegressor(random_state=42), {"n_estimators":[300,600], "learning_rate":[0.05,0.1]}),
    }

    best_model, best_name, best_rmse = None, None, 1e18
    results = {}
    for name, (est, grid) in candidates.items():
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("pre", pre), ("est", est)])
        if grid:
            gs = GridSearchCV(pipe, param_grid={f"est__{k}":v for k,v in grid.items()}, cv=3, n_jobs=-1)
            gs.fit(X_tr, y_tr)
            mdl = gs.best_estimator_
        else:
            mdl = pipe.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        mae = float(mean_absolute_error(y_te, pred))
        rmse = float(mean_squared_error(y_te, pred, squared=False))
        r2 = float(r2_score(y_te, pred))
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        if rmse < best_rmse:
            best_model, best_name, best_rmse = mdl, name, rmse

    dump(best_model, args.out)
    with open(args.report, "w") as f: json.dump({"results": results, "best": best_name}, f, indent=2)
    print(json.dumps({"results": results, "best": best_name}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    main(p.parse_args())
