import argparse, json, os
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main(args):
    df = pd.read_csv(args.input)
    y = df["Price"]
    X = df.drop(columns=["Price"])
    model = load(args.model)
    pred = model.predict(X)
    mae = float(mean_absolute_error(y, pred))
    rmse = float(mean_squared_error(y, pred, squared=False))
    r2 = float(r2_score(y, pred))

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({"MAE": mae, "RMSE": rmse, "R2": r2}, f, indent=2)

    plt.figure()
    plt.scatter(y, pred, s=10)
    plt.xlabel("True Price"); plt.ylabel("Predicted Price"); plt.title("True vs Pred")
    plt.savefig(os.path.join(args.outdir, "true_vs_pred.png"), bbox_inches="tight")

    plt.figure()
    plt.hist(pred - y, bins=40)
    plt.xlabel("Residuals"); plt.title("Residuals Histogram")
    plt.savefig(os.path.join(args.outdir, "residuals.png"), bbox_inches="tight")

    print(json.dumps({"MAE": mae, "RMSE": rmse, "R2": r2}, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    main(p.parse_args())
