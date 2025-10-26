from src.features import make_features
import pandas as pd

def test_pipeline_fits():
    df = pd.DataFrame({
        "Brand":["A","B"], "Processor":["i5","i7"], "Storage":["512 SSD","1TB SSD"],
        "GPU":["Intel","NVIDIA"], "RAM":[8,16], "ScreenSize":[13.3,15.6], "Price":[700,1200]
    })
    X = df.drop(columns=["Price"]); y = df["Price"]
    pre = make_features(); Xtr = pre.fit_transform(X, y)
    assert Xtr.shape[0] == len(y)
