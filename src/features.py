from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

CATEG = ["Brand", "Processor", "Storage", "GPU"]
NUM = ["RAM", "ScreenSize"]

def make_features():
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(
        transformers=[("cat", cat, CATEG), ("num", num, NUM)],
        remainder="drop",
    )

@dataclass
class Columns:
    target: str = "Price"
