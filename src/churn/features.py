from __future__ import annotations
import pandas as pd

TARGET = "Churn"
DROP = ["customerID", TARGET]

def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw IBM rows and derive stable features before encoding."""
    x = frame.drop(columns=[c for c in DROP if c in frame.columns]).copy()
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in x:
            x[col] = pd.to_numeric(x[col].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
    if "tenure" in x:
        x["tenure_group"] = pd.cut(x.tenure, [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"]).astype("object")
        x["month_to_month"] = (x.get("Contract") == "Month-to-month").astype(int)
        x["tenure_months_plus1"] = x.tenure.fillna(0) + 1
        if "TotalCharges" in x:
            x["charge_per_tenure_month"] = x.TotalCharges / x.tenure_months_plus1
    if "MonthlyCharges" in x:
        x["monthly_charge_segment"] = pd.cut(x.MonthlyCharges, [-1, 35, 65, 90, 150], labels=["low", "mid", "high", "very_high"]).astype("object")
    service_cols = [c for c in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"] if c in x]
    if service_cols:
        x["service_count"] = x[service_cols].apply(lambda s: s.isin(["Yes", "Yes - DSL", "Yes - Fiber optic"]).sum(), axis=1)
    x = x.drop(columns=["tenure_months_plus1"], errors="ignore")
    return x

def clean_target(y: pd.Series) -> pd.Series:
    return y.map({"No": 0, "Yes": 1}).astype(int)
