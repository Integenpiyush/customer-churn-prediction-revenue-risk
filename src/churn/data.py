"""Dataset access for local runs and hosted inference."""
from pathlib import Path
import pandas as pd

IBM_DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data/raw/Telco-Customer-Churn.csv"

def load_dataset(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(IBM_DATA_URL)

def download_dataset(path: Path | str = DEFAULT_DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(IBM_DATA_URL)
    frame.to_csv(path, index=False)
    return path

if __name__ == "__main__":
    print(download_dataset())
