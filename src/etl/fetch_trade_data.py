"""Download NHL trade history (1918-2024) from Kaggle.

Source: kaggle.com/datasets/kpfmma/nhl-trades-dataset
"""

from pathlib import Path

import kagglehub
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET = "kpfmma/nhl-trades-dataset"


def fetch_trade_data() -> pd.DataFrame:
    """Download (or reuse the kagglehub cache) and return the raw trade dataset."""
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {dataset_path}")

    df = pd.read_csv(csv_files[0])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "trades_raw.csv", index=False)
    return df


if __name__ == "__main__":
    df = fetch_trade_data()
    print(f"Downloaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
