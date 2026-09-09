"""Download the NHL draft history dataset (1963-2022) from Kaggle.

Source: kaggle.com/datasets/mattop/nhl-draft-hockey-player-data-1963-2022
"""

from pathlib import Path

import kagglehub
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET = "mattop/nhl-draft-hockey-player-data-1963-2022"


def fetch_draft_data() -> pd.DataFrame:
    """Download (or reuse the kagglehub cache) and return the raw draft dataset."""
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in downloaded dataset at {dataset_path}")

    df = pd.read_csv(csv_files[0])

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DIR / "draft_history_raw.csv", index=False)
    return df


if __name__ == "__main__":
    df = fetch_draft_data()
    print(f"Downloaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
