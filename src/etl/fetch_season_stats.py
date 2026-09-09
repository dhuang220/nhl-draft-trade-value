"""Download NHL player season-by-season stats from Kaggle (MoneyPuck-derived data).

Source: kaggle.com/datasets/mexwell/nhl-database
Used for computing a player's "trailing value" at the time of a trade.
"""

from pathlib import Path

import kagglehub
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET = "mexwell/nhl-database"


def fetch_season_stats() -> dict[str, pd.DataFrame]:
    """Download (or reuse the kagglehub cache) and return all CSVs in the dataset.

    We don't yet know which file(s) hold per-season skater stats, so this loads
    everything and hands it back keyed by filename - inspect the output and we'll
    narrow this down to the specific file/columns we need in the next step.
    """
    dataset_path = Path(kagglehub.dataset_download(DATASET))
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in downloaded dataset at {dataset_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        tables[csv_file.stem] = df
        df.to_csv(RAW_DIR / f"season_stats_{csv_file.stem}.csv", index=False)
    return tables


if __name__ == "__main__":
    tables = fetch_season_stats()
    for name, df in tables.items():
        print(f"\n=== {name} ===")
        print(f"{len(df):,} rows, columns: {list(df.columns)}")
        print(df.head(3))
