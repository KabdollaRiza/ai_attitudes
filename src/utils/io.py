from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

def read_csv(path: str) -> pd.DataFrame:
    """Load CSV with utf-8 encoding."""
    return pd.read_csv(path, encoding="utf-8", dtype=str).fillna("")

def write_csv(df: pd.DataFrame, path: str) -> None:
    """Write CSV to disk, creating parents if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8")
