"""
Run this once before opening the notebooks.
Loads raw JSON files, cleans and merges them, saves to data/processed/.

Usage:
    python main.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from extract import extract
from transform import transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def main() -> None:
    log.info("=== Extract ===")
    portfolio, profile, transcript = extract()

    log.info("=== Transform ===")
    tables = transform(portfolio, profile, transcript)

    log.info("=== Save ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("  %s: %d rows → %s", name, len(df), path.name)

    master = tables["master_table"]
    log.info("Done — %d rows, %d customers, %.1f%% conversion",
             len(master), master["person"].nunique(), master["completed"].mean() * 100)


if __name__ == "__main__":
    main()
