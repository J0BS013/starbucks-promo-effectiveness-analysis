from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
REPORTS_FIGURES = ROOT_DIR / "reports" / "figures"

PORTFOLIO_FILE = DATA_RAW / "portfolio.json"
PROFILE_FILE = DATA_RAW / "profile.json"
TRANSCRIPT_FILE = DATA_RAW / "transcript.json"

MASTER_TABLE_FILE = DATA_PROCESSED / "master_table.parquet"
OFFER_EVENTS_FILE = DATA_PROCESSED / "offer_events.parquet"
PROFILE_CLEAN_FILE = DATA_PROCESSED / "profile_clean.parquet"

# Statistical thresholds
ALPHA = 0.05
POWER = 0.80

# Feature engineering bins
SNAPSHOT_DATE = "2018-08-01"
AGE_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
INCOME_BINS = [0, 40_000, 60_000, 80_000, 200_000]
INCOME_LABELS = ["low", "mid", "high", "premium"]
