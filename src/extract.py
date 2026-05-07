import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from constants import PORTFOLIO_FILE, PROFILE_FILE, TRANSCRIPT_FILE

log = logging.getLogger(__name__)


def extract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Starbucks JSON files from data/raw."""
    log.info("Loading portfolio from %s", PORTFOLIO_FILE)
    portfolio = pd.read_json(PORTFOLIO_FILE, orient="records", lines=True)
    log.info("  portfolio: %d rows, %d cols", *portfolio.shape)

    log.info("Loading profile from %s", PROFILE_FILE)
    profile = pd.read_json(PROFILE_FILE, orient="records", lines=True)
    log.info("  profile: %d rows, %d cols", *profile.shape)

    log.info("Loading transcript from %s", TRANSCRIPT_FILE)
    transcript = pd.read_json(TRANSCRIPT_FILE, orient="records", lines=True)
    log.info("  transcript: %d rows, %d cols", *transcript.shape)

    return portfolio, profile, transcript


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    extract()
