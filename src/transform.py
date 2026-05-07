import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from constants import (
    SNAPSHOT_DATE,
    AGE_BINS,
    AGE_LABELS,
    INCOME_BINS,
    INCOME_LABELS,
    DATA_PROCESSED,
)

log = logging.getLogger(__name__)


def clean_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid records and engineer demographic features."""
    df = profile.copy()
    before = len(df)
    # age=118 is a sentinel for missing demographic data in this dataset
    df = df[df["age"] != 118].copy()
    log.info("  Profile: removed %d rows with age=118 sentinel", before - len(df))

    df["became_member_on"] = pd.to_datetime(df["became_member_on"], format="%Y%m%d")
    snapshot = pd.Timestamp(SNAPSHOT_DATE)
    df["membership_days"] = (snapshot - df["became_member_on"]).dt.days

    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["income_group"] = pd.cut(df["income"], bins=INCOME_BINS, labels=INCOME_LABELS)

    log.info("  Profile clean: %d customers", len(df))
    return df


def expand_transcript_value(transcript: pd.DataFrame) -> pd.DataFrame:
    """Unpack the 'value' dict column into flat columns.

    The transcript uses two key formats:
    - 'offer id' (with space) for offer received/viewed events
    - 'offer_id' (with underscore) for offer completed events
    Both are normalized to a single 'offer_id' column.
    """
    df = transcript.copy()
    df["offer_id"] = df["value"].apply(
        lambda x: x.get("offer id") or x.get("offer_id")
    )
    df["amount"] = df["value"].apply(lambda x: x.get("amount", np.nan))
    df["reward"] = df["value"].apply(lambda x: x.get("reward", np.nan))
    return df.drop(columns=["value"])


def build_offer_events(transcript: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (customer, offer) with received/viewed/completed flags."""
    df = expand_transcript_value(transcript)

    received = (
        df[df["event"] == "offer received"][["person", "offer_id", "time"]]
        .rename(columns={"time": "time_received"})
    )
    viewed = (
        df[df["event"] == "offer viewed"][["person", "offer_id", "time"]]
        .rename(columns={"time": "time_viewed"})
    )
    completed = (
        df[df["event"] == "offer completed"][["person", "offer_id", "time", "reward"]]
        .rename(columns={"time": "time_completed"})
    )

    events = received.merge(viewed, on=["person", "offer_id"], how="left")
    events = events.merge(completed, on=["person", "offer_id"], how="left")

    events["viewed"] = events["time_viewed"].notna()
    events["completed"] = events["time_completed"].notna()

    log.info("  Offer events: %d rows (%d unique customers)", len(events), events["person"].nunique())
    return events


def transform(
    portfolio: pd.DataFrame,
    profile: pd.DataFrame,
    transcript: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run all transformations and return named DataFrames."""
    log.info("Cleaning profile...")
    profile_clean = clean_profile(profile)

    log.info("Building offer events table...")
    offer_events = build_offer_events(transcript)

    log.info("Building master table...")
    master = (
        offer_events
        .merge(profile_clean, left_on="person", right_on="id", how="left")
        .merge(portfolio, left_on="offer_id", right_on="id", how="left", suffixes=("", "_offer"))
    )
    master = master.drop(columns=["id", "id_offer"], errors="ignore")
    log.info("  Master table: %d rows", len(master))

    log.info("  Conversion rate (overall): %.1f%%", master["completed"].mean() * 100)
    log.info("  View rate (overall): %.1f%%", master["viewed"].mean() * 100)
    log.info(
        "  Offer type distribution:\n%s",
        master["offer_type"].value_counts().to_string(),
    )

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    return {
        "master_table": master,
        "offer_events": offer_events,
        "profile_clean": profile_clean,
    }
