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


def build_offer_events(
    transcript: pd.DataFrame, portfolio: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build one row per received offer with temporally valid attribution.

    Each transcript event is assigned to the latest still-valid exposure for
    the same customer and offer. This prevents the many-to-many joins caused
    by repeated sends. Starbucks transcript time is in hours; portfolio
    duration is in days. Omitting portfolio enables a no-expiry fixture mode.
    """
    df = expand_transcript_value(transcript).copy()
    df = df[df["event"].isin(["offer received", "offer viewed", "offer completed"])]
    df = df.sort_values("time", kind="stable").reset_index(drop=True)

    received = df[df["event"] == "offer received"][["person", "offer_id", "time"]].copy()
    received = received.rename(columns={"time": "time_received"})
    received["_sequence"] = received.groupby(["person", "offer_id"]).cumcount() + 1
    received["exposure_id"] = (
        received["person"].astype(str) + "::" + received["offer_id"].astype(str)
        + "::" + received["time_received"].astype(str) + "::" + received["_sequence"].astype(str)
    )
    if portfolio is None:
        received["valid_until"] = np.inf
    else:
        durations = portfolio[["id", "duration"]].rename(columns={"id": "offer_id"})
        received = received.merge(durations, on="offer_id", how="left", validate="many_to_one")
        if received["duration"].isna().any():
            missing = received.loc[received["duration"].isna(), "offer_id"].unique()
            raise ValueError(f"Missing duration for offer IDs: {missing.tolist()}")
        received["valid_until"] = received["time_received"] + received["duration"] * 24
        received = received.drop(columns="duration")

    events = received[["exposure_id", "person", "offer_id", "time_received", "valid_until"]].copy()
    events["time_viewed"] = np.nan
    events["time_completed"] = np.nan
    events["reward"] = 0.0
    for event_type, time_column in (("offer viewed", "time_viewed"), ("offer completed", "time_completed")):
        for _, event in df[df["event"] == event_type].sort_values("time", kind="stable").iterrows():
            eligible = events[
                (events["person"] == event["person"])
                & (events["offer_id"] == event["offer_id"])
                & (events["time_received"] <= event["time"])
                & (event["time"] <= events["valid_until"])
                & events[time_column].isna()
            ]
            if eligible.empty:
                continue
            exposure_index = eligible.sort_values("time_received", kind="stable").index[-1]
            events.loc[exposure_index, time_column] = event["time"]
            if event_type == "offer completed":
                events.loc[exposure_index, "reward"] = event["reward"]

    events["viewed"] = events["time_viewed"].notna()
    events["completed"] = events["time_completed"].notna()
    events["reward"] = events["reward"].fillna(0.0)
    if not events["exposure_id"].is_unique:
        raise AssertionError("offer_exposure grain violated: exposure_id is not unique")
    log.info("  Offer exposures: %d rows (%d unique customers)", len(events), events["person"].nunique())
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
    offer_events = build_offer_events(transcript, portfolio)

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
