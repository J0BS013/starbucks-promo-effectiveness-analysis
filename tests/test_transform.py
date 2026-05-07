import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transform import clean_profile, build_offer_events, expand_transcript_value


@pytest.fixture
def raw_profile():
    return pd.DataFrame({
        "id": ["c1", "c2", "c3", "c4"],
        "age": [30, 118, 45, 25],
        "gender": ["M", "M", "F", "F"],
        "income": [50000.0, 60000.0, 80000.0, 35000.0],
        "became_member_on": [20170101, 20160601, 20180101, 20150901],
    })


@pytest.fixture
def raw_transcript():
    return pd.DataFrame({
        "person": ["c1", "c1", "c1", "c2", "c2"],
        "event": ["offer received", "offer viewed", "offer completed", "offer received", "offer received"],
        "time": [0, 6, 18, 0, 0],
        "value": [
            {"offer id": "o1"},
            {"offer id": "o1"},
            {"offer_id": "o1", "reward": 5},
            {"offer id": "o2"},
            {"offer id": "o3"},
        ],
    })


class TestCleanProfile:
    def test_removes_sentinel_age(self, raw_profile):
        result = clean_profile(raw_profile)
        assert 118 not in result["age"].values

    def test_keeps_valid_rows(self, raw_profile):
        result = clean_profile(raw_profile)
        assert len(result) == 3

    def test_age_group_created(self, raw_profile):
        result = clean_profile(raw_profile)
        assert "age_group" in result.columns
        assert result["age_group"].notna().all()

    def test_income_group_created(self, raw_profile):
        result = clean_profile(raw_profile)
        assert "income_group" in result.columns

    def test_membership_days_positive(self, raw_profile):
        result = clean_profile(raw_profile)
        assert (result["membership_days"] > 0).all()


class TestExpandTranscriptValue:
    def test_offer_id_normalized(self, raw_transcript):
        result = expand_transcript_value(raw_transcript)
        # c1's completed event uses 'offer_id' key — must be normalized
        completed = result[result["event"] == "offer completed"]
        assert completed["offer_id"].iloc[0] == "o1"

    def test_amount_nan_for_non_transactions(self, raw_transcript):
        result = expand_transcript_value(raw_transcript)
        assert result[result["event"] != "transaction"]["amount"].isna().all()

    def test_reward_extracted(self, raw_transcript):
        result = expand_transcript_value(raw_transcript)
        completed = result[result["event"] == "offer completed"]
        assert completed["reward"].iloc[0] == 5


class TestBuildOfferEvents:
    def test_viewed_flag(self, raw_transcript):
        events = build_offer_events(raw_transcript)
        c1_o1 = events[(events["person"] == "c1") & (events["offer_id"] == "o1")]
        assert c1_o1["viewed"].iloc[0] == True

    def test_not_viewed_flag(self, raw_transcript):
        events = build_offer_events(raw_transcript)
        c2_o2 = events[(events["person"] == "c2") & (events["offer_id"] == "o2")]
        assert c2_o2["viewed"].iloc[0] == False

    def test_completed_flag(self, raw_transcript):
        events = build_offer_events(raw_transcript)
        c1_o1 = events[(events["person"] == "c1") & (events["offer_id"] == "o1")]
        assert c1_o1["completed"].iloc[0] == True

    def test_not_completed_flag(self, raw_transcript):
        events = build_offer_events(raw_transcript)
        c2_o2 = events[(events["person"] == "c2") & (events["offer_id"] == "o2")]
        assert c2_o2["completed"].iloc[0] == False
