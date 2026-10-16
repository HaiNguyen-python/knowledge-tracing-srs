"""Loaders that turn public knowledge-tracing datasets into this repo's log schema.

The canonical log has one row per answer:

    user_id (str), item_id (str), skill_id (str), t (float, days), correct (0/1)

The tests and the report use the synthetic simulator only (no downloads in CI). These loaders
let the real data be dropped in; see README "Real data".
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CANONICAL = ["user_id", "item_id", "skill_id", "t", "correct"]
ASSISTMENTS_COLUMNS = ["order_id", "user_id", "problem_id", "skill_id", "correct"]
DUOLINGO_COLUMNS = [
    "timestamp",
    "user_id",
    "learning_language",
    "lexeme_id",
    "lexeme_string",
    "session_seen",
    "session_correct",
]
_POS_TAG = re.compile(r"<([^>]+)>")


def _require(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def load_assistments(path: Path) -> pd.DataFrame:
    """ASSISTments 2009-2010 skill-builder CSV (`skill_builder_data.csv`).

    * Rows without a skill are dropped (they cannot be traced per skill).
    * Problems tagged with several skills appear once per skill; the first row of each
      `order_id` is kept so an answer is not counted twice.
    * The file has no reliable timestamps, so `order_id` gives the order and `t` is a step
      counter (one minute per answer). Time-based features (HLR's `delta_days`) are therefore
      not meaningful on this dataset; BKT and PFA are.
    """
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    _require(df, ASSISTMENTS_COLUMNS, "ASSISTments")
    df = df.dropna(subset=["skill_id"])
    df = df[df["correct"].isin([0, 1])]
    df = df.sort_values("order_id", kind="stable").drop_duplicates("order_id")
    out = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str),
            "item_id": df["problem_id"].astype(str),
            "skill_id": df["skill_id"].astype("int64").astype(str),
            "t": pd.Series(range(len(df)), index=df.index, dtype=float) / 1440.0,
            "correct": df["correct"].astype(int),
        }
    )
    return out.reset_index(drop=True)


def part_of_speech(lexeme_string: str) -> str:
    """First tag in a Duolingo lexeme string, e.g. 'lernt/lernen<vblex><pri><p3><sg>' -> 'vblex'."""
    match = _POS_TAG.search(lexeme_string)
    return match.group(1) if match else "unknown"


def load_duolingo_hlr(path: Path) -> pd.DataFrame:
    """Duolingo half-life regression traces (Settles & Meeder 2016, `learning_traces.13m.csv`).

    * One row per practice session of a word; `correct` = 1 when every answer in the session
      was right (the paper uses the fraction `p_recall`; a binary label keeps one schema).
    * `skill_id` = language + part of speech, e.g. 'de:vblex'.
    * `t` = unix timestamp in days. The file is a sample of sessions, so counts built from it
      undercount a learner's real history (the file's own `history_seen` covers all of it).
    """
    df = pd.read_csv(path)
    _require(df, DUOLINGO_COLUMNS, "Duolingo HLR")
    skills = df["learning_language"] + ":" + df["lexeme_string"].map(part_of_speech)
    out = pd.DataFrame(
        {
            "user_id": df["user_id"].astype(str),
            "item_id": df["learning_language"] + ":" + df["lexeme_id"].astype(str),
            "skill_id": skills,
            "t": df["timestamp"].astype(float) / 86400.0,
            "correct": (df["session_correct"] == df["session_seen"]).astype(int),
        }
    )
    return out.reset_index(drop=True)


def load_log(path: Path, fmt: str) -> pd.DataFrame:
    """Load a log in one of the known formats: 'canonical', 'assistments' or 'duolingo'."""
    if fmt == "assistments":
        return load_assistments(path)
    if fmt == "duolingo":
        return load_duolingo_hlr(path)
    if fmt == "canonical":
        df = pd.read_csv(path, dtype={"user_id": str, "item_id": str, "skill_id": str})
        _require(df, CANONICAL, "canonical log")
        return df
    raise ValueError(f"unknown format {fmt!r}")
