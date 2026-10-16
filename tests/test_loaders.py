import pytest

from ktrace.data.loaders import load_log, part_of_speech

ASSISTMENTS_CSV = """order_id,assignment_id,user_id,problem_id,original,correct,skill_id,skill_name
33,1,70363,51424,1,1,10,Box and Whisker
31,1,70363,51425,1,0,10,Box and Whisker
31,1,70363,51425,1,0,11,Circle Graph
40,1,70364,51424,1,1,,
45,1,70364,51430,1,1,11,Circle Graph
"""

DUOLINGO_HEADER = (
    "p_recall,timestamp,delta,user_id,learning_language,ui_language,lexeme_id,lexeme_string,"
    "history_seen,history_correct,session_seen,session_correct\n"
)
DUOLINGO_CSV = (
    DUOLINGO_HEADER
    + """1.0,1362076081,27649635,u:FO,de,en,76390c13,lernt/lernen<vblex><pri><p3><sg>,6,4,2,2
0.5,1362076081,27649635,u:FO,de,en,7dfd7086,die/die<det><def><f><sg><nom>,4,4,2,1
1.0,1362082032,444407,u:FO,de,en,35a54c25,mann/mann<n><m><sg><nom>,5,4,1,1
"""
)


def test_assistments_dedupes_multi_skill_rows_and_orders_by_order_id(tmp_path):
    path = tmp_path / "skill_builder_data.csv"
    path.write_text(ASSISTMENTS_CSV)
    df = load_log(path, "assistments")
    assert list(df.columns) == ["user_id", "item_id", "skill_id", "t", "correct"]
    # order_id 31 appears twice (two skills) -> kept once; the row without a skill is dropped.
    assert df["item_id"].tolist() == ["51425", "51424", "51430"]
    assert df["skill_id"].tolist() == ["10", "10", "11"]
    assert df["t"].is_monotonic_increasing


def test_duolingo_maps_sessions_to_binary_answers(tmp_path):
    path = tmp_path / "learning_traces.csv"
    path.write_text(DUOLINGO_CSV)
    df = load_log(path, "duolingo")
    assert df["correct"].tolist() == [1, 0, 1]
    assert df["skill_id"].tolist() == ["de:vblex", "de:det", "de:n"]
    assert df["t"].iloc[2] - df["t"].iloc[0] == pytest.approx((1362082032 - 1362076081) / 86400)
    assert part_of_speech("no tags here") == "unknown"


def test_missing_columns_and_unknown_format_fail_loudly(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("user_id,correct\n1,0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_log(path, "assistments")
    with pytest.raises(ValueError, match="unknown format"):
        load_log(path, "kaggle")
