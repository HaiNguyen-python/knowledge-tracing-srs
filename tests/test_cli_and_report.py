import json

from ktrace.cli import main
from ktrace.predictor import ModelBundle
from ktrace.report import ReportConfig, write
from ktrace.scheduler import CourseConfig


def test_generate_then_train_on_the_csv_saves_a_loadable_bundle(tmp_path):
    csv, model = tmp_path / "log.csv", tmp_path / "model.json"
    main(["generate", "--out", str(csv), "--learners", "60", "--days", "12"])
    main(["train", "--data", str(csv), "--model", str(model)])
    bundle = ModelBundle.load(model)
    assert json.loads(model.read_text())["model"]["w"]["sqrt_succ_item"] > 0
    assert len(bundle.difficulty) > 30


def test_report_writes_results_and_both_charts(tmp_path):
    cfg = ReportConfig(
        n_learners=80,
        n_days=14,
        scheduler_seeds=(1, 2),
        course=CourseConfig(n_learners=3, n_days=10),
        budgets=(5, 20),
    )
    write(tmp_path, cfg)
    text = (tmp_path / "results.md").read_text()
    assert "Synthetic data" in text and "Model-based (HLR)" in text
    assert "| 5 | " in text and "| 20 | " in text
    for name in ("model_comparison.png", "retention_by_scheduler.png"):
        assert (tmp_path / "img" / name).stat().st_size > 10_000
