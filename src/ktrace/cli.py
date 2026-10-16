"""Command line: ktrace generate | train | report | serve."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger("ktrace")
DEFAULT_MODEL = Path("artifacts/model.json")


def cmd_generate(args: argparse.Namespace) -> None:
    from .simulator import generate_log, make_catalog

    df = generate_log(make_catalog(), args.learners, args.days, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("wrote %d answers from %d learners to %s", len(df), args.learners, args.out)


def cmd_train(args: argparse.Namespace) -> None:
    from .benchmark import run_benchmark
    from .data.loaders import load_log
    from .simulator import generate_log, make_catalog

    if args.data:
        df = load_log(args.data, args.format)
    else:
        df = generate_log(make_catalog(), args.learners, args.days, seed=args.seed)
    bench = run_benchmark(df)
    bench.bundle.save(args.model)
    print(bench.metrics[["auc", "log_loss", "ece"]].round(4).to_string())
    log.info("saved the HLR model bundle to %s", args.model)


def cmd_report(args: argparse.Namespace) -> None:
    from .report import ReportConfig, write

    study = write(args.out, ReportConfig(include_deep=args.deep))
    print(json.dumps(study.bench.metrics["auc"].round(4).to_dict(), indent=2))
    log.info("wrote %s and charts in %s", args.out / "results.md", args.out / "img")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app
    from .predictor import ModelBundle

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found: run `ktrace train` first")
    uvicorn.run(create_app(ModelBundle.load(args.model)), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ktrace", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic practice log as CSV")
    gen.add_argument("--out", type=Path, default=Path("data/synthetic_log.csv"))
    train = sub.add_parser("train", help="fit and score all models, save the HLR bundle")
    train.add_argument("--data", type=Path, help="a CSV log; synthetic data if omitted")
    train.add_argument(
        "--format", choices=["canonical", "assistments", "duolingo"], default="canonical"
    )
    train.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    for p in (gen, train):
        p.add_argument("--learners", type=int, default=400)
        p.add_argument("--days", type=int, default=30)
        p.add_argument("--seed", type=int, default=1)

    report = sub.add_parser("report", help="write docs/results.md and docs/img/*.png")
    report.add_argument("--out", type=Path, default=Path("docs"))
    report.add_argument("--deep", action="store_true", help="include DKT (needs .[deep])")

    serve = sub.add_parser("serve", help="run the API")
    serve.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    gen.set_defaults(func=cmd_generate)
    train.set_defaults(func=cmd_train)
    report.set_defaults(func=cmd_report)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
