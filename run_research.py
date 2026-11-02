"""Research entry point: catalog, plan, ingest, report (no hidden full downloads)."""
import argparse
import json

from src.reef.catalog import DERIVED, LABEL_SOURCE, serializable_catalog
from src.reef.planning import load_config, make_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "plan", "ingest", "report", "labels", "drivers", "model",
                                            "colonies", "cohort", "coralwatch", "survey-design"])
    parser.add_argument("--config", help="Study YAML. survey-design defaults to the full GBR configuration")
    parser.add_argument("--profile")
    parser.add_argument("--root", help="Completed run for report/cohort, or output directory for observations")
    parser.add_argument("--start", help="Inclusive UTC date")
    parser.add_argument("--end", help="Inclusive UTC date")
    parser.add_argument("--common-window", action="store_true",
                        help="Plan or ingest the reviewed intersection of the selected sources")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("SOUTH", "NORTH", "WEST", "EAST"))
    parser.add_argument("--parameters", type=int, default=1_000_000)
    parser.add_argument("--observations", help="Override the command's default observational Parquet table")
    parser.add_argument("--lead", type=int, default=7)
    parser.add_argument("--lookback", type=int, default=14)
    parser.add_argument("--expected-sha256", help="Require this CoralWatch export digest")
    args = parser.parse_args()
    if args.config is None:
        args.config = "configs/gbr_study.yaml" if args.command == "survey-design" else "configs/research.yaml"
    overrides = {"time": {}, "grid": {}}
    for key in ("start", "end"):
        if getattr(args, key):
            overrides["time"][key] = getattr(args, key)
    if args.bbox:
        overrides["grid"].update(dict(zip(["lat_min", "lat_max", "lon_min", "lon_max"], args.bbox)))
    if (args.common_window or any(overrides.values())) and args.command not in ("plan", "ingest", "survey-design"):
        parser.error("Window and grid options apply to plan, ingest and survey-design")
    if args.command == "drivers":
        from pathlib import Path
        output = json.loads((Path(__file__).parent/"src/reef/data/bleaching_drivers.json").read_text(encoding="utf-8"))
    elif args.command == "model":
        from src.reef.learning import build_forecaster
        config = load_config(args.config, args.profile)
        plan = make_plan(config)
        model = build_forecaster(plan["fields"], args.parameters)
        output = {"architecture": model.architecture, "source_fields": plan["feature_names"],
                  "warning": "Capacity preview includes reference/diagnostic fields; choose a scientifically reviewed input contract before training"}
    elif args.command == "catalog":
        from src.reef.colonies import RELEASE
        from src.reef.coralwatch import SOURCE
        output = {"sources": serializable_catalog(), "derived": DERIVED, "labels": LABEL_SOURCE,
                  "additional_observations": {"aims_colony_ordinal_2022": RELEASE, "coralwatch_chart": SOURCE}}
    elif args.command == "report":
        from src.reef.observability import summarize
        if not args.root:
            parser.error("report requires --root")
        output = summarize(args.root)
    elif args.command == "cohort":
        from src.reef.colonies import audit_colony_overlap
        if not args.root:
            parser.error("cohort requires --root pointing to a completed daily run")
        output = audit_colony_overlap(args.root, args.observations or "data/labels_gbr/aims_colonies_2022/observations.parquet",
                                      args.lookback, args.lead)
    elif args.command == "survey-design":
        from src.reef.study import design_chart_study
        config = load_config(args.config, args.profile, overrides, common_window=args.common_window)
        output = design_chart_study(args.observations or "data/labels_gbr/coralwatch/surveys.parquet", config,
                                    args.lookback, args.lead)
    elif args.command == "coralwatch":
        from src.reef.coralwatch import fetch_coralwatch
        config = load_config(args.config, args.profile)
        output = fetch_coralwatch(args.root or "data/labels_gbr", config["execution"], args.expected_sha256)
    elif args.command == "colonies":
        from src.reef.colonies import fetch_colonies
        config = load_config(args.config, args.profile)
        output = fetch_colonies(args.root or "data/labels_gbr", config["execution"])
    elif args.command == "labels":
        from src.reef.labels import fetch_labels
        config = load_config(args.config, args.profile)
        output = fetch_labels(args.root or "data/labels", config["execution"])
    else:
        config = load_config(args.config, args.profile, overrides, common_window=args.common_window)
        if args.command == "plan":
            output = make_plan(config)
        else:
            from src.reef.pipeline import run_pipeline
            output = {"run_directory": str(run_pipeline(config))}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
