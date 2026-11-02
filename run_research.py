"""Research entry point: catalog, plan, ingest, report (no hidden full downloads)."""
import argparse
import json

from src.reef.catalog import DERIVED, LABEL_SOURCE, serializable_catalog
from src.reef.planning import load_config, make_plan, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["catalog", "plan", "ingest", "report", "labels"])
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--profile")
    parser.add_argument("--root", help="A completed run directory for report")
    parser.add_argument("--start", help="Inclusive UTC date")
    parser.add_argument("--end", help="Inclusive UTC date")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("SOUTH", "NORTH", "WEST", "EAST"))
    args = parser.parse_args()
    if args.command == "catalog":
        output = {"sources": serializable_catalog(), "derived": DERIVED, "labels": LABEL_SOURCE}
    elif args.command == "report":
        from src.reef.observability import summarize
        if not args.root:
            parser.error("report requires --root")
        output = summarize(args.root)
    elif args.command == "labels":
        from src.reef.labels import fetch_labels
        config = load_config(args.config, args.profile)
        output = fetch_labels(args.root or "data/labels", config["execution"])
    else:
        overrides = {"time": {}, "grid": {}}
        if args.start:
            overrides["time"]["start"] = args.start
        if args.end:
            overrides["time"]["end"] = args.end
        if args.bbox:
            overrides["grid"].update(dict(zip(["lat_min", "lat_max", "lon_min", "lon_max"], args.bbox)))
        config = load_config(args.config, args.profile, overrides)
        if args.command == "plan":
            output = make_plan(config)
        else:
            from src.reef.pipeline import run_pipeline
            output = {"run_directory": str(run_pipeline(config))}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
