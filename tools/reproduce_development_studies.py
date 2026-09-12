"""Regenerate the archived compact temporal figure outside the manuscript."""

import argparse
from importlib.machinery import SourceFileLoader
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "revision").mkdir()
    path = root / "examples/development_studies/sources/import_temporal_study.py.txt"
    loader = SourceFileLoader("archived_temporal_figure", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = args.output
    module.generate(root / "examples/temporal_resolution/data")


if __name__ == "__main__":
    main()
