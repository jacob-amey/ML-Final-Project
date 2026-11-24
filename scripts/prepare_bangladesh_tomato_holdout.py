"""Restore and screen a pinned publisher collection for independent tomato evaluation."""

import argparse
from pathlib import Path

from prepare_tom2024 import DEFAULT as TOM2024
from prepare_tom2024 import (
    MANIFESTS, ROOT, build_registered_release, extend_protected,
    fetch_registered_images, protected_with_okara,
)


NAME = "leafcheck_bangladesh_tomato_v1"
RAW = ROOT / "data/raw/leafcheck_field_v3/bangladesh_tomato"
DEFAULT = ROOT / "data/processed" / NAME
REGISTRY = MANIFESTS / f"{NAME}_files.csv"
SOURCE = MANIFESTS / f"{NAME}_source.json"


def protected_with_tom2024():
    return extend_protected(protected_with_okara(), TOM2024, MANIFESTS / "leafcheck_tom2024_v1_release.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build"))
    parser.add_argument("--output", default=DEFAULT, type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch_registered_images(REGISTRY, RAW)
    else:
        build_registered_release(args.output, REGISTRY, SOURCE, RAW, "bangladesh_tomato", "test",
                                 protected_with_tom2024)
