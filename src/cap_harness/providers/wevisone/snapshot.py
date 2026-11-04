"""Read-only validation for a downloaded WeVisOne-VL snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cap_harness.providers.wevisone.model_path import resolve_model_source, verify_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a local WeVisOne-VL model snapshot")
    parser.add_argument("--model", help="snapshot directory; defaults to env/local snapshot")
    args = parser.parse_args()
    path = resolve_model_source(args.model)
    if not Path(path).is_dir():
        parser.error(f"resolved model source is not a local directory: {path}")
    print(json.dumps(verify_snapshot(path), indent=2))


if __name__ == "__main__":
    main()
