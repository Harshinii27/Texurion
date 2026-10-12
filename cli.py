"""Command line interface for single-image texture analysis.

Examples
--------
python cli.py images/brick.jpeg
python cli.py images/brick.jpeg --radius 2 --method nri_uniform --json out.json
"""

import argparse
import json
import os
import sys

from analysis import analyze


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run the LBP texture analysis pipeline on one image.")
    parser.add_argument("image", nargs="?", help="Path to a JPG/JPEG/PNG image.")
    parser.add_argument("--radius", type=int, default=3,
                        help="LBP circle radius (default 3).")
    parser.add_argument("--points", type=int, default=None,
                        help="Number of LBP neighbours P (default 8*radius).")
    parser.add_argument("--method", default="uniform",
                        choices=["uniform", "nri_uniform", "ror", "var", "default"],
                        help="LBP method (default uniform).")
    parser.add_argument("--json", metavar="OUT.json", default=None,
                        help="Also write a JSON snapshot to this path.")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    path = args.image or os.environ.get("LBP_IMAGE")
    if not path:
        parser.error("provide an image path or set LBP_IMAGE")

    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    try:
        result = analyze(path, radius=args.radius, points=args.points,
                         method=args.method)
    except Exception as exc:  # noqa: BLE001
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1

    print(f"Image      : {result.name}")
    print(f"Class      : {result.label} ({result.confidence:.1f}%)")
    print(f"Roughness  : {result.classification['roughness']:.3f}")
    print("Features:")
    for key, value in result.features.items():
        print(f"  {key:<14}: {value:.4f}")

    if args.json:
        snapshot = {
            "source": path,
            "parameters": {
                "radius": result.radius,
                "points": result.points,
                "method": result.method,
                "bins": result.nbins,
            },
            "classification": {
                "label": result.label,
                "confidence": round(result.confidence, 4),
                "roughness": round(result.classification["roughness"], 6),
            },
            "features": {k: round(float(v), 6) for k, v in result.features.items()},
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        print(f"Snapshot : {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())