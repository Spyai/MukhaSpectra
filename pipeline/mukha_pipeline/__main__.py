"""Command line: run the analysis pipeline without the API.

    python -m mukha_pipeline analyze --demo high --out /tmp/high
    python -m mukha_pipeline analyze --manifest sample_data/synthetic/high/manifest.json --out out/
    python -m mukha_pipeline schema            # print the ScreeningResult JSON Schema

Writes ``result.json`` (the ScreeningResult) and every artifact under ``--out``.
Research prototype — not for clinical use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mukha_pipeline.manifest import ManifestError, build_screening_input
from mukha_pipeline.orchestrator import analyze_screening
from mukha_pipeline.schema.export import render


def _analyze(args: argparse.Namespace) -> int:
    if args.demo:
        from mukha_pipeline.synthetic import make_demo_patient

        inp = make_demo_patient(args.demo, seed=args.seed)
    else:
        manifest_path = Path(args.manifest)
        base = manifest_path.parent
        data = json.loads(manifest_path.read_text(encoding="utf-8"))

        def read_file(field: str) -> tuple[bytes, str] | None:
            matches = sorted(p for p in base.iterdir() if p.is_file() and p.stem == field)
            if not matches:
                return None
            return matches[0].read_bytes(), matches[0].name

        try:
            inp = build_screening_input(data, read_file).screening_input
        except ManifestError as err:
            print(f"error: {err}", file=sys.stderr)
            return 2

    out = analyze_screening(inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(out.result.model_dump_json(indent=2), encoding="utf-8")
    for key, blob in out.artifacts.items():
        path = out_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    t = out.result.triage
    print(f"{out.result.notices.prototype_banner}")
    print(f"Overall risk level: {t.overall_risk_level.value} (score {t.score})")
    print(f"Action: {t.recommended_action.summary}")
    print(f"Why: {t.explanation}")
    print(f"Wrote {out_dir / 'result.json'} and {len(out.artifacts)} artifacts")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mukha_pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    an = sub.add_parser("analyze", help="analyse a demo patient or a manifest folder")
    src = an.add_mutually_exclusive_group(required=True)
    src.add_argument("--demo", choices=["low", "moderate", "high"])
    src.add_argument("--manifest", help="path to manifest.json; files sit next to it")
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--out", required=True)
    sub.add_parser("schema", help="print the ScreeningResult JSON Schema")
    args = parser.parse_args(argv)
    if args.command == "schema":
        sys.stdout.write(render())
        return 0
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
