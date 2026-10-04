"""Assemble actual unconstrained same-engine repetitions for run.py quality checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def assemble(paths):
    records = [json.loads(path.read_text()) for path in paths]
    if len(records) < 3:
        raise ValueError(
            "at least three actual unconstrained cell repetitions required"
        )
    identity = records[0]["reference_identity"]
    for record in records:
        if (
            not record.get("reference_only")
            or not record["ok"]
            or record["reference_identity"] != identity
        ):
            raise ValueError(
                "reference must be successful, unconstrained and identically authored"
            )
        if len(record["requests"]) != 6 or any(
            not row["fresh_sampling_evidence"] for row in record["requests"]
        ):
            raise ValueError("all six references need actual fresh sampling evidence")
    return [
        {
            "identity": identity,
            "artifacts": [record["requests"][n]["artifacts"][0] for record in records],
        }
        for n in range(6)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.write_text(json.dumps(assemble(a.results), indent=2) + "\n")


if __name__ == "__main__":
    main()
