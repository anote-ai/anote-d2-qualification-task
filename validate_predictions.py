#!/usr/bin/env python3
"""Validate a D2C injury-classification predictions JSON file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BODY_REGIONS = {"upper_extremity", "lower_extremity"}
LATERALITIES = {"left", "right"}
INJURY_TYPES = {"wound", "amputation", "no_injury", "not_testable"}
EXPECTED_SITES = {(region, side) for region in BODY_REGIONS for side in LATERALITIES}
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def validate_predictions(data: Any) -> list[str]:
    """Return all validation errors; an empty list means the file is valid."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["$: top-level value must be a JSON object"]

    _check_exact_keys(data, {"schema_version", "submission", "predictions"}, "$", errors)

    if data.get("schema_version") != "1.0":
        errors.append("$.schema_version: must equal '1.0'")

    submission = data.get("submission")
    if not isinstance(submission, dict):
        errors.append("$.submission: must be an object")
    else:
        required = {"team_name", "version", "email"}
        _check_exact_keys(submission, required, "$.submission", errors)
        for field in required:
            value = submission.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"$.submission.{field}: must be a non-empty string")
        email = submission.get("email")
        if isinstance(email, str) and email.strip() and not EMAIL_PATTERN.fullmatch(email):
            errors.append("$.submission.email: must be a plain email address, e.g. team@example.com")

    predictions = data.get("predictions")
    if not isinstance(predictions, list):
        errors.append("$.predictions: must be a list")
        return errors
    if not predictions:
        errors.append("$.predictions: must contain at least one image")

    seen_images: set[str] = set()
    for image_index, image in enumerate(predictions):
        image_path = f"$.predictions[{image_index}]"
        if not isinstance(image, dict):
            errors.append(f"{image_path}: must be an object")
            continue

        _check_exact_keys(image, {"image_id", "sites"}, image_path, errors)
        image_id = image.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            errors.append(f"{image_path}.image_id: must be a non-empty string")
        elif image_id in seen_images:
            errors.append(f"{image_path}.image_id: duplicate image_id {image_id!r}")
        else:
            seen_images.add(image_id)

        sites = image.get("sites")
        if not isinstance(sites, list):
            errors.append(f"{image_path}.sites: must be a list")
            continue

        seen_sites: set[tuple[str, str]] = set()
        for site_index, site in enumerate(sites):
            site_path = f"{image_path}.sites[{site_index}]"
            if not isinstance(site, dict):
                errors.append(f"{site_path}: must be an object")
                continue

            _check_exact_keys(
                site, {"body_region", "laterality", "injury_type"}, site_path, errors
            )
            region = site.get("body_region")
            side = site.get("laterality")
            injury = site.get("injury_type")

            _check_enum(region, BODY_REGIONS, f"{site_path}.body_region", errors)
            _check_enum(side, LATERALITIES, f"{site_path}.laterality", errors)
            _check_enum(injury, INJURY_TYPES, f"{site_path}.injury_type", errors)

            if isinstance(region, str) and isinstance(side, str):
                site_key = (region, side)
                if site_key in seen_sites:
                    errors.append(f"{site_path}: duplicate site {region}/{side}")
                seen_sites.add(site_key)

        missing = EXPECTED_SITES - seen_sites
        extra = seen_sites - EXPECTED_SITES
        if missing:
            errors.append(
                f"{image_path}.sites: missing required sites: "
                + ", ".join(f"{region}/{side}" for region, side in sorted(missing))
            )
        if extra:
            errors.append(
                f"{image_path}.sites: unexpected sites: "
                + ", ".join(f"{region}/{side}" for region, side in sorted(extra))
            )

    return errors


def _check_exact_keys(
    value: dict[str, Any], expected: set[str], path: str, errors: list[str]
) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        errors.append(f"{path}: missing required fields: {', '.join(sorted(missing))}")
    if extra:
        errors.append(f"{path}: unexpected fields: {', '.join(sorted(extra))}")


def _check_enum(value: Any, allowed: set[str], path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        errors.append(f"{path}: must be one of: {', '.join(sorted(allowed))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Predictions JSON file to validate")
    args = parser.parse_args()

    try:
        data = json.loads(args.predictions.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"INVALID: file not found: {args.predictions}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"INVALID: could not read {args.predictions}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"INVALID JSON: line {exc.lineno}, column {exc.colno}: {exc.msg}", file=sys.stderr)
        return 1

    errors = validate_predictions(data)
    if errors:
        print(f"INVALID: {len(errors)} validation error(s)")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"VALID: {args.predictions} contains "
        f"{len(data['predictions'])} image prediction(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())