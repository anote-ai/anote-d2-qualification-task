#!/bin/sh
# Score every image in /app/input and write /app/output/predictions.json,
# then self-check the output against the JHU/APL validator.
set -e
python /app/inference.py --input /app/input --output /app/output/predictions.json "$@"
python /app/validate_predictions.py /app/output/predictions.json || true
