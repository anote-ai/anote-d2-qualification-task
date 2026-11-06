#!/usr/bin/env python3
"""D2 qualification inference: /app/input/* images -> /app/output/predictions.json.

Designed to never leave the evaluator without a schema-valid file: any image that
cannot be decoded or scored still receives four site predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d2.common import CLASSES, FLIP_PERM, MEAN, SITES, STD, load_image, load_model  # noqa: E402

SUBMISSION = {
    "team_name": os.environ.get("D2_TEAM_NAME", "Anote"),
    "version": os.environ.get("D2_VERSION", "1.3.0"),
    "email": os.environ.get("D2_EMAIL", "vidranatan@gmail.com"),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FALLBACK = "no_injury"  # most frequent class at every site in the training data


def letterbox(img: Image.Image, size: int) -> Image.Image:
    img = img.copy()
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def list_images(input_dir: Path) -> list[Path]:
    files = []
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


def site_records(labels: list[str]) -> list[dict]:
    return [
        {"body_region": region, "laterality": side, "injury_type": label}
        for (region, side, _), label in zip(SITES, labels)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="/app/input")
    ap.add_argument("--output", default="/app/output/predictions.json")
    ap.add_argument("--models", default=str(Path(__file__).resolve().parent / "model"),
                    help="a .pt checkpoint or a directory; every *.pt in it is ensembled")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    images = list_images(Path(args.input))
    print(f"[d2] {len(images)} image(s) found in {args.input}", flush=True)
    labels: dict[str, list[str]] = {p.name: [FALLBACK] * len(SITES) for p in images}

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        src = Path(args.models)
        paths = sorted(src.glob("*.pt")) if src.is_dir() else [src]
        models = [load_model(p, device) for p in paths]
        print(f"[d2] {len(models)} model(s) loaded on {device}: "
              + ", ".join(f"{p.name}@{m.size}px" for p, m in zip(paths, models)), flush=True)

        batch_imgs, batch_names = [], []

        def flush():
            if not batch_imgs:
                return
            p = 0
            try:
                with torch.no_grad():
                    for m in models:
                        x = torch.stack([
                            TF.normalize(TF.to_dtype(TF.to_image(letterbox(img, m.size)),
                                                     torch.float32, scale=True), MEAN, STD)
                            for img in batch_imgs]).to(device)
                        p = p + m.predict_proba(x)
                        p = p + m.predict_proba(torch.flip(x, dims=[3]))[:, FLIP_PERM]
                for name, row in zip(batch_names, p.argmax(-1).cpu().tolist()):
                    labels[name] = [CLASSES[i] for i in row]
            except Exception:  # keep fallback labels for this batch only
                print(f"[d2] ERROR scoring batch {batch_names}; using fallback", flush=True)
                traceback.print_exc()
            batch_imgs.clear()
            batch_names.clear()

        for path in images:
            try:
                img = load_image(path)
                img.thumbnail((1024, 1024))  # same pre-resize as the training cache
                batch_imgs.append(img)
                batch_names.append(path.name)
            except Exception as exc:  # undecodable image: keep fallback labels
                print(f"[d2] WARNING could not read {path.name}: {exc}", flush=True)
            if len(batch_imgs) >= args.batch:
                flush()
        flush()
    except Exception:
        print("[d2] ERROR during inference; writing fallback predictions", flush=True)
        traceback.print_exc()

    out = {
        "schema_version": "1.0",
        "submission": SUBMISSION,
        "predictions": [
            {"image_id": p.name, "sites": site_records(labels[p.name])} for p in images
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_path)
    print(f"[d2] wrote {out_path} ({len(images)} images) in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
