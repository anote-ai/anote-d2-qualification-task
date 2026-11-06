#!/usr/bin/env python3
"""Train the D2 four-site injury classifier.

Example:
    python train.py --csv data/train/train_dataset.csv --images data/train/images \
        --pretrained weights/tf_efficientnet_b3_ns-9d44bf68.pth --out model/model.pt

Use --cv-folds N to run casualty-grouped cross-validation instead of a final fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from d2.common import CLASSES, FLIP_PERM, MEAN, SITES, STD, SiteClassifier, load_image


def letterbox(img, size):
    img = img.copy()
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def load_dataset(csv_path, image_dir, size, media="image"):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["media_format"] == media]
    xs, ys, groups, names = [], [], [], []
    for r in rows:
        p = Path(image_dir) / os.path.basename(r["filepath"])
        if not p.exists():
            print(f"skip missing {p.name}")
            continue
        img = letterbox(load_image(p), size)
        xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
        ys.append([CLASSES.index(r[col]) for _, _, col in SITES])
        groups.append(r["casualty_id"])
        names.append(p.name)
    return torch.stack(xs), torch.tensor(ys), np.array(groups), names


def make_aug():
    return T.Compose([
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ColorJitter(0.3, 0.3, 0.3, 0.03),
    ])


def to_input(x_uint8):
    x = x_uint8.float() / 255.0
    return T.functional.normalize(x, MEAN, STD)


def predict(model, x, batch=16):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = to_input(x[i:i + batch])
            p = model(xb).softmax(-1)
            pf = model(torch.flip(xb, dims=[3])).softmax(-1)[:, FLIP_PERM]
            out.append((p + pf) / 2)
    return torch.cat(out)


NT = CLASSES.index("not_testable")


def site_loss(logits, y, partial, class_w, smoothing=0.05):
    """Cross-entropy per site. For `partial` samples (video frames, whose labels describe
    the casualty rather than the frame) a limb may be out of view, so the target set is
    {label, not_testable}: loss = -log(p[label] + p[not_testable])."""
    logits = logits.reshape(-1, len(CLASSES))
    y = y.reshape(-1)
    part = partial.repeat_interleave(len(SITES))
    ce = F.cross_entropy(logits, y, weight=class_w, label_smoothing=smoothing, reduction="none")
    if part.any():
        logp = logits.log_softmax(-1)
        lp = torch.logaddexp(logp.gather(1, y[:, None])[:, 0], logp[:, NT])
        lp = torch.where(y == NT, logp[:, NT], lp)
        w = class_w[y] if class_w is not None else torch.ones_like(lp)
        ce = torch.where(part, -lp * w, ce)
    return ce.mean()


def sample_weights(y, groups, partial, args):
    w = torch.ones(len(y))
    if args.balance_casualty:
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        w = w / torch.tensor(counts[inv], dtype=torch.float).sqrt()
    if args.video_weight != 1.0:
        w = torch.where(partial, w * args.video_weight, w)
    return w / w.sum()


def fit(x, y, args, x_val=None, y_val=None, groups=None, partial=None):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if partial is None:
        partial = torch.zeros(len(x), dtype=torch.bool)
    if groups is None:
        groups = np.zeros(len(x))
    class_w = None
    if args.class_weight:
        cnt = torch.bincount(y[~partial].reshape(-1), minlength=len(CLASSES)).float()
        class_w = (cnt.sum() / cnt.clamp(min=1)) ** args.class_weight
        class_w = class_w / class_w.mean()
        print("class weights", [round(v, 2) for v in class_w.tolist()], flush=True)
    sw = sample_weights(y, groups, partial, args)
    model = SiteClassifier(args.backbone, args.pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * (((args.epoch_size or len(x)) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    aug = make_aug()
    start = 0
    if args.ckpt and os.path.exists(args.ckpt):
        st = torch.load(args.ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start = st["epoch"]
        print(f"resumed from {args.ckpt} at epoch {start}", flush=True)
    for epoch in range(start, args.epochs):
        torch.manual_seed(args.seed * 1000 + epoch)
        model.train()
        perm = torch.multinomial(sw, args.epoch_size or len(x), replacement=True)
        t0, tot = time.time(), 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb, pb = x[idx].clone(), y[idx].clone(), partial[idx]
            flip = torch.rand(len(idx)) < 0.5
            xb[flip] = torch.flip(xb[flip], dims=[3])
            yb[flip] = yb[flip][:, FLIP_PERM]
            xb = torch.stack([aug(img) for img in xb])
            logits = model(to_input(xb))
            loss = site_loss(logits, yb, pb, class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        msg = f"epoch {epoch + 1}/{args.epochs} loss {tot / len(perm):.4f} {time.time() - t0:.0f}s"
        if x_val is not None and (epoch + 1) % args.eval_every == 0:
            m = metrics(predict(model, x_val), y_val)
            msg += f" val_acc {m['accuracy']:.4f} val_macroF1 {m['macro_f1']:.4f}"
        print(msg, flush=True)
        if args.ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch + 1}, args.ckpt + ".tmp")
            os.replace(args.ckpt + ".tmp", args.ckpt)
    return model


def metrics(probs, y):
    pred = probs.argmax(-1).numpy().reshape(-1)
    true = y.numpy().reshape(-1)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro")),
        "per_class_f1": dict(zip(CLASSES, f1_score(true, pred, average=None, labels=range(4)).tolist())),
        "image_all_correct": float((probs.argmax(-1) == y).all(-1).float().mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--backbone", default="tf_efficientnet_b3.ns_jft_in1k")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=0)
    ap.add_argument("--holdout-only", action="store_true",
                    help="with --cv-folds, run only the first casualty-grouped fold")
    ap.add_argument("--holdout-seed", type=int, default=None,
                    help="hold out ~20%% of casualties (never the 4 largest) chosen with this seed")
    ap.add_argument("--video-frames", default=None,
                    help="dir of extracted video frames + frames.csv (same schema, media_format=frame)")
    ap.add_argument("--epoch-size", type=int, default=0,
                    help="samples drawn per epoch (0 = dataset size)")
    ap.add_argument("--video-weight", type=float, default=1.0)
    ap.add_argument("--balance-casualty", action="store_true",
                    help="sample images with weight 1/sqrt(images of that casualty)")
    ap.add_argument("--class-weight", type=float, default=0.0,
                    help="exponent for inverse-frequency class weights (0 = off, 0.5 = sqrt)")
    ap.add_argument("--ckpt", default=None, help="per-epoch resume checkpoint path")
    ap.add_argument("--save-probs", default=None, help="holdout mode: save val probabilities here")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--out", default="model/model.pt")
    args = ap.parse_args()

    x, y, groups, _ = load_dataset(args.csv, args.images, args.size)
    print(f"loaded {len(x)} images, {len(set(groups))} casualties", flush=True)
    partial = torch.zeros(len(x), dtype=torch.bool)
    if args.video_frames:
        xv, yv, gv, _ = load_dataset(Path(args.video_frames) / "frames.csv", args.video_frames,
                                     args.size, media="frame")
        print(f"loaded {len(xv)} video frames", flush=True)
        x, y, groups = torch.cat([x, xv]), torch.cat([y, yv]), np.concatenate([groups, gv])
        partial = torch.cat([partial, torch.ones(len(xv), dtype=torch.bool)])

    if args.holdout_seed is not None:
        img_groups = groups[~partial.numpy()]
        uniq, counts = np.unique(img_groups, return_counts=True)
        big = set(uniq[np.argsort(-counts)[:4]])
        cand = sorted(set(uniq) - big)
        rng = np.random.default_rng(args.holdout_seed)
        held = set(rng.choice(cand, size=round(0.2 * len(uniq)), replace=False))
        va = np.array([g in held for g in groups]) & ~partial.numpy()
        tr = np.array([g not in held for g in groups])
        print(f"holdout: {va.sum()} images / {len(held)} casualties; "
              f"majority acc {(y[va] == 0).float().mean():.4f}", flush=True)
        model = fit(x[tr], y[tr], args, x[va], y[va], groups[tr], partial[tr])
        probs = predict(model, x[va])
        print(json.dumps(metrics(probs, y[va]), indent=2))
        if args.save_probs:
            torch.save({"probs": probs, "y": y[va]}, args.save_probs)
        return

    if args.cv_folds:
        all_probs = torch.zeros(len(x), len(SITES), len(CLASSES))
        for k, (tr, va) in enumerate(GroupKFold(args.cv_folds).split(x, y, groups)):
            print(f"--- fold {k + 1}/{args.cv_folds}", flush=True)
            model = fit(x[tr], y[tr], args, x[va], y[va], groups[tr], partial[tr])
            all_probs[va] = predict(model, x[va])
            if args.holdout_only:
                print(json.dumps(metrics(all_probs[va], y[va]), indent=2))
                return
        m = metrics(all_probs, y)
        print(json.dumps(m, indent=2))
        return

    model = fit(x, y, args, groups=groups, partial=partial)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": args.backbone, "size": args.size, "state_dict": model.state_dict(),
                "classes": CLASSES}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
