# Team Anote: D2 Sprint Qualification Task

This is Team Anote's submission for **Part 2 of the DARPA Documentation and Decision Support Sprint (D2) qualification**. It conforms to the *Qualification Task Interface Control Document*, Version 1.0 (August 2026).

The container classifies the injury state of each extremity of a simulated casualty in every input image:
- **Body sites:** left and right upper extremity, left and right lower extremity.
- **Classes:** `no_injury`, `wound`, `amputation` or `not_testable`.

Left and right are the casualty's anatomical sides.

## Submission

| Field | Value |
|---|---|
| Team Name | Anote |
| GitLab Username | anote1 |
| Project URL | https://gitlab.com/anote1/anote-d2-qualification-task |
| Container Image | `registry.gitlab.com/anote1/anote-d2-qualification-task/anote-d2-qualification-task` |
| Image Tag | `v1.3.0` |
| Image Digest | `sha256:ccd54114a9f0739a8fa8ea43c47348741eb18688a75c061ed7374886654442c2` |
| Technical Contact | vidranatan@gmail.com |

The same image, with an identical digest, is also published at `ghcr.io/anote-ai/anote-d2-qualification-task/anote-d2-qualification-task`.

## Running the container

The container takes no arguments, and the command is the one from ICD §3.3:

```bash
docker run --rm \
  --gpus all \
  --network none \
  -v /input:/app/input:ro \
  -v /output:/app/output:rw \
  registry.gitlab.com/anote1/anote-d2-qualification-task/anote-d2-qualification-task@sha256:ccd54114a9f0739a8fa8ea43c47348741eb18688a75c061ed7374886654442c2
```

- **Input:** every `.jpg`, `.jpeg` and `.png` file in `/app/input` (flat directory).
- **Output:** `/app/output/predictions.json`, in the ICD §3.2.1 schema, with exactly four site predictions per image.
- **Self-check:** after writing the file, the container validates it with the JHU/APL `validate_predictions.py`, which is included in this repository.

## Compliance with the ICD

| Requirement | Status |
|---|---|
| Offline execution (`--network none`) | All dependencies and model weights are inside the image. |
| GPU | 1 GPU. The PyTorch CUDA 12.8 runtime is bundled; no kernel driver is packaged. Runs on A40, L40 and H100 with driver 580 or later. Falls back to CPU if no GPU is visible. |
| GPU memory / RAM / disk | Well under 16 GB / 32 GB / 20 GB: three EfficientNet-B3 models at batch size 8. |
| Image size | About 4 GB compressed (limit 20 GB). |
| Runtime | A few minutes for the qualification set on GPU (limit 1 hour). |
| Output schema | Checked with `validate_predictions.py` on every build. |
| Malware scan | The exported container filesystem passes ClamAV (0 infected files) on every build. |

### Robustness

- **Image decoding:** images are decoded by content, so HEIC photos saved with a `.png` extension are handled. EXIF orientation is applied.
- **Hidden files** (for example `.DS_Store`) are ignored.
- **Fallbacks:** an image that can't be decoded, or a batch that fails, still receives four schema-valid site predictions.
- **Atomic output:** the output file is written atomically.

## Method

- **Model:** an ensemble of three EfficientNet-B3 networks (Noisy Student ImageNet initialization) fine-tuned at 320, 416 and 512 px input resolution. Each has one 4-way classification head per body site, and the ensemble averages their probabilities.
- **Anatomical laterality:** mirroring an image swaps the casualty's left and right. Training mirrors images and swaps the left/right labels together. At inference, each image is also scored mirrored, with its left/right predictions mapped back, and the two results are averaged.
- **Training data:** the qualification training images, plus frames sampled from the training videos. Video labels describe the casualty rather than the individual frame, so frames are trained with a partial-label objective: the target is the labeled class **or** `not_testable`.
- **Sampling:** casualty-balanced, so heavily photographed casualties don't dominate training.

### Validation

These results are on 12 casualties held out entirely from training (164 images, 656 site predictions):

| Model | Overall accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.639 | — |
| Single model (320 px) | 0.698 | 0.509 |
| **Submitted 3-model ensemble** | **0.742** | **0.568** |

The submitted models were then retrained on all training data.

## Repository contents

```
Dockerfile               container build (python:3.11-slim + PyTorch 2.8 / CUDA 12.8)
entrypoint.sh            runs inference, then the output validator
inference.py             /app/input -> /app/output/predictions.json
d2/common.py             body sites, classes, image loading, model definition
model/                   the three trained checkpoints (every *.pt is ensembled)
train.py                 training code
validate_predictions.py  JHU/APL output validator
.github/workflows/       build -> ICD smoke test -> validation -> ClamAV scan -> publish
```
