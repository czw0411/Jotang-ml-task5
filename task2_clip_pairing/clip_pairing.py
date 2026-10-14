"""Task 2: pair 4 images with 4 captions using OpenAI's public CLIP (RN50) weights.

Experiments
  1. Encode images and texts, print feature shapes, L2-normalize both.
  2. 4x4 cosine-similarity matrix + heat map; check top-1 pairing accuracy.
  3. Add one deliberately wrong ("hard negative") caption -> 5x5 matrix.
  4. Image x image similarity, and a caption-perturbation probe.
  5. Prompt engineering probe: bare class name vs "a photo of a ..." vs ensembling.

Inference only; no training.
"""

import json
import pathlib

import clip
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
IMG_DIR = ROOT / "images"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_LINES = []


def log(*parts):
    text = " ".join(str(p) for p in parts)
    print(text)
    LOG_LINES.append(text)


IMAGE_FILES = ["01_dog.jpg", "02_cat.jpg", "03_apple.jpg", "04_airplane.jpg"]
SHORT_NAMES = ["dog", "cat", "apple", "airplane"]

# One short English description per image (written by looking at each photo).
CAPTIONS = [
    "a golden retriever dog running on the grass",
    "a gray and white cat playing with a lizard",
    "a red apple on a white background",
    "a triplane flying in the blue sky",
]

# Deliberately similar to image 3 (same layout/colour, wrong object).
HARD_NEGATIVE = "a red tomato on a white background"

CAPTIONS_WITH_NEGATIVE = CAPTIONS + [HARD_NEGATIVE]

# Caption perturbations for the dog photo: correct + colour / count / action / species errors.
PERTURBATIONS = [
    ("correct (reference)", "a golden retriever dog running on the grass"),
    ("colour wrong", "a black dog running on the grass"),
    ("count wrong", "two dogs running on the grass"),
    ("action wrong", "a golden retriever sleeping on a bed"),
    ("species wrong", "a white cat running on the grass"),
]

# Zero-shot prompt engineering probe.
BARE_LABELS = ["dog", "cat", "apple", "airplane"]
PHOTO_LABELS = [f"a photo of a {c}" for c in BARE_LABELS]
TEMPLATES = [
    "a photo of a {}",
    "a photo of a small {}",
    "a photo of a large {}",
    "a close-up photo of a {}",
    "a bright photo of a {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a blurry photo of a {}",
]
def l2_normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


def sim_matrix(a_norm, b_norm):
    """Cosine similarity: both inputs are already L2-normalized."""
    return (a_norm @ b_norm.T).numpy()


def heatmap(matrix, xlabels, ylabels, title, path, correct_diag=True, note=None):
    fig, ax = plt.subplots(figsize=(1.5 + 1.55 * len(xlabels), 1.3 + 1.0 * len(ylabels)))
    im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-0.05, vmax=0.45)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="black")
    if correct_diag:
        for k in range(min(matrix.shape)):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                       edgecolor="lime", lw=2.5))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("text features")
    ax.set_ylabel("image features")
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="cosine similarity")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    log(f"[saved] {path}")


def top1_report(matrix, row_names, col_names):
    lines = []
    hits = 0
    for i, row in enumerate(row_names):
        j = int(np.argmax(matrix[i]))
        ok = j == i
        hits += ok
        order = np.argsort(-matrix[i])
        runner_up = order[1]
        lines.append(
            f"  image {i + 1} ({row}): pred='{col_names[j]}' "
            f"sim={matrix[i, j]:.4f} {'OK' if ok else 'MISS'} | "
            f"runner-up '{col_names[runner_up]}' {matrix[i, runner_up]:.4f} "
            f"(margin {matrix[i, j] - matrix[i, runner_up]:+.4f})"
        )
    return hits, lines


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    log(f"device               : {device}")
    log(f"model                : RN50 (OpenAI CLIP, 1024-d joint embedding space)")
    log(f"logit_scale value    : {model.logit_scale.exp().item():.4f} "
        f"(learned temperature tau = {1 / model.logit_scale.exp().item():.4f})")

    images = torch.stack([preprocess(Image.open(IMG_DIR / f).convert("RGB"))
                          for f in IMAGE_FILES]).to(device)
    log(f"\ninput image tensor   : {tuple(images.shape)}  (N, 3, 224, 224)")

    # contact sheet of the four inputs, next to their captions
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.1))
    for ax, fname, caption in zip(axes, IMAGE_FILES, CAPTIONS):
        ax.imshow(Image.open(IMG_DIR / fname).convert("RGB"))
        ax.set_title(f"{fname}\n{caption}", fontsize=8.5)
        ax.axis("off")
    fig.suptitle("Input images and the captions used for pairing", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "images_overview.png", dpi=160)
    plt.close(fig)
    log(f"[saved] {OUT_DIR / 'images_overview.png'}")

    with torch.no_grad():
        image_features_raw = model.encode_image(images)
        text_tokens = clip.tokenize(CAPTIONS, truncate=True).to(device)
        text_features_raw = model.encode_text(text_tokens)

    log(f"text  token shape    : {tuple(text_tokens.shape)}  (N, 77 context length)")
    log(f"image features (raw) : {tuple(image_features_raw.shape)}")
    log(f"text features  (raw) : {tuple(text_features_raw.shape)}")

    image_features = l2_normalize(image_features_raw)
    text_features = l2_normalize(text_features_raw)

    log("\nL2 norms before / after normalization (image):")
    for name, raw, norm in zip(SHORT_NAMES, image_features_raw, image_features):
        log(f"  {name:<9} |f|={raw.norm().item():7.3f} -> |f|= {norm.norm().item():.4f}")
    log("L2 norms before / after normalization (text):")
    for name, raw, norm in zip(SHORT_NAMES, text_features_raw, text_features):
        log(f"  {name:<9} |f|={raw.norm().item():7.3f} -> |f|= {norm.norm().item():.4f}")
    log(f"\nnormalized image features: {tuple(image_features.shape)}")
    log(f"normalized text  features: {tuple(text_features.shape)}")

    # ---------------- experiment 1/2: 4x4 matrix ----------------
    sim = sim_matrix(image_features, text_features)
    log("\n=== Experiment 1: 4x4 cosine similarity matrix ===")
    header = "image \\ text".ljust(14) + "".join(f"{n:>12}" for n in SHORT_NAMES)
    log(header)
    for i, name in enumerate(SHORT_NAMES):
        log(f"{name:<14}" + "".join(f"{v:>12.4f}" for v in sim[i]))
    hits, lines = top1_report(sim, SHORT_NAMES, SHORT_NAMES)
    log("\ntop-1 pairing per image:")
    for line in lines:
        log(line)
    log(f"top-1 accuracy: {hits}/4")

    np.savetxt(OUT_DIR / "similarity_4x4.csv", sim, delimiter=",", fmt="%.6f",
               header=",".join(SHORT_NAMES), comments="")
    heatmap(sim, SHORT_NAMES, SHORT_NAMES,
            "CLIP RN50: image (rows) vs caption (columns)\ngreen box = correct pairing",
            OUT_DIR / "similarity_4x4_heatmap.png")

    # also show the CLIP scaled logits (cosine sim * learned logit scale)
    with torch.no_grad():
        prob = (model.logit_scale.exp() * image_features @ text_features.T).softmax(dim=-1).numpy()
    log("\nimage->text probabilities (softmax over the learned logit scale):")
    for i, name in enumerate(SHORT_NAMES):
        log(f"  {name:<9} " + " ".join(f"{c}={p:.5f}" for c, p in zip(SHORT_NAMES, prob[i])))

    # ---------------- experiment 3: hard negative ----------------
    with torch.no_grad():
        neg_raw = model.encode_text(clip.tokenize([HARD_NEGATIVE], truncate=True).to(device))
    neg_norm = l2_normalize(neg_raw)
    text_all = torch.cat([text_features, neg_norm], dim=0)
    sim5 = sim_matrix(image_features, text_all)
    col_names = SHORT_NAMES + ["..tomato"]

    log("\n=== Experiment 2: adding the deliberately wrong caption ===")
    log(f'hard negative: "{HARD_NEGATIVE}"  (image 3 is an apple on a white background)')
    log("image \\ text".ljust(14) + "".join(f"{n:>12}" for n in col_names))
    for i, name in enumerate(SHORT_NAMES):
        log(f"{name:<14}" + "".join(f"{v:>12.4f}" for v in sim5[i]))
    log(f"  apple vs 'a red apple ... '  : {sim[2, 2]:.4f}")
    log(f"  apple vs 'a red tomato ...'  : {sim5[2, 4]:.4f}  "
        f"(delta {sim5[2, 4] - sim[2, 2]:+.4f})")
    log(f"  all-other-image vs tomato    : "
        f"{np.mean(np.delete(sim5[:, 4], 2)):.4f} (mean of the 3 unrelated images)")
    hits5, lines5 = top1_report(sim5[:, :4], SHORT_NAMES, SHORT_NAMES)
    log(f"top-1 accuracy with the 5th caption present: {hits5}/4")
    for line in lines5:
        log(line)

    np.savetxt(OUT_DIR / "similarity_5x5.csv", sim5, delimiter=",", fmt="%.6f",
               header=",".join(col_names), comments="")
    heatmap(sim5, col_names, SHORT_NAMES,
            "CLIP RN50 with one hard-negative caption (fifth column)",
            OUT_DIR / "similarity_5x5_heatmap.png", correct_diag=False,
            note="green box would mark the correct pairs; column 5 is deliberately wrong")

    # ---------------- experiment 4: image x image ----------------
    sim_ii = sim_matrix(image_features, image_features)
    log("\n=== Experiment 3: image x image cosine similarity ===")
    log("image \\ image".ljust(14) + "".join(f"{n:>12}" for n in SHORT_NAMES))
    for i, name in enumerate(SHORT_NAMES):
        log(f"{name:<14}" + "".join(f"{v:>12.4f}" for v in sim_ii[i]))
    np.savetxt(OUT_DIR / "similarity_image_image.csv", sim_ii, delimiter=",", fmt="%.6f",
               header=",".join(SHORT_NAMES), comments="")

    # caption perturbation probe on the dog photo
    pert_caps = [c for _, c in PERTURBATIONS]
    with torch.no_grad():
        pert_feat = l2_normalize(model.encode_text(clip.tokenize(pert_caps, truncate=True).to(device)))
    dog_sims = (image_features[0] @ pert_feat.T).numpy()

    log("\n=== Experiment 4: caption perturbations for the dog photo ===")
    for (kind, cap), s in zip(PERTURBATIONS, dog_sims):
        log(f"  {kind:<20} sim={s:.4f}  <- \"{cap}\"")
    log(f"  drop caused by the colour error : {dog_sims[1] - dog_sims[0]:+.4f}")
    log(f"  drop caused by the count error  : {dog_sims[2] - dog_sims[0]:+.4f}")
    log(f"  drop caused by the action error : {dog_sims[3] - dog_sims[0]:+.4f}")
    log(f"  drop caused by the species error: {dog_sims[4] - dog_sims[0]:+.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    colours = ["#2b7a3d"] + ["#b8452f"] * (len(PERTURBATIONS) - 1)
    ax.barh([k for k, _ in PERTURBATIONS][::-1], dog_sims[::-1], color=colours[::-1])
    for i, v in enumerate(dog_sims[::-1]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, max(dog_sims) * 1.25)
    ax.set_xlabel("cosine similarity with the dog image")
    ax.set_title("Same photo, one wrong detail each time (CLIP RN50)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "caption_perturbation_dog.png", dpi=170)
    plt.close(fig)
    log(f"[saved] {OUT_DIR / 'caption_perturbation_dog.png'}")

    # ---------------- experiment 5: prompt engineering ----------------
    def classify(caption_list):
        with torch.no_grad():
            feat = l2_normalize(model.encode_text(clip.tokenize(caption_list, truncate=True).to(device)))
        scores = (image_features @ feat.T).numpy()
        return scores, [SHORT_NAMES[int(np.argmax(row))] for row in scores]

    log("\n=== Experiment 5: prompt engineering (zero-shot classification) ===")
    prompt_runs = {}
    for name, caps in [("bare label", BARE_LABELS),
                       ("'a photo of a ...'", PHOTO_LABELS),
                       ("'a photo of a small ...'", [f"a photo of a small {c}" for c in BARE_LABELS]),
                       ("'a photo of a large ...'", [f"a photo of a large {c}" for c in BARE_LABELS])]:
        scores, preds = classify(caps)
        prompt_runs[name] = {"captions": caps, "preds": preds}
        log(f"  {name:<24} preds={preds} "
            f"acc={sum(p == n for p, n in zip(preds, SHORT_NAMES))}/4")

    # ensembling over the template set
    with torch.no_grad():
        feat_stack = []
        for t in TEMPLATES:
            feat_stack.append(l2_normalize(model.encode_text(
                clip.tokenize([t.format(c) for c in SHORT_NAMES], truncate=True).to(device))))
        ens = torch.stack(feat_stack).mean(dim=0)
        ens = l2_normalize(ens)
    ens_scores = (image_features @ ens.T).numpy()
    ens_preds = [SHORT_NAMES[int(np.argmax(r))] for r in ens_scores]
    prompt_runs["ensemble of 8 templates"] = {"templates": TEMPLATES, "preds": ens_preds}
    log(f"  {'ensemble of 8 templates':<24} preds={ens_preds} "
        f"acc={sum(p == n for p, n in zip(ens_preds, SHORT_NAMES))}/4")

    # dog-specific probe: bare "dog" vs template, as in the reading notes question
    with torch.no_grad():
        probe = model.encode_text(clip.tokenize(
            ["dog", "a photo of a dog", "a photo of a black dog"], truncate=True).to(device))
        probe = l2_normalize(probe)
    probe_sims = (image_features[0] @ probe.T).numpy()
    log("\n  dog image similarity to different prompts:")
    for cap, s in zip(["dog", "a photo of a dog", "a photo of a black dog"], probe_sims):
        log(f"    {s:.4f}  <- \"{cap}\"")

    # ---------------- experiment 6: temperature (logit scale) sweep ----------------
    log("\n=== Experiment 6: what the learned temperature does to the softmax ===")
    log("CLIP computes softmax(logit_scale * cosine_similarity); logit_scale = 1/tau.")
    sims_t = torch.from_numpy(sim).double()
    temp_table = {}
    for scale in [1.0, 5.0, 20.0, 100.0]:
        tau = 1.0 / scale
        probs = (scale * sims_t).softmax(dim=-1).numpy()
        row_entropy = [float(-(p * np.log(p + 1e-12)).sum()) for p in probs]
        temp_table[f"scale={scale:g} (tau={tau:g})"] = {
            "probabilities": probs.tolist(),
            "mean_entropy_nats": float(np.mean(row_entropy)),
            "mean_max_prob": float(np.mean(probs.max(axis=1))),
        }
        log(f"  scale={scale:<6g} (tau={tau:<7g}) mean entropy={np.mean(row_entropy):.3f} nats, "
            f"mean max prob={np.mean(probs.max(axis=1)):.4f}, apple row={np.round(probs[2], 4).tolist()}")
    log("  (max possible entropy for 4 classes = ln 4 = "
        f"{np.log(4):.4f} nats)")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    widths = 0.2
    for k, (label, payload) in enumerate(temp_table.items()):
        vals = np.array(payload["probabilities"])[2]  # apple row
        ax.bar(np.arange(4) + (k - 1.5) * widths, vals, width=widths, label=label)
    ax.set_xticks(range(4), SHORT_NAMES)
    ax.set_ylabel("softmax probability")
    ax.set_title("Apple image: temperature controls how peaked the distribution is", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "temperature_sweep_apple_row.png", dpi=170)
    plt.close(fig)
    log(f"[saved] {OUT_DIR / 'temperature_sweep_apple_row.png'}")

    results = {
        "model": "RN50 (OpenAI CLIP)",
        "logit_scale": model.logit_scale.exp().item(),
        "images": IMAGE_FILES,
        "captions": CAPTIONS,
        "hard_negative": HARD_NEGATIVE,
        "similarity_4x4": sim.tolist(),
        "similarity_5x5": sim5.tolist(),
        "similarity_image_image": sim_ii.tolist(),
        "top1_accuracy": f"{hits}/4",
        "perturbations": [{"kind": k, "caption": c, "similarity": float(s)}
                          for (k, c), s in zip(PERTURBATIONS, dog_sims)],
        "prompt_runs": prompt_runs,
        "ensembling_templates": TEMPLATES,
        "dog_prompt_probe": dict(zip(["dog", "a photo of a dog", "a photo of a black dog"],
                                     map(float, probe_sims))),
        "temperature_sweep": temp_table,
    }
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"[saved] {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
    (OUT_DIR / "experiment_log.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
