"""Analysis script: per-pathology stats, AUC, controls, CAM examples."""
import json
import pickle

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from .data import VinDrCXRDataset
from .model import create_model
from .metrics import per_class_auc
from .analysis.randomization import run_controls
from .visualization import (
    plot_results_summary, plot_per_class_metrics,
    plot_controls_per_class, plot_cam_grid,
)


def _compute_test_auc(model, dataset, batch_size: int = 64) -> dict:
    """Per-class ROC AUC of the trained classifier on the test set."""
    model.eval()
    all_probs, all_labels = [], []

    n = len(dataset)
    for start in tqdm(range(0, n, batch_size), desc="AUC inference"):
        end = min(start + batch_size, n)
        images, labels_list = [], []
        for i in range(start, end):
            img, labels_vec, _ = dataset[i]
            images.append(img)
            labels_list.append(labels_vec)
        images = jnp.array(np.stack(images))
        labels = np.stack(labels_list)

        logits, _, _ = model(images)
        all_probs.append(np.asarray(jax.nn.sigmoid(logits)))
        all_labels.append(labels)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return per_class_auc(probs, labels, dataset.classes)


def analyze(
    config: dict,
    checkpoint: str = "outputs/checkpoints/best_model.pkl",
    n_samples: int = 100,
    cam_examples: bool = True,
):
    """Run full per-pathology statistical analysis."""
    print("=" * 60)
    print("  XLoc-CXR Analysis (per pathology)")
    print("=" * 60)

    # --- Load CAM metric results ---
    print("\n[analyze] Loading results...")
    df = pd.read_csv("outputs/results.csv")

    # ---- Global summary ----
    print("\n[analyze] Summary by layer × method × metric:")
    summary = df.groupby(["layer", "method", "metric"])["score"].agg(["mean", "std"])
    print(summary.to_string())

    # ---- PER PATHOLOGY: all metrics × layer × method ----
    print("\n" + "=" * 60)
    print("  PER-PATHOLOGY RESULTS")
    print("=" * 60)
    per_class_table = (
        df[df["class"] != "No finding"]
        .groupby(["class", "layer", "method", "metric"])["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    for cls in per_class_table["class"].unique():
        sub = per_class_table[per_class_table["class"] == cls]
        n = int(sub["count"].max())
        print(f"\n--- {cls} (n={n}) ---")
        pivot = sub.pivot_table(
            index=["layer", "method"], columns="metric",
            values="mean",
        )
        print(pivot.round(4).to_string())

    # No finding diffusivity separately
    nf = df[df["class"] == "No finding"]
    if not nf.empty:
        print(f"\n--- No finding (n={len(nf)//2}) — diffusivity ---")
        print(nf.groupby(["layer", "method", "metric"])["score"].mean().round(4).to_string())

    # ---- Plots ----
    print("\n[analyze] Generating summary plots...")
    plot_results_summary(df)
    for metric in ["pointing_game", "energy_box", "iou", "distance"]:
        plot_per_class_metrics(df, metric=metric)

    # --- Load trained model ---
    print("\n[analyze] Loading trained model...")
    model, _ = create_model(num_classes=config["data"]["num_classes"])
    with open(checkpoint, "rb") as f:
        nnx.update(model, pickle.load(f))

    dataset = VinDrCXRDataset(
        root=config["data"]["root"],
        split="test",
        image_size=config["data"]["image_size"],
    )
    dataset.build_cache()

    # ---- Per-class AUC ----
    print("\n[analyze] Computing per-class AUC on test set...")
    aucs = _compute_test_auc(model, dataset, batch_size=config["eval"]["batch_size"])
    print(f"\n  MACRO AUC: {aucs['MACRO']:.3f}")
    for name in dataset.classes:
        v = aucs.get(name, float("nan"))
        if not np.isnan(v):
            print(f"  {name:20s} AUC={v:.3f}")

    # ---- Controls (global + per pathology) ----
    print(f"\n[analyze] Running control suite (n_samples={n_samples})...")

    def fresh_model():
        m, _ = create_model(num_classes=config["data"]["num_classes"], seed=123)
        return m

    controls = run_controls(
        trained_model=model,
        fresh_model_fn=fresh_model,
        dataset=dataset,
        cam_method="grad_cam",
        layer="layer4",
        n_samples=n_samples,
    )

    # Global controls
    print("\n[analyze] Global control results:")
    g = controls["global"]
    t = g["trained"]
    print(f"  Trained: {t['mean']:.4f} ± {t['std']:.4f} (n={t['n']})")
    for name in ("untrained", "randomized_weights", "random_cam"):
        r = g[name]
        sig = "✓ SIGNIFICANT" if r["significant"] else "✗ not significant"
        print(f"  vs {name:20s} null={r['null_mean']:.4f}±{r['null_std']:.4f}  p={r['p_value']:.2e}  {sig}")

    # Per-pathology controls
    print("\n[analyze] Per-pathology control results (trained vs randomized weights):")
    print(f"  {'Class':22s} {'trained':>8s} {'rand':>8s} {'p-value':>10s}  sig")
    for cls, entry in sorted(
        controls["per_class"].items(),
        key=lambda kv: -kv[1]["trained"]["mean"],
    ):
        tr = entry["trained"]
        rw = entry["randomized_weights"]
        sig = "✓" if rw["significant"] else "✗"
        p_str = f"{rw['p_value']:.2e}" if not np.isnan(rw["p_value"]) else "  nan"
        print(f"  {cls:22s} {tr['mean']:8.4f} {rw['null_mean']:8.4f} {p_str:>10s}  {sig} (n={tr['n']})")

    plot_controls_per_class(controls)

    # ---- CAM example figures ----
    if cam_examples:
        print("\n[analyze] Generating CAM example figures...")
        plot_cam_grid(dataset, model, n_examples=6)

    # ---- Save everything ----
    with open("outputs/controls.json", "w") as f:
        json.dump(controls, f, indent=2)
    with open("outputs/auc.json", "w") as f:
        json.dump(aucs, f, indent=2)
    per_class_table.to_csv("outputs/results_per_class.csv", index=False)
    print("\n[analyze] Saved outputs/controls.json, auc.json, results_per_class.csv")
    print("[analyze] Done!")
