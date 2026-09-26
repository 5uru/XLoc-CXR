"""Visualization utilities for training and results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_accuracies: list,
    output_dir: str = "outputs/figures",
):
    """Plot training curves."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accuracies, label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()
    print(f"[plots] Saved training curves to {output_dir}/training_curves.png")


def plot_results_summary(
    df: pd.DataFrame,
    output_dir: str = "outputs/figures",
):
    """Plot summary of evaluation results (layer × method, per metric)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    summary = df.groupby(["layer", "method", "metric"])["score"].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ["pointing_game", "energy_box", "iou", "distance"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        data = summary[summary["metric"] == metric]

        methods = data["method"].unique()
        layers = data["layer"].unique()
        x = np.arange(len(methods))
        width = 0.35

        for i, layer in enumerate(layers):
            layer_data = data[data["layer"] == layer].set_index("method").reindex(methods)
            ax.bar(x + i * width, layer_data["score"], width, label=layer)

        ax.set_xlabel("CAM Method")
        ax.set_ylabel("Score")
        if metric == "iou":
            ax.set_title("IoU")
        else:
            ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(methods)
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/results_summary.png", dpi=150)
    plt.close()
    print(f"[plots] Saved results summary to {output_dir}/results_summary.png")


def plot_per_class_metrics(
    df: pd.DataFrame,
    output_dir: str = "outputs/figures",
    metric: str = "pointing_game",
):
    """Bar plot of a metric per pathology, grouped by layer × method."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = (
        df[df["metric"] == metric]
        .groupby(["class", "layer", "method"])["score"]
        .mean()
        .reset_index()
    )
    classes = (
        df[df["metric"] == metric]
        .groupby("class")["score"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )
    combos = data.groupby(["layer", "method"]).groups.keys()

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(classes))
    width = 0.8 / max(len(combos), 1)

    for i, (layer, method) in enumerate(sorted(combos)):
        subset = data[(data["layer"] == layer) & (data["method"] == method)]
        scores = subset.set_index("class")["score"].reindex(classes).fillna(0)
        ax.bar(x + i * width, scores, width, label=f"{layer} + {method}")

    ax.set_xlabel("Pathology")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_', ' ').title()} per Pathology")
    ax.set_xticks(x + width * len(combos) / 2)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = f"{output_dir}/per_class_{metric}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plots] Saved per-class metric to {out}")


def plot_controls_per_class(
    controls: dict,
    output_dir: str = "outputs/figures",
):
    """Plot trained vs null conditions per pathology."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    per_class = controls["per_class"]
    classes = sorted(per_class.keys(),
                     key=lambda c: -per_class[c]["trained"]["mean"])

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(classes))
    width = 0.2

    conditions = ["trained", "untrained", "randomized_weights", "random_cam"]
    colors = {"trained": "#2ecc71", "untrained": "#e74c3c",
              "randomized_weights": "#f39c12", "random_cam": "#95a5a6"}

    for i, cond in enumerate(conditions):
        means = []
        for c in classes:
            entry = per_class[c].get(cond, {})
            if cond == "trained":
                means.append(entry.get("mean", 0))
            else:
                means.append(entry.get("null_mean", 0))
        ax.bar(x + i * width, means, width, label=cond, color=colors[cond])

    ax.set_xlabel("Pathology")
    ax.set_ylabel("Localization score (PG+EB)/2")
    ax.set_title("Trained model vs controls, per pathology")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = f"{output_dir}/controls_per_class.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plots] Saved controls per class to {out}")
