"""Training script with logging and progress tracking (Flax NNX)."""
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm

from .data import VinDrCXRDataset
from .model import create_model, binary_cross_entropy_loss, compute_accuracy
from .visualization import plot_training_curves


@nnx.jit
def train_step(model, optimizer, images, labels):
    """Single training step."""
    def loss_fn(model):
        logits, _, _ = model(images)
        return binary_cross_entropy_loss(logits, labels)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    logits, _, _ = model(images)
    acc = compute_accuracy(logits, labels)
    return loss, acc


@nnx.jit
def eval_step(model, images, labels):
    """Single evaluation step."""
    logits, _, _ = model(images)
    loss = binary_cross_entropy_loss(logits, labels)
    acc = compute_accuracy(logits, labels)
    return loss, acc


def make_batches(dataset, batch_size, shuffle=True, seed=0):
    """Generate batches from dataset."""
    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        np.random.RandomState(seed).shuffle(indices)

    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        images, labels_list = [], []
        for i in batch_indices:
            img, labels_vec, _ = dataset[i]
            images.append(img)
            labels_list.append(labels_vec)
        yield jnp.array(np.stack(images)), jnp.array(np.stack(labels_list))


def train(config: dict):
    """Main training function."""
    print("=" * 60)
    print("  XLoc-CXR Training (Flax NNX)")
    print("=" * 60)

    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    print("\n[train] Loading dataset...")
    train_ds = VinDrCXRDataset(
        root=config["data"]["root"],
        split="train",
        image_size=config["data"]["image_size"],
    )
    val_ds = VinDrCXRDataset(
        root=config["data"]["root"],
        split="val",
        image_size=config["data"]["image_size"],
    )
    test_ds = VinDrCXRDataset(
        root=config["data"]["root"],
        split="test",
        image_size=config["data"]["image_size"],
    )
    print(f"[train] Train samples: {len(train_ds)}")
    print(f"[train] Val samples:   {len(val_ds)}")
    print(f"[train] Test samples:  {len(test_ds)} (held out — not used during training)")

    # Build preprocessing cache once (skips if already done)
    print("\n[train] Building preprocessing cache (first run only)...")
    train_ds.build_cache()
    val_ds.build_cache()
    test_ds.build_cache()

    # --- Model ---
    print("\n[train] Creating model...")
    model, optimizer = create_model(
        num_classes=config["data"]["num_classes"],
        seed=config["train"]["seed"],
    )
    n_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"[train] Parameters: {n_params:,}")

    # --- Config ---
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]
    log_interval = config["train"]["log_interval"]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    print(f"\n[train] Training for {epochs} epochs, batch_size={batch_size}")
    print("-" * 60)

    for epoch in range(epochs):
        t0 = time.time()

        # === Training ===
        model.train()
        train_loss, train_acc, n_train = 0.0, 0.0, 0

        pbar = tqdm(
            make_batches(train_ds, batch_size, shuffle=True, seed=epoch),
            desc=f"Epoch {epoch+1}/{epochs} [train]",
            ncols=100,
        )
        for images, labels in pbar:
            loss, acc = train_step(model, optimizer, images, labels)
            train_loss += float(loss)
            train_acc += acc
            n_train += 1
            if n_train % log_interval == 0:
                pbar.set_postfix(loss=f"{train_loss/n_train:.4f}", acc=f"{train_acc/n_train:.4f}")

        train_loss /= max(n_train, 1)
        train_acc /= max(n_train, 1)

        # === Validation ===
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0

        for images, labels in tqdm(
            make_batches(val_ds, batch_size, shuffle=False),
            desc=f"Epoch {epoch+1}/{epochs} [val]  ",
            ncols=100,
        ):
            loss, acc = eval_step(model, images, labels)
            val_loss += float(loss)
            val_acc += acc
            n_val += 1

        val_loss /= max(n_val, 1)
        val_acc /= max(n_val, 1)
        elapsed = time.time() - t0

        # === Log epoch ===
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
        print(f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}")
        print(f"  Time   {elapsed:.1f}s")

        # === Save best ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / "best_model.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(nnx.to_pure_dict(nnx.state(model)), f)
            print(f"  [checkpoint] Saved best model (val_loss={val_loss:.4f})")

        print("-" * 60)

    # === Plots ===
    plot_training_curves(
        history["train_loss"], history["val_loss"],
        history["val_acc"],
        output_dir="outputs/figures",
    )

    print("\n[train] Done!")
    return model, optimizer, history
