"""Main CLI entry point."""
import argparse
import yaml
import sys
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_train(args):
    from .train import train
    config = load_config(args.config)
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    train(config)


def cmd_evaluate(args):
    from .evaluate import evaluate
    config = load_config(args.config)
    evaluate(config, checkpoint=args.checkpoint)


def cmd_analyze(args):
    from .analyze import analyze
    config = load_config(args.config)
    analyze(config, checkpoint=args.checkpoint, n_samples=args.n_samples,
            cam_examples=not args.no_cam_examples)


def cmd_examples(args):
    from .data import VinDrCXRDataset
    from .model import create_model
    from .visualization import plot_success_failure, plot_cam_grid
    import pickle
    from flax import nnx

    config = load_config(args.config)
    model, _ = create_model(num_classes=config["data"]["num_classes"])
    with open(args.checkpoint, "rb") as f:
        nnx.update(model, pickle.load(f))
    model.eval()

    dataset = VinDrCXRDataset(
        root=config["data"]["root"], split="test",
        image_size=config["data"]["image_size"],
    )
    dataset.build_cache()

    plot_success_failure(
        model, dataset,
        success_class=args.success_class, failure_class=args.failure_class,
        n_each=args.n,
    )
    plot_cam_grid(dataset, model, n_examples=args.n)


def main():
    parser = argparse.ArgumentParser(description="XLoc-CXR: Explainability Localization on CXR")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate CAM metrics")
    p_eval.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p_eval.set_defaults(func=cmd_evaluate)

    # Analyze
    p_analyze = subparsers.add_parser("analyze", help="Run statistical analysis")
    p_analyze.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pkl")
    p_analyze.add_argument("--n-samples", type=int, default=100,
                           help="Samples per control condition (default 100)")
    p_analyze.add_argument("--no-cam-examples", action="store_true",
                           help="Skip CAM example figure generation")
    p_analyze.set_defaults(func=cmd_analyze)

    # Examples
    p_ex = subparsers.add_parser("examples", help="Generate CAM example figures")
    p_ex.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pkl")
    p_ex.add_argument("--success-class", default="Cardiomegaly")
    p_ex.add_argument("--failure-class", default="Nodule/Mass")
    p_ex.add_argument("--n", type=int, default=3, help="Examples per category")
    p_ex.set_defaults(func=cmd_examples)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
