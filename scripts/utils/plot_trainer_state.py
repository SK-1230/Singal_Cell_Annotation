import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_trainer_state(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "log_history" not in data:
        raise ValueError("Invalid trainer_state.json: missing 'log_history'")

    return data


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sort_xy(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    pairs.sort(key=lambda t: t[0])
    if not pairs:
        return [], []
    xs_sorted = [p[0] for p in pairs]
    ys_sorted = [p[1] for p in pairs]
    return xs_sorted, ys_sorted


def collect_train_metric(log_history, metric_name):
    xs, ys = [], []
    for item in log_history:
        # 训练日志通常有 loss，但不带 eval_loss
        if metric_name in item and "eval_loss" not in item:
            step = item.get("step")
            value = item.get(metric_name)
            if step is not None and value is not None:
                xs.append(step)
                ys.append(value)
    return sort_xy(xs, ys)


def collect_eval_metric(log_history, metric_name):
    xs, ys = [], []
    for item in log_history:
        # 验证日志通常带 eval_xxx
        if metric_name in item:
            step = item.get("step")
            value = item.get(metric_name)
            if step is not None and value is not None:
                xs.append(step)
                ys.append(value)
    return sort_xy(xs, ys)


def save_plot(xs, ys, title, xlabel, ylabel, save_path: Path):
    if not xs or not ys:
        print(f"[Skip] No data for {save_path.name}")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {save_path}")


def collect_epoch_curve(log_history):
    xs, ys = [], []
    for item in log_history:
        if "epoch" in item and "step" in item and "eval_loss" not in item:
            xs.append(item["step"])
            ys.append(item["epoch"])
    return sort_xy(xs, ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer_state",
        type=str,
        required=True,
        help="Path to trainer_state.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save images. Default: sibling 'images' folder next to trainer_state.json",
    )
    args = parser.parse_args()

    trainer_state_path = Path(args.trainer_state).resolve()
    data = load_trainer_state(trainer_state_path)
    log_history = data["log_history"]

    if args.output_dir is None:
        output_dir = trainer_state_path.parent / "images"
    else:
        output_dir = Path(args.output_dir).resolve()

    ensure_dir(output_dir)

    # 训练指标
    train_loss_x, train_loss_y = collect_train_metric(log_history, "loss")
    train_lr_x, train_lr_y = collect_train_metric(log_history, "learning_rate")
    train_grad_x, train_grad_y = collect_train_metric(log_history, "grad_norm")
    train_acc_x, train_acc_y = collect_train_metric(log_history, "mean_token_accuracy")
    train_epoch_x, train_epoch_y = collect_epoch_curve(log_history)

    # 验证指标
    eval_loss_x, eval_loss_y = collect_eval_metric(log_history, "eval_loss")
    eval_acc_x, eval_acc_y = collect_eval_metric(log_history, "eval_mean_token_accuracy")
    eval_runtime_x, eval_runtime_y = collect_eval_metric(log_history, "eval_runtime")
    eval_sps_x, eval_sps_y = collect_eval_metric(log_history, "eval_samples_per_second")
    eval_stepsps_x, eval_stepsps_y = collect_eval_metric(log_history, "eval_steps_per_second")

    # 保存图片
    save_plot(
        train_loss_x, train_loss_y,
        "Train Loss", "Step", "Loss",
        output_dir / "train_loss.png"
    )

    save_plot(
        eval_loss_x, eval_loss_y,
        "Eval Loss", "Step", "Eval Loss",
        output_dir / "eval_loss.png"
    )

    save_plot(
        train_lr_x, train_lr_y,
        "Train Learning Rate", "Step", "Learning Rate",
        output_dir / "train_learning_rate.png"
    )

    save_plot(
        train_grad_x, train_grad_y,
        "Train Grad Norm", "Step", "Grad Norm",
        output_dir / "train_grad_norm.png"
    )

    save_plot(
        train_acc_x, train_acc_y,
        "Train Token Accuracy", "Step", "Mean Token Accuracy",
        output_dir / "train_token_acc.png"
    )

    save_plot(
        eval_acc_x, eval_acc_y,
        "Eval Token Accuracy", "Step", "Eval Mean Token Accuracy",
        output_dir / "eval_token_acc.png"
    )

    save_plot(
        train_epoch_x, train_epoch_y,
        "Train Epoch", "Step", "Epoch",
        output_dir / "train_epoch.png"
    )

    save_plot(
        eval_runtime_x, eval_runtime_y,
        "Eval Runtime", "Step", "Seconds",
        output_dir / "eval_runtime.png"
    )

    save_plot(
        eval_sps_x, eval_sps_y,
        "Eval Samples Per Second", "Step", "Samples/sec",
        output_dir / "eval_samples_per_second.png"
    )

    save_plot(
        eval_stepsps_x, eval_stepsps_y,
        "Eval Steps Per Second", "Step", "Steps/sec",
        output_dir / "eval_steps_per_second.png"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()