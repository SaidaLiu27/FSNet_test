import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt  # pip install matplotlib

def read_log(csv_path):
    epochs, total_loss, task_loss_f, reg_loss_rho, stab_loss_p, raw_phi = [], [], [], [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            total_loss.append(float(row["total_loss"]))
            task_loss_f.append(float(row["task_loss_f"]))
            reg_loss_rho.append(float(row["reg_loss_rho"]))
            stab_loss_p.append(float(row["stab_loss_p"]))
            raw_phi.append(float(row["raw_violation_phi"]))
    return epochs, total_loss, task_loss_f, reg_loss_rho, stab_loss_p, raw_phi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/training_log.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--logy", action="store_true")
    args = ap.parse_args()

    epochs, total_loss, task_loss_f, reg_loss_rho, stab_loss_p, raw_phi = read_log(args.csv)

    # Figure 1: Losses
    plt.figure(figsize=(8,5))
    plt.plot(epochs, total_loss, label="Total Loss")
    plt.plot(epochs, task_loss_f, label="Task f(â)")
    if any(v != 0.0 for v in reg_loss_rho):
        plt.plot(epochs, reg_loss_rho, label="Reg ρ||a-â||^2")
    if any(v != 0.0 for v in stab_loss_p):
        plt.plot(epochs, stab_loss_p, label="Stability P(a_raw)")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    if args.logy: plt.yscale("log")
    plt.title("FSNet Training Losses"); plt.grid(True, alpha=0.3); plt.legend()
    fig1_path = args.out or "results/fsnet_losses.png"
    Path(fig1_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(fig1_path, dpi=160)

    # Figure 2: Raw φ
    plt.figure(figsize=(8,4.5))
    plt.plot(epochs, raw_phi, label="Raw Violation φ")
    plt.xlabel("Epoch"); plt.ylabel("φ"); plt.yscale("log")
    plt.title("Raw Violation φ (log)"); plt.grid(True, alpha=0.3); plt.legend()
    fig2_path = Path(fig1_path).with_name("fsnet_phi.png")
    plt.tight_layout(); plt.savefig(fig2_path, dpi=160)

    if args.show: plt.show()
    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig2_path}")

if __name__ == "__main__":
    main()
