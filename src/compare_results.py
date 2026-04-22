import os
from pathlib import Path
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    results = [
        {"name": "Baseline Eval", "mse": 1.317788e-01, "rel_l2": 5.224842e-01},
        {"name": "Active V1 Before", "mse": 1.065557e-01, "rel_l2": 4.708752e-01},
        {"name": "Active V1 After", "mse": 1.213141e-01, "rel_l2": 5.024271e-01},
        {"name": "Active V2 Before", "mse": 1.009018e-01, "rel_l2": 4.582125e-01},
        {"name": "Active V2 After", "mse": 1.699038e-01, "rel_l2": 5.945918e-01},
    ]

    names = [r["name"] for r in results]
    mse_values = [r["mse"] for r in results]
    rel_l2_values = [r["rel_l2"] for r in results]

    ensure_dir(OUTPUT_DIR)

    mse_path = OUTPUT_DIR / "compare_mse.png"
    rel_l2_path = OUTPUT_DIR / "compare_rel_l2.png"

    plt.figure(figsize=(10, 5))
    plt.bar(names, mse_values)
    plt.ylabel("MSE")
    plt.title("Comparison of MSE Across Sampling Strategies")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(mse_path)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(names, rel_l2_values)
    plt.ylabel("Relative L2 Error")
    plt.title("Comparison of Relative L2 Error Across Sampling Strategies")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(rel_l2_path)
    plt.close()

    print("Comparison finished.")
    print(f"Saved MSE bar chart to {mse_path}")
    print(f"Saved Relative L2 bar chart to {rel_l2_path}")

    print("\nSummary:")
    for r in results:
        print(f'{r["name"]}: MSE={r["mse"]:.6e}, Relative L2={r["rel_l2"]:.6e}')


if __name__ == "__main__":
    main()