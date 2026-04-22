import numpy as np
import torch
import matplotlib.pyplot as plt

from model import PINN
import config


def build_grid(nx=256, nt=100):
    x = np.linspace(config.X_MIN, config.X_MAX, nx)
    t = np.linspace(config.T_MIN, config.T_MAX, nt)
    X, T = np.meshgrid(x, t)
    return x, t, X, T


def predict_on_grid(model, X, T, device):
    x_tensor = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
    t_tensor = torch.tensor(T.reshape(-1, 1), dtype=torch.float32).to(device)

    with torch.no_grad():
        u_pred = model(x_tensor, t_tensor).cpu().numpy()

    U = u_pred.reshape(X.shape)
    return U


def reference_solution(X, T, nu):
    return -np.sin(np.pi * X) * np.exp(-(np.pi ** 2) * nu * T)


def plot_heatmap(X, T, U, save_path, title):
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(X, T, U, shading="auto")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_time_slices(x, t, U_pred, U_ref, save_path, time_indices=(0, 25, 50, 75, 99)):
    plt.figure(figsize=(8, 5))
    for idx in time_indices:
        idx = min(idx, len(t) - 1)
        plt.plot(x, U_pred[idx, :], label=f"pred t={t[idx]:.2f}", linestyle="-")
        plt.plot(x, U_ref[idx, :], label=f"ref t={t[idx]:.2f}", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Prediction vs Reference at Different Time Slices")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_errors(U_pred, U_ref):
    abs_err = np.abs(U_pred - U_ref)
    mse = np.mean((U_pred - U_ref) ** 2)
    l2_rel = np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref)
    return abs_err, mse, l2_rel


def main():
    device = torch.device(config.DEVICE)

    model = PINN(
        hidden_dim=config.HIDDEN_DIM,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
    ).to(device)

    model.load_state_dict(
        torch.load("../outputs/checkpoints/pinn_burgers.pt", map_location=device)
    )
    model.eval()

    x, t, X, T = build_grid(nx=256, nt=100)
    U_pred = predict_on_grid(model, X, T, device)
    U_ref = reference_solution(X, T, config.NU)

    abs_err, mse, l2_rel = compute_errors(U_pred, U_ref)

    plot_heatmap(
        X, T, U_pred,
        "../outputs/figures/prediction_heatmap.png",
        "PINN Prediction Heatmap"
    )
    plot_heatmap(
        X, T, U_ref,
        "../outputs/figures/reference_heatmap.png",
        "Reference Solution Heatmap"
    )
    plot_heatmap(
        X, T, abs_err,
        "../outputs/figures/absolute_error_heatmap.png",
        "Absolute Error Heatmap"
    )
    plot_time_slices(
        x, t, U_pred, U_ref,
        "../outputs/figures/time_slices_comparison.png"
    )

    print("Evaluation finished.")
    print("Saved prediction heatmap to ../outputs/figures/prediction_heatmap.png")
    print("Saved reference heatmap to ../outputs/figures/reference_heatmap.png")
    print("Saved absolute error heatmap to ../outputs/figures/absolute_error_heatmap.png")
    print("Saved comparison plot to ../outputs/figures/time_slices_comparison.png")
    print(f"MSE: {mse:.6e}")
    print(f"Relative L2 Error: {l2_rel:.6e}")


if __name__ == "__main__":
    main()