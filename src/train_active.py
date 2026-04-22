import torch
import numpy as np
from tqdm import tqdm

from model import PINN
from physics import burgers_residual
from utils import ensure_dir, plot_losses
import config


def reference_solution(x, t, nu):
    return -np.sin(np.pi * x) * np.exp(-(np.pi ** 2) * nu * t)


def sample_collocation_uniform(n):
    x = np.random.uniform(config.X_MIN, config.X_MAX, (n, 1))
    t = np.random.uniform(config.T_MIN, config.T_MAX, (n, 1))
    return x, t


def sample_boundary(n):
    t = np.random.uniform(config.T_MIN, config.T_MAX, (n, 1))
    x_left = np.full((n // 2, 1), config.X_MIN)
    x_right = np.full((n - n // 2, 1), config.X_MAX)

    x = np.vstack([x_left, x_right])
    t = np.vstack([t[: n // 2], t[n // 2 :]])
    u = np.zeros((n, 1))
    return x, t, u


def sample_initial(n):
    x = np.random.uniform(config.X_MIN, config.X_MAX, (n, 1))
    t = np.zeros((n, 1))
    u = -np.sin(np.pi * x)
    return x, t, u


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32).to(device)


def train_model(model, x_f, t_f, x_b, t_b, u_b, x_i, t_i, u_i, epochs, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    x_f = to_tensor(x_f, device)
    t_f = to_tensor(t_f, device)
    x_b = to_tensor(x_b, device)
    t_b = to_tensor(t_b, device)
    u_b = to_tensor(u_b, device)
    x_i = to_tensor(x_i, device)
    t_i = to_tensor(t_i, device)
    u_i = to_tensor(u_i, device)

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        f = burgers_residual(model, x_f, t_f, config.NU)
        loss_f = torch.mean(f ** 2)

        pred_b = model(x_b, t_b)
        loss_b = torch.mean((pred_b - u_b) ** 2)

        pred_i = model(x_i, t_i)
        loss_i = torch.mean((pred_i - u_i) ** 2)

        loss = loss_f + loss_b + loss_i
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return loss_history


def predict(model, x, t, device):
    x_tensor = to_tensor(x, device)
    t_tensor = to_tensor(t, device)

    with torch.no_grad():
        u = model(x_tensor, t_tensor).cpu().numpy()
    return u


def build_eval_grid(nx=120, nt=80):
    x = np.linspace(config.X_MIN, config.X_MAX, nx)
    t = np.linspace(config.T_MIN, config.T_MAX, nt)
    X, T = np.meshgrid(x, t)
    return X.reshape(-1, 1), T.reshape(-1, 1), X, T


def select_high_error_points(model, n_select, device):
    x_grid, t_grid, X, T = build_eval_grid()
    u_pred = predict(model, x_grid, t_grid, device)
    u_ref = reference_solution(x_grid, t_grid, config.NU)

    err = np.abs(u_pred - u_ref).reshape(-1)
    idx = np.argsort(err)[-n_select:]

    x_new = x_grid[idx]
    t_new = t_grid[idx]

    noise_x = 0.02 * np.random.randn(*x_new.shape)
    noise_t = 0.02 * np.random.randn(*t_new.shape)

    x_new = np.clip(x_new + noise_x, config.X_MIN, config.X_MAX)
    t_new = np.clip(t_new + noise_t, config.T_MIN, config.T_MAX)

    mse = np.mean((u_pred - u_ref) ** 2)
    rel_l2 = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)

    return x_new, t_new, mse, rel_l2


def main():
    device = torch.device(config.DEVICE)

    ensure_dir("../outputs/figures")
    ensure_dir("../outputs/checkpoints")

    n_f_initial = 1000
    n_f_add = 500
    n_b = 200
    n_i = 200
    epochs_stage1 = 2000
    epochs_stage2 = 1500
    lr = config.LR

    x_f, t_f = sample_collocation_uniform(n_f_initial)
    x_b, t_b, u_b = sample_boundary(n_b)
    x_i, t_i, u_i = sample_initial(n_i)

    model = PINN(
        hidden_dim=config.HIDDEN_DIM,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
    ).to(device)

    print("Stage 1: training baseline active-learning model...")
    loss_stage1 = train_model(
        model, x_f, t_f, x_b, t_b, u_b, x_i, t_i, u_i,
        epochs=epochs_stage1, lr=lr, device=device
    )

    x_add, t_add, mse_before, rel_l2_before = select_high_error_points(
        model, n_select=n_f_add, device=device
    )

    print(f"Before enrichment - MSE: {mse_before:.6e}")
    print(f"Before enrichment - Relative L2 Error: {rel_l2_before:.6e}")

    x_f_enriched = np.vstack([x_f, x_add])
    t_f_enriched = np.vstack([t_f, t_add])

    print("Stage 2: retraining after adding high-error points...")
    loss_stage2 = train_model(
        model, x_f_enriched, t_f_enriched, x_b, t_b, u_b, x_i, t_i, u_i,
        epochs=epochs_stage2, lr=lr, device=device
    )

    _, _, mse_after, rel_l2_after = select_high_error_points(
        model, n_select=n_f_add, device=device
    )

    print(f"After enrichment - MSE: {mse_after:.6e}")
    print(f"After enrichment - Relative L2 Error: {rel_l2_after:.6e}")

    full_loss = loss_stage1 + loss_stage2
    plot_losses(full_loss, "../outputs/figures/active_learning_loss.png")

    torch.save(model.state_dict(), "../outputs/checkpoints/pinn_burgers_active.pt")

    print("Saved loss curve to ../outputs/figures/active_learning_loss.png")
    print("Saved model to ../outputs/checkpoints/pinn_burgers_active.pt")


if __name__ == "__main__":
    main()