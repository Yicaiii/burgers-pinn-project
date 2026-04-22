import torch
import numpy as np
from tqdm import tqdm

from model import PINN
from physics import burgers_residual
from utils import ensure_dir, plot_losses
import config


def sample_collocation(n):
    x = np.random.uniform(config.X_MIN, config.X_MAX, (n, 1))
    t = np.random.uniform(config.T_MIN, config.T_MAX, (n, 1))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)


def sample_boundary(n):
    t = np.random.uniform(config.T_MIN, config.T_MAX, (n, 1))
    x_left = np.full((n // 2, 1), config.X_MIN)
    x_right = np.full((n - n // 2, 1), config.X_MAX)

    x = np.vstack([x_left, x_right])
    t = np.vstack([t[: n // 2], t[n // 2 :]])
    u = np.zeros((n, 1))

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32),
    )


def sample_initial(n):
    x = np.random.uniform(config.X_MIN, config.X_MAX, (n, 1))
    t = np.zeros((n, 1))
    u = -np.sin(np.pi * x)

    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(t, dtype=torch.float32),
        torch.tensor(u, dtype=torch.float32),
    )


def main():
    device = torch.device(config.DEVICE)

    model = PINN(
        hidden_dim=config.HIDDEN_DIM,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    x_f, t_f = sample_collocation(config.N_F)
    x_b, t_b, u_b = sample_boundary(config.N_B)
    x_i, t_i, u_i = sample_initial(config.N_I)

    x_f, t_f = x_f.to(device), t_f.to(device)
    x_b, t_b, u_b = x_b.to(device), t_b.to(device), u_b.to(device)
    x_i, t_i, u_i = x_i.to(device), t_i.to(device), u_i.to(device)

    loss_history = []

    for epoch in tqdm(range(config.EPOCHS)):
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

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    ensure_dir("../outputs/figures")
    ensure_dir("../outputs/checkpoints")

    plot_losses(loss_history, "../outputs/figures/loss.png")
    torch.save(model.state_dict(), "../outputs/checkpoints/pinn_burgers.pt")

    print("Training finished.")
    print("Saved loss curve to ../outputs/figures/loss.png")
    print("Saved model to ../outputs/checkpoints/pinn_burgers.pt")


if __name__ == "__main__":
    main()