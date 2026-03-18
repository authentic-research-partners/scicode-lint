import random

import torch


def run_monte_carlo(n_agents, n_steps):
    positions = torch.randn(n_agents, 2)
    velocities = torch.randn(n_agents, 2) * 0.1

    trajectory = []
    for _ in range(n_steps):
        random_force = torch.randn_like(positions) * 0.05
        for i in range(n_agents):
            if random.random() > 0.7:
                velocities[i] *= 0.9
        velocities += random_force
        positions += velocities
        trajectory.append(positions.clone())

    return torch.stack(trajectory)


if __name__ == "__main__":
    results = run_monte_carlo(100, 500)
    print(f"Simulation shape: {results.shape}")
