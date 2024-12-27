# /// script
# title = "Fit oscillatory GRN"
# description = "Optimize a GRN to produce antiphase oscillations among 4 species."
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
# keywords = ["GRNs", "oscillations", "pytorch"]
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Developers",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.24.0",
#     "matplotlib",
#     "torch"
# ]
# ///

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class OscillatoryGRN(nn.Module):
    """Gene Regulatory Network designed for oscillatory behavior."""

    def __init__(self, num_genes: int):
        super().__init__()
        self.num_genes = num_genes
        self.weights = nn.Parameter(torch.randn(num_genes, num_genes) * 0.1)
        self.thresholds = nn.Parameter(torch.ones(num_genes) * 0.5)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate the next state using a sigmoid activation function."""
        input_signal = torch.sigmoid(self.weights @ state - self.thresholds)
        return input_signal

    def integrate(self, state: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """Perform one step of Euler integration."""
        derivatives = self.forward(state) - state
        return state + dt * derivatives

def generate_target_pattern(num_genes: int, num_steps: int, period: int) -> torch.Tensor:
    """Create a target pattern of antiphase sinusoidal oscillations."""
    time = torch.linspace(0, num_steps / period, num_steps)
    target = torch.zeros(num_genes, num_steps)
    for gene in range(num_genes):
        target[gene, :] = 0.5 * (1 + torch.sin(2 * np.pi * time - (2 * np.pi * gene / num_genes)))
    return target

def optimize_grn(
    num_genes: int, 
    num_steps: int, 
    period: int, 
    learning_rate: float = 0.01, 
    num_epochs: int = 1000, 
    annealing_period: int = 200
):
    """Optimize the GRN to match the target oscillatory pattern with cosine annealing."""
    grn = OscillatoryGRN(num_genes)
    optimizer = torch.optim.Adam(grn.parameters(), lr=learning_rate)
    
    # Apply a cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=annealing_period, eta_min=0.0001)
    
    # Initialize the state and target
    state = torch.rand(num_genes, requires_grad=False)
    target = generate_target_pattern(num_genes, num_steps, period)
    
    loss_fn = nn.MSELoss()
    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Simulate the GRN
        state_history = torch.zeros(num_genes, num_steps, requires_grad=False)
        current_state = state.clone()
        for t in range(num_steps):
            current_state = grn.integrate(current_state)
            state_history[:, t] = current_state
        
        # Compute the loss
        loss = loss_fn(state_history, target)
        loss.backward()
        optimizer.step()
        
        # Update the learning rate using the scheduler
        scheduler.step()
        
        loss_history.append(loss.item())
        if epoch % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Learning Rate: {current_lr:.6f}")
    
    return grn, loss_history


def plot_results(grn: OscillatoryGRN, num_genes: int, num_steps: int, period: int, loss_history: List[float]):
    """Visualize the results of the optimization."""
    state = torch.rand(num_genes, requires_grad=False)
    state_history = torch.zeros(num_genes, num_steps, requires_grad=False)
    
    current_state = state.clone()
    for t in range(num_steps):
        current_state = grn.integrate(current_state)
        state_history[:, t] = current_state
    
    target = generate_target_pattern(num_genes, num_steps, period)
    
    # Plot the loss history
    plt.figure()
    plt.plot(loss_history)
    plt.title("Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()

    # Plot the oscillatory behavior and the target
    plt.figure()
    time = np.arange(num_steps)
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    for i in range(num_genes):
        plt.plot(time, state_history[i, :].detach().numpy(), color=colors[i % len(colors)], label=f"Gene {i+1} Output", linestyle='-')
        plt.plot(time, target[i, :].detach().numpy(), color=colors[i % len(colors)], label=f"Gene {i+1} Target", linestyle='--')
    
    plt.title("Optimized Oscillatory Behavior")
    plt.xlabel("Time Steps")
    plt.ylabel("Expression Level")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    num_genes = 4
    num_steps = 500
    period = 100
    learning_rate = 0.01
    num_epochs = 10000
    annealing_period = 200

    grn, loss_history = optimize_grn(num_genes, num_steps, period, learning_rate, num_epochs, annealing_period)
    plot_results(grn, num_genes, num_steps, period, loss_history)
