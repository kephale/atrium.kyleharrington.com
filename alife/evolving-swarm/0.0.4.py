# /// script
# title = "3D evolving swarm"
# description = "A 3D swarm simulation using energy dynamics, collision handling, and reproduction."
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.1.0"
# keywords = ["swarm simulation", "genetic algorithms", "napari", "3D visualization"]
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.12",
#     "Topic :: Scientific/Engineering :: Visualization",
# ]
# requires-python = ">=3.12"
# dependencies = [
#     "napari[all]>=0.5.4",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "rstar-python @ git+https://github.com/kephale/rstar-python"
# ]
# ///

import napari
import numpy as np
import matplotlib.pyplot as plt
from rstar_python import PyRTree
from qtpy.QtCore import QTimer
import pandas as pd
import os


class SwarmAgent:
    def __init__(self, position, energy, hidden_size=4):
        self.position = np.array(position)
        self.velocity = np.random.rand(3) * 2 - 1
        self.acceleration = np.zeros(3)
        self.energy = energy
        self.age = 0
        self.hidden_state = np.random.rand(hidden_size) * 0.1
        self.weights = np.random.rand(hidden_size + 2, 3) - 0.5

    def update(self, neighbors, energy_sources):
        # Reset acceleration
        self.acceleration = np.zeros(3)

        # Gather features
        num_neighbors = len(neighbors)
        avg_neighbor_direction = np.zeros(3)
        avg_neighbor_orientation = np.zeros(3)
        avg_neighbor_distance = 0.0

        if num_neighbors > 0:
            avg_neighbor_direction = np.mean([neighbor.position - self.position for neighbor in neighbors], axis=0)
            avg_neighbor_orientation = np.mean([neighbor.velocity for neighbor in neighbors], axis=0)
            avg_neighbor_distance = np.mean([np.linalg.norm(neighbor.position - self.position) for neighbor in neighbors])

        energy_distances = np.array([np.linalg.norm(source.position - self.position) for source in energy_sources])
        closest_energy_source = np.argmin(energy_distances) if len(energy_sources) > 0 else None
        vector_to_closest_energy = (
            (energy_sources[closest_energy_source].position - self.position)
            if closest_energy_source is not None
            else np.zeros(3)
        )
        distance_to_closest_food = energy_distances[closest_energy_source] if closest_energy_source is not None else float('inf')

        # Check for food consumption
        if closest_energy_source is not None and distance_to_closest_food < 5.0:
            food_source = energy_sources[closest_energy_source]
            if food_source.energy > 0.1:
                energy_gained = min(food_source.energy, 0.1)
                self.energy += energy_gained
                food_source.energy -= energy_gained        

        # Combine features with weights
        inputs = np.concatenate(([avg_neighbor_distance], [distance_to_closest_food], self.hidden_state))
        avg_neighbor_direction_weight = np.dot(inputs, self.weights[:, 0])
        avg_neighbor_orientation_weight = np.dot(inputs, self.weights[:, 1])
        vector_to_closest_energy_weight = np.dot(inputs, self.weights[:, 2])

        # Apply weights to compute acceleration
        self.acceleration += avg_neighbor_direction_weight * avg_neighbor_direction
        self.acceleration += avg_neighbor_orientation_weight * avg_neighbor_orientation
        self.acceleration += vector_to_closest_energy_weight * vector_to_closest_energy

        # Limit acceleration
        if np.linalg.norm(self.acceleration) > 10.0:
            self.acceleration /= np.linalg.norm(self.acceleration)

        # Update velocity and position
        self.velocity += self.acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > 1.0:
            self.velocity /= speed
        self.position += self.velocity

        # Update hidden state
        self.hidden_state += np.tanh(self.hidden_state)
        self.hidden_state = np.clip(self.hidden_state, -1.0, 1.0)

        # Consume energy and age
        self.energy -= 0.005
        self.age += 1

    def reproduce(self):
        if self.energy > 1.5:
            self.energy /= 2.1
            new_weights = self.weights + (np.random.rand(*self.weights.shape) - 0.5) * 0.1
            new_hidden_state = np.random.rand(len(self.hidden_state)) * 0.1
            return SwarmAgent(self.position + np.random.uniform(-2, 2, 3), self.energy, len(new_hidden_state)), new_weights
        return None

    def is_alive(self):
        return self.energy > 0


class FoodSource:
    def __init__(self, position, energy, radius=10.0):
        self.position = np.array(position)
        self.energy = energy
        self.radius = radius

    def replenish(self):
        self.energy = min(self.energy + 0.02, 2.0)

    def migrate(self, bounds):
        self.position = np.random.uniform(0, bounds, 3)
        self.energy = 2.0


def main():
    viewer = napari.Viewer(ndisplay=3)
    bounds = 200
    min_population = 100
    agents = [SwarmAgent(np.random.uniform(0, bounds, 3), energy=1.0 - np.random.rand() * 0.2) for _ in range(min_population)]
    energy_sources = [FoodSource(np.random.uniform(0, bounds, 3), energy=2.0) for _ in range(25)]

    agent_layer = viewer.add_points(
        np.array([agent.position for agent in agents]),
        size=1,
        name="Agents",
        face_color="gray",
    )
    food_layer = viewer.add_points(
        np.array([source.position for source in energy_sources]),
        size=5,
        name="Energy Sources",
        face_color=[[1, 1, 0, 1] for _ in energy_sources],
    )

    # Initialize live plots
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Swarm Simulation Metrics")
    time_steps = []
    population_sizes = []
    total_agent_energies = []
    total_food_energies = []
    average_ages = []

    def update_plot():
        ax.clear()
        ax.plot(time_steps, population_sizes, label="Population Size")
        ax.plot(time_steps, total_agent_energies, label="Total Agent Energy")
        ax.plot(time_steps, total_food_energies, label="Total Food Energy")
        ax.plot(time_steps, average_ages, label="Average Age of Agents")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Values")
        ax.legend()
        fig.canvas.draw_idle()
        plt.pause(0.001)

    def save_population(step):
        data = {
            "Position": [agent.position.tolist() for agent in agents],
            "Energy": [agent.energy for agent in agents],
            "Age": [agent.age for agent in agents],
        }
        os.makedirs("population_data", exist_ok=True)
        pd.DataFrame(data).to_csv(f"population_data/step_{step}.csv", index=False)

    def respawn_agents():
        if len(agents) < min_population:
            num_to_spawn = min_population - len(agents)
            new_agents = [SwarmAgent(np.random.uniform(0, bounds, 3), energy=1.0) for _ in range(num_to_spawn)]
            agents.extend(new_agents)

    def update_viewer():
        nonlocal agents, energy_sources

        # Build R* tree for efficient neighbor queries
        rtree = PyRTree(dims=3)
        positions = [agent.position.tolist() for agent in agents]
        for pos in positions:
            rtree.insert(pos)

        # Update agents
        new_agents = []
        for i, agent in enumerate(agents):
            # Query neighbors using R* tree
            neighbor_positions = rtree.neighbors_within_radius(agent.position.tolist(), radius=50.0)
            # Find indices of neighbors by matching positions
            neighbors = [agents[j] for j, pos in enumerate(positions) 
                       if pos in neighbor_positions and j != i]
            agent.update(neighbors, energy_sources)
            offspring = agent.reproduce()
            if offspring:
                new_agents.append(offspring[0])
        agents.extend(new_agents)
        agents = [agent for agent in agents if agent.is_alive()]

        # Respawn agents if below threshold
        respawn_agents()

        # Update food sources
        for source in energy_sources:
            if source.energy <= 0.1 or np.random.rand() < 0.001:
                source.migrate(bounds)
            else:
                source.replenish()

        # Update viewer
        agent_positions = np.array([agent.position for agent in agents])
        agent_colors = [[1 - agent.energy, agent.energy, 0, 1] for agent in agents]
        food_positions = np.array([source.position for source in energy_sources])
        food_colors = [[1, source.energy / 2, 0, 1] for source in energy_sources]

        agent_layer.data = agent_positions
        agent_layer.face_color = agent_colors
        food_layer.data = food_positions
        food_layer.face_color = food_colors

        # Update metrics and plot
        time_steps.append(len(time_steps))
        population_sizes.append(len(agents))
        total_agent_energies.append(sum(agent.energy for agent in agents))
        total_food_energies.append(sum(source.energy for source in energy_sources))
        average_ages.append(np.mean([agent.age for agent in agents]) if agents else 0)

        if len(time_steps) % 100 == 0:
            save_population(len(time_steps))

        update_plot()

    timer = QTimer()
    timer.timeout.connect(update_viewer)
    timer.start(100)

    napari.run()


if __name__ == "__main__":
    main()