# /// script
# title = "3D Swarm Simulation with Genetic Algorithms in Napari"
# description = "A 3D swarm simulation with agents using mutation and crossover for behavior control, including energy dynamics and reproduction."
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
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
#     "typer",
# ]
# ///

import typer
import napari
import numpy as np
import random
from scipy.spatial.distance import cdist
from qtpy.QtCore import QTimer

app = typer.Typer(help="3D Swarm Simulation with Genetic Algorithms in Napari")

class SwarmAgent:
    def __init__(self, position, genome, energy):
        self.position = np.array(position)
        self.velocity = np.random.rand(3) * 2 - 1
        self.genome = genome
        self.energy = energy

    def update(self, neighbors, energy_sources):
        center_weight, alignment_weight, neighbor_weight = self.genome

        # Calculate swarm behaviors
        center_of_mass = np.mean([agent.position for agent in neighbors], axis=0) if neighbors else self.position
        closest_neighbor = min(neighbors, key=lambda a: np.linalg.norm(a.position - self.position), default=self).position
        alignment = np.mean([agent.velocity for agent in neighbors], axis=0) if neighbors else self.velocity

        # Update velocity based on genome
        self.velocity += (
            center_weight * (center_of_mass - self.position) +
            neighbor_weight * (closest_neighbor - self.position) +
            alignment_weight * alignment
        )
        self.velocity = self.velocity / np.linalg.norm(self.velocity)  # Normalize
        self.position += self.velocity

        # Consume energy for moving
        self.energy -= 0.5

        # Apply collision penalty
        collisions = sum(1 for agent in neighbors if np.linalg.norm(agent.position - self.position) < 1.5)
        self.energy -= collisions * 2

        # Check energy sources
        for source in energy_sources:
            if np.linalg.norm(self.position - source["position"]) < source["radius"]:
                self.energy += source["value"]
                source["depleted"] = True

    def reproduce(self):
        if self.energy > 100:
            child_genome = self.mutate_genome()
            self.energy /= 2
            return SwarmAgent(self.position + np.random.rand(3) - 0.5, child_genome, self.energy)
        return None

    def mutate_genome(self):
        return [max(0, g + np.random.uniform(-0.1, 0.1)) for g in self.genome]


def initialize_energy_sources(num_sources, bounds):
    return [
        {"position": np.random.uniform(0, bounds, 3), "radius": 2.0, "value": 50, "depleted": False}
        for _ in range(num_sources)
    ]


@app.command()
def run(
    num_agents: int = typer.Option(100, help="Initial number of agents in the simulation."),
    num_sources: int = typer.Option(20, help="Number of energy sources in the simulation."),
    bounds: float = typer.Option(100.0, help="Size of the simulation space (cube side length)."),
    update_interval: int = typer.Option(100, help="Update interval in milliseconds."),
):
    """
    Run the 3D swarm simulation with genetic algorithms in Napari.
    """
    viewer = napari.Viewer(ndisplay=3)

    # Initialize agents and energy sources
    agents = [SwarmAgent(np.random.uniform(0, bounds, 3), [1.0, 1.0, 1.0], 50) for _ in range(num_agents)]
    energy_sources = initialize_energy_sources(num_sources, bounds)

    # Visualization layers
    agent_layer = viewer.add_points(
        np.array([agent.position for agent in agents]),
        size=1,
        name="Agents",
        face_color="gray",
    )
    energy_layer = viewer.add_points(
        np.array([source["position"] for source in energy_sources if not source["depleted"]]),
        size=5,
        face_color="yellow",
        name="Energy Sources",
    )

    def update_viewer():
        nonlocal agents, energy_sources

        # Ensure minimum number of agents
        if len(agents) < num_agents / 2:
            agents.extend(
                [SwarmAgent(np.random.uniform(0, bounds, 3), [1.0, 1.0, 1.0], 50) for _ in range(num_agents // 2 - len(agents))]
            )

        new_agents = []
        for agent in agents:
            neighbors = [a for a in agents if np.linalg.norm(a.position - agent.position) < 10 and a != agent]
            agent.update(neighbors, energy_sources)
            offspring = agent.reproduce()
            if offspring:
                new_agents.append(offspring)

        agents += new_agents
        agents = [agent for agent in agents if agent.energy > 0]

        for source in energy_sources:
            if source["depleted"]:
                source["position"] = np.random.uniform(0, bounds, 3)
                source["depleted"] = False

        agent_positions = np.array([agent.position for agent in agents])
        agent_colors = np.array([
            [1 - min(agent.energy / 100, 1), min(agent.energy / 100, 1), 0] for agent in agents
        ])
        energy_positions = np.array([source["position"] for source in energy_sources if not source["depleted"]])

        agent_layer.data = agent_positions
        agent_layer.face_color = agent_colors
        energy_layer.data = energy_positions

    # Timer for updates
    timer = QTimer()
    timer.timeout.connect(update_viewer)
    timer.start(update_interval)

    napari.run()


if __name__ == "__main__":
    app()
