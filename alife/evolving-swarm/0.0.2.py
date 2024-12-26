# /// script
# title = "3D Swarm Simulation with Enhanced Dynamics in Napari"
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
# ]
# ///

import napari
import numpy as np
import matplotlib.pyplot as plt
from qtpy.QtCore import QTimer
from random import uniform, random


class SwarmAgent:
    def __init__(self, position, energy):
        self.position = np.array(position)
        self.velocity = np.random.rand(3) * 2 - 1
        self.acceleration = np.zeros(3)
        self.energy = energy
        self.breed_count = 0

    def update(self, neighbors, energy_sources):
        # Energy consumption for movement
        self.energy -= 0.005

        # Interaction with neighbors
        for neighbor in neighbors:
            distance = np.linalg.norm(self.position - neighbor.position)
            if distance < 2.0:  # Collision penalty
                self.energy -= 0.01

        # Interaction with energy sources
        for source in energy_sources:
            distance = np.linalg.norm(self.position - source["position"])
            if distance < source["radius"] and source["value"] > 0:
                self.energy += min(source["value"], 0.1)  # Consume food energy
                source["value"] = max(source["value"] - 0.1, 0)

        # Update position and velocity
        self.velocity += self.acceleration
        self.velocity = self.velocity / np.linalg.norm(self.velocity)  # Normalize velocity
        self.position += self.velocity

    def reproduce(self, world):
        if self.energy > 1.5:
            offset = np.random.uniform(-2.0, 2.0, 3)
            child = SwarmAgent(self.position + offset, self.energy / 2)
            self.energy /= 2
            world.add_entity(child)

    def is_alive(self):
        return self.energy > 0


class FoodSource:
    def __init__(self, position, energy):
        self.position = position
        self.radius = 5
        self.value = energy

    def replenish(self):
        self.value = min(self.value + 0.08, 2.0)

    def migrate(self, bounds):
        if random() < 0.01:  # Migration probability
            self.position = np.random.uniform(0, bounds, 3)


class SimulationWorld:
    def __init__(self, bounds):
        self.bounds = bounds
        self.agents = []
        self.food_sources = []

    def add_entity(self, entity):
        if isinstance(entity, SwarmAgent):
            self.agents.append(entity)
        elif isinstance(entity, FoodSource):
            self.food_sources.append(entity)

    def update(self):
        for agent in self.agents:
            neighbors = [a for a in self.agents if np.linalg.norm(a.position - agent.position) < 10 and a != agent]
            agent.update(neighbors, self.food_sources)
            agent.reproduce(self)

        for food in self.food_sources:
            food.replenish()
            food.migrate(self.bounds)

        # Remove dead agents
        self.agents = [agent for agent in self.agents if agent.is_alive()]


def main():
    viewer = napari.Viewer(ndisplay=3)
    bounds = 200
    world = SimulationWorld(bounds)

    # Initialize agents and food sources
    for _ in range(100):
        position = np.random.uniform(0, bounds, 3)
        world.add_entity(SwarmAgent(position, energy=1.0))

    for _ in range(25):
        position = np.random.uniform(0, bounds, 3)
        world.add_entity(FoodSource(position, energy=2.0))

    agent_layer = viewer.add_points(
        np.array([agent.position for agent in world.agents]),
        size=1,
        name="Agents",
        face_color="gray",
    )
    food_layer = viewer.add_points(
        np.array([food.position for food in world.food_sources]),
        size=5,
        name="Food Sources",
        face_color=[[1, 1, 0, 1] for _ in world.food_sources],
    )

    def update_viewer():
        world.update()

        # Update agent and food positions
        agent_positions = np.array([agent.position for agent in world.agents])
        agent_colors = np.array([
            [1 - min(agent.energy / 1.5, 1), min(agent.energy / 1.5, 1), 0, 1] for agent in world.agents
        ])
        food_positions = np.array([food.position for food in world.food_sources])
        food_colors = np.array([
            [1, min(food.value / 2.0, 1), 0, 1] for food in world.food_sources
        ])

        agent_layer.data = agent_positions
        agent_layer.face_color = agent_colors
        food_layer.data = food_positions
        food_layer.face_color = food_colors

    timer = QTimer()
    timer.timeout.connect(update_viewer)
    timer.start(100)  # Update every 100 ms

    napari.run()


if __name__ == "__main__":
    main()
