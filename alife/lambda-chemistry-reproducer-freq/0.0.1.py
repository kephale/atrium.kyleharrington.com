# /// script
# title = "Lambda Chemistry Reproducer Frequency"
# description = "Implements a minimal lambda chemistry system with PyTorch and plots reproducer frequency."
# author = "AI Assistant via PR"
# license = "MIT"
# version = "0.0.1"
# keywords = ["lambda chemistry", "artificial life", "reproducers", "population dynamics", "pytorch"]
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.11",
#     "Topic :: Scientific/Engineering :: Artificial Life",
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
import random
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum

class ExpressionType(Enum):
    VARIABLE = 0
    ABSTRACTION = 1
    APPLICATION = 2

class LambdaChemistryNet(nn.Module):
    """Neural network model for lambda chemistry simulation."""
    
    def __init__(self, expr_type_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        # Expression type embedding
        self.expr_type_embedding = nn.Embedding(expr_type_dim, hidden_dim)
        
        # Interaction network
        self.interaction_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Reproduction probability prediction
        self.reproduction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Expression type prediction
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, expr_type_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, expr1_type: torch.Tensor, expr2_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict interaction outcomes between two expressions.
        
        Args:
            expr1_type: Type indices of first expressions [batch_size]
            expr2_type: Type indices of second expressions [batch_size]
            
        Returns:
            Tuple of (reproduction_prob, resulting_expr_type_probs)
        """
        # Get embeddings for both expressions
        expr1_emb = self.expr_type_embedding(expr1_type)
        expr2_emb = self.expr_type_embedding(expr2_type)
        
        # Concatenate embeddings and pass through interaction network
        concat_emb = torch.cat([expr1_emb, expr2_emb], dim=1)
        interaction_emb = self.interaction_net(concat_emb)
        
        # Predict reproduction probability and result type
        repro_prob = self.reproduction_head(interaction_emb)
        result_type_probs = self.type_head(interaction_emb)
        
        return repro_prob, result_type_probs

class Expression:
    """Base class for lambda calculus expressions."""
    
    def __init__(self, expr_type: ExpressionType):
        self.expr_type = expr_type
    
    def __repr__(self) -> str:
        return f"Expression({self.expr_type})"
    
    def to_tensor(self) -> int:
        """Convert the expression type to a tensor index."""
        return self.expr_type.value
    
    @staticmethod
    def from_tensor(expr_type_idx: int) -> 'Expression':
        """Create an expression from a tensor index."""
        expr_type = ExpressionType(expr_type_idx)
        
        if expr_type == ExpressionType.VARIABLE:
            return Variable()
        elif expr_type == ExpressionType.ABSTRACTION:
            return Abstraction()
        else:  # APPLICATION
            return Application()

class Variable(Expression):
    """Variable in lambda calculus."""
    
    def __init__(self):
        super().__init__(ExpressionType.VARIABLE)
    
    def __repr__(self) -> str:
        return "Variable"

class Abstraction(Expression):
    """Lambda abstraction (function) in lambda calculus."""
    
    def __init__(self):
        super().__init__(ExpressionType.ABSTRACTION)
    
    def __repr__(self) -> str:
        return "Abstraction"

class Application(Expression):
    """Function application in lambda calculus."""
    
    def __init__(self):
        super().__init__(ExpressionType.APPLICATION)
    
    def __repr__(self) -> str:
        return "Application"

class LambdaChemistry:
    """A minimal lambda chemistry simulation environment using neural networks."""
    
    def __init__(self, 
                 population_size: int = 100,
                 reaction_pairs: int = 20,
                 max_steps: int = 1000,
                 reproducer_init_ratio: float = 0.1,
                 expr_type_dim: int = 3,
                 hidden_dim: int = 64,
                 device: str = "cpu",
                 learning_rate: float = 0.001,
                 seed: Optional[int] = None):
        """
        Initialize the lambda chemistry environment.
        
        Args:
            population_size: Initial number of expressions
            reaction_pairs: Number of expression pairs selected for reactions each step
            max_steps: Maximum number of simulation steps
            reproducer_init_ratio: Initial ratio of reproducers in the population
            expr_type_dim: Dimension of expression type embeddings
            hidden_dim: Hidden dimension of the neural network
            device: Device to run the model on
            learning_rate: Learning rate for optimizer
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            
        self.population_size = population_size
        self.reaction_pairs = reaction_pairs
        self.max_steps = max_steps
        self.reproducer_init_ratio = reproducer_init_ratio
        self.device = device
        
        # Initialize model
        self.model = LambdaChemistryNet(expr_type_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize population
        self.population = self._create_initial_population()
        
        # Define known reproducers (type combinations that should reproduce)
        self.reproducer_types = {
            (ExpressionType.APPLICATION.value, ExpressionType.ABSTRACTION.value)
        }
        
        # Tracking metrics
        self.reproducer_counts = []
        self.population_counts = []
        self.type_distributions = []
    
    def _create_initial_population(self) -> List[Expression]:
        """Create the initial population with a mix of expression types."""
        population = []
        
        # Determine number of each type
        reproducer_count = int(self.population_size * self.reproducer_init_ratio)
        non_reproducer_count = self.population_size - reproducer_count
        
        # Add reproducers (Applications)
        for _ in range(reproducer_count):
            population.append(Application())
        
        # Add non-reproducers (mix of Variables and Abstractions)
        for _ in range(non_reproducer_count):
            expr_type = random.choice([ExpressionType.VARIABLE, ExpressionType.ABSTRACTION])
            if expr_type == ExpressionType.VARIABLE:
                population.append(Variable())
            else:
                population.append(Abstraction())
        
        # Shuffle the population
        random.shuffle(population)
        return population
    
    def _count_reproducers(self) -> int:
        """Count the number of reproducers (Applications) in the current population."""
        return sum(1 for expr in self.population if expr.expr_type == ExpressionType.APPLICATION)
    
    def _get_type_distribution(self) -> Dict[ExpressionType, int]:
        """Get the distribution of expression types in the population."""
        distribution = {expr_type: 0 for expr_type in ExpressionType}
        
        for expr in self.population:
            distribution[expr.expr_type] += 1
            
        return distribution
    
    def _is_reproducer_pair(self, type1: int, type2: int) -> bool:
        """Check if a pair of expression types forms a reproducer pattern."""
        return (type1, type2) in self.reproducer_types or (type2, type1) in self.reproducer_types
    
    def _generate_training_data(self, batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate training data for the model."""
        # Create random pairs of expression types
        expr1_types = torch.randint(0, len(ExpressionType), (batch_size,), device=self.device)
        expr2_types = torch.randint(0, len(ExpressionType), (batch_size,), device=self.device)
        
        # Determine reproduction labels (1 if reproducer pair, 0 otherwise)
        repro_labels = torch.zeros(batch_size, 1, device=self.device)
        result_type_labels = torch.zeros(batch_size, len(ExpressionType), device=self.device)
        
        for i in range(batch_size):
            type1 = expr1_types[i].item()
            type2 = expr2_types[i].item()
            
            # If it's a reproducer pair, set reproduction label to 1
            # and result type to APPLICATION
            if self._is_reproducer_pair(type1, type2):
                repro_labels[i, 0] = 1.0
                result_type_labels[i, ExpressionType.APPLICATION.value] = 1.0
            else:
                # Otherwise, randomly assign a result type (weighted)
                if random.random() < 0.7:  # 70% chance to be the same as one of the inputs
                    result_type = random.choice([type1, type2])
                else:  # 30% chance to be a different type
                    available_types = [t.value for t in ExpressionType if t.value != type1 and t.value != type2]
                    result_type = random.choice(available_types) if available_types else type1
                
                result_type_labels[i, result_type] = 1.0
        
        return expr1_types, expr2_types, repro_labels, result_type_labels
    
    def _train_model(self, epochs: int = 5):
        """Train the model on generated data."""
        self.model.train()
        
        for _ in range(epochs):
            # Generate training data
            expr1_types, expr2_types, repro_labels, result_type_labels = self._generate_training_data()
            
            # Forward pass
            self.optimizer.zero_grad()
            repro_probs, result_type_probs = self.model(expr1_types, expr2_types)
            
            # Compute loss
            repro_loss = nn.BCELoss()(repro_probs, repro_labels)
            type_loss = nn.CrossEntropyLoss()(result_type_probs, torch.argmax(result_type_labels, dim=1))
            
            total_loss = repro_loss + type_loss
            
            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()
    
    def _react(self, expr1: Expression, expr2: Expression) -> Optional[Expression]:
        """
        Simulate a reaction between two expressions using the model.
        
        Args:
            expr1: First expression
            expr2: Second expression
            
        Returns:
            Resulting expression or None if no reproduction occurs
        """
        self.model.eval()
        
        # Convert expression types to tensors
        expr1_type = torch.tensor([expr1.to_tensor()], device=self.device)
        expr2_type = torch.tensor([expr2.to_tensor()], device=self.device)
        
        # Predict interaction
        with torch.no_grad():
            repro_prob, result_type_probs = self.model(expr1_type, expr2_type)
            
            # Determine if reproduction occurs
            if random.random() < repro_prob.item():
                # Determine the resulting expression type
                result_type_idx = torch.multinomial(result_type_probs, 1).item()
                
                # Create and return the new expression
                return Expression.from_tensor(result_type_idx)
            
        return None
    
    def simulate(self, verbose: bool = False) -> Tuple[List[int], List[int], List[Dict]]:
        """
        Run the lambda chemistry simulation.
        
        Args:
            verbose: Whether to print progress information
        
        Returns:
            Tuple of (reproducer_counts, population_counts, type_distributions)
        """
        # Initial training
        self._train_model(epochs=10)
        
        for step in range(self.max_steps):
            # Track metrics
            reproducer_count = self._count_reproducers()
            self.reproducer_counts.append(reproducer_count)
            self.population_counts.append(len(self.population))
            
            type_dist = self._get_type_distribution()
            self.type_distributions.append({k.value: v for k, v in type_dist.items()})
            
            # Print progress
            if verbose and step % (self.max_steps // 10) == 0:
                print(f"Step {step}: Population={len(self.population)}, "
                      f"Reproducers={reproducer_count} "
                      f"({reproducer_count/len(self.population)*100:.1f}%)")
                
                type_percentages = {
                    t.name: count/len(self.population)*100 
                    for t, count in type_dist.items()
                }
                print(f"Type Distribution: {type_percentages}")
            
            # Select reaction pairs
            if len(self.population) < 2:
                break
                
            new_expressions = []
            for _ in range(min(self.reaction_pairs, len(self.population) // 2)):
                # Select two expressions at random
                indices = random.sample(range(len(self.population)), 2)
                expr1 = self.population[indices[0]]
                expr2 = self.population[indices[1]]
                
                # Simulate reaction
                result = self._react(expr1, expr2)
                
                if result is not None:
                    new_expressions.append(result)
            
            # Periodically retrain the model
            if step % 50 == 0:
                self._train_model(epochs=3)
            
            # Update population
            if new_expressions:
                # Add new expressions
                self.population.extend(new_expressions)
                
                # Remove random expressions to maintain population size
                while len(self.population) > self.population_size * 1.5:
                    idx = random.randint(0, len(self.population) - 1)
                    self.population.pop(idx)
        
        return self.reproducer_counts, self.population_counts, self.type_distributions

def plot_results(
    reproducer_counts: List[int], 
    population_counts: List[int],
    type_distributions: List[Dict[int, int]],
    max_steps: int
):
    """
    Plot the results of the lambda chemistry simulation.
    
    Args:
        reproducer_counts: List of reproducer counts at each step
        population_counts: List of total population counts at each step
        type_distributions: List of dictionaries with type distributions
        max_steps: Total number of simulation steps
    """
    time_steps = np.arange(len(reproducer_counts))
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Population & Reproducers
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, reproducer_counts, 'r-', label='Reproducers')
    plt.plot(time_steps, population_counts, 'b-', label='Total Population')
    plt.title('Population Dynamics')
    plt.xlabel('Time Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Reproducer Frequency
    plt.subplot(2, 2, 2)
    reproducer_freq = [r/p*100 if p > 0 else 0 
                       for r, p in zip(reproducer_counts, population_counts)]
    plt.plot(time_steps, reproducer_freq, 'g-')
    plt.title('Reproducer Frequency')
    plt.xlabel('Time Steps')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # Plot 3: Type Distribution
    plt.subplot(2, 2, 3)
    
    expr_types = [t.value for t in ExpressionType]
    expr_names = [t.name for t in ExpressionType]
    
    type_counts = {t: [dist.get(t, 0) for dist in type_distributions] for t in expr_types}
    
    for t, counts in type_counts.items():
        plt.plot(time_steps, counts, label=expr_names[t])
    
    plt.title('Expression Type Distribution')
    plt.xlabel('Time Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Type Frequency
    plt.subplot(2, 2, 4)
    
    type_freqs = {}
    for t in expr_types:
        type_freqs[t] = [(count / pop) * 100 if pop > 0 else 0 
                         for count, pop in zip(type_counts[t], population_counts)]
    
    for t, freqs in type_freqs.items():
        plt.plot(time_steps, freqs, label=expr_names[t])
    
    plt.title('Expression Type Frequency')
    plt.xlabel('Time Steps')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def run_simulation(
    population_size: int = 100,
    steps: int = 500,
    reaction_pairs: int = 20,
    reproducer_ratio: float = 0.1,
    hidden_dim: int = 64,
    learning_rate: float = 0.001,
    seed: Optional[int] = None,
    verbose: bool = True
):
    """
    Run a lambda chemistry simulation with neural network-based interactions.
    
    Args:
        population_size: Initial population size
        steps: Number of simulation steps
        reaction_pairs: Number of reaction pairs per step
        reproducer_ratio: Initial ratio of reproducers
        hidden_dim: Hidden dimension of the neural network
        learning_rate: Learning rate for the optimizer
        seed: Random seed for reproducibility
        verbose: Whether to print progress information
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize and run simulation
    chemistry = LambdaChemistry(
        population_size=population_size,
        reaction_pairs=reaction_pairs,
        max_steps=steps,
        reproducer_init_ratio=reproducer_ratio,
        hidden_dim=hidden_dim,
        device=str(device),
        learning_rate=learning_rate,
        seed=seed
    )
    
    reproducer_counts, population_counts, type_distributions = chemistry.simulate(verbose=verbose)
    
    # Print summary
    final_pop = population_counts[-1]
    final_reproducers = reproducer_counts[-1]
    final_frequency = (final_reproducers / final_pop * 100) if final_pop > 0 else 0
    
    print(f"\nSimulation completed.")
    print(f"Final population: {final_pop}")
    print(f"Final reproducers: {final_reproducers} ({final_frequency:.1f}%)")
    
    # Plot the results
    plot_results(reproducer_counts, population_counts, type_distributions, steps)

if __name__ == "__main__":
    # Run the simulation with default parameters
    run_simulation(verbose=True)
