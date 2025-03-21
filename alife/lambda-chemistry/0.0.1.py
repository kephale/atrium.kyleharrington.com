# /// script
# title = "Lambda Chemistry Simulation"
# description = "Implements a minimal lambda chemistry and plots reproducer frequency over time."
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.0.1"
# keywords = ["artificial chemistry", "lambda chemistry", "reproducer", "artificial life", "simulation"]
# classifiers = [
#     "Development Status :: 4 - Beta",
#     "Intended Audience :: Science/Research",
#     "License :: OSI Approved :: MIT License",
#     "Programming Language :: Python :: 3.10",
#     "Topic :: Scientific/Engineering :: Artificial Life",
# ]
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24.0",
#     "matplotlib",
#     "typer",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt
import typer
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
import time

app = typer.Typer(help="Lambda Chemistry simulation that tracks reproducer frequency")

class Expression:
    """Base class for lambda calculus expressions"""
    
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self) -> str:
        return self.name
    
    def reduce(self, env: Dict = None) -> 'Expression':
        """Reduce the expression"""
        return self

class Variable(Expression):
    """Variable in lambda calculus"""
    
    def __init__(self, name: str):
        super().__init__(name)

class Abstraction(Expression):
    """Lambda abstraction (function)"""
    
    def __init__(self, param: str, body: Expression):
        super().__init__(f"(λ{param}.{body})")
        self.param = param
        self.body = body
    
    def reduce(self, env: Dict = None) -> Expression:
        """Reduce the body of the abstraction"""
        if env is None:
            env = {}
        # In a more complete implementation, we'd handle name collisions
        # by renaming variables, but we'll keep it simple
        return Abstraction(self.param, self.body.reduce(env))

class Application(Expression):
    """Function application"""
    
    def __init__(self, func: Expression, arg: Expression):
        super().__init__(f"({func} {arg})")
        self.func = func
        self.arg = arg
    
    def reduce(self, env: Dict = None) -> Expression:
        """Beta reduction: Apply function to argument"""
        if env is None:
            env = {}
        
        # First, reduce the function part
        reduced_func = self.func.reduce(env)
        
        # If it's a lambda abstraction, we can apply it
        if isinstance(reduced_func, Abstraction):
            # Create new environment with parameter bound to argument
            new_env = env.copy()
            new_env[reduced_func.param] = self.arg
            
            # Substitute and reduce
            return reduced_func.body.reduce(new_env)
        
        # Otherwise, just reduce both sides
        reduced_arg = self.arg.reduce(env)
        return Application(reduced_func, reduced_arg)

def substitute(expr: Expression, var_name: str, replacement: Expression) -> Expression:
    """Substitute all occurrences of var_name with replacement in expr"""
    if isinstance(expr, Variable):
        if expr.name == var_name:
            return replacement
        return expr
    
    elif isinstance(expr, Abstraction):
        if expr.param == var_name:
            # Variable is bound, so don't substitute in the body
            return expr
        # Substitute in the body
        new_body = substitute(expr.body, var_name, replacement)
        return Abstraction(expr.param, new_body)
    
    elif isinstance(expr, Application):
        new_func = substitute(expr.func, var_name, replacement)
        new_arg = substitute(expr.arg, var_name, replacement)
        return Application(new_func, new_arg)
    
    return expr

class LambdaChemistry:
    """A minimal lambda chemistry simulation environment"""
    
    def __init__(self, 
                 population_size: int = 1000,
                 reaction_pool_size: int = 100,
                 max_steps: int = 1000,
                 mutation_rate: float = 0.05,
                 max_reduction_steps: int = 10,
                 seed: Optional[int] = None):
        """
        Initialize the lambda chemistry environment.
        
        Args:
            population_size: Initial number of expressions
            reaction_pool_size: Number of expressions selected for reactions each step
            max_steps: Maximum number of simulation steps
            mutation_rate: Probability of mutation during reproduction
            max_reduction_steps: Maximum reduction steps for an expression
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.population_size = population_size
        self.reaction_pool_size = reaction_pool_size
        self.max_steps = max_steps
        self.mutation_rate = mutation_rate
        self.max_reduction_steps = max_reduction_steps
        
        # Initialize with simple expressions
        self.population = self._create_initial_population()
        
        # Tracking metrics
        self.reproducer_counts = []
        self.population_counts = []
        
        # Define some known reproducers for tracking
        self.reproducers = self._identify_reproducers()
    
    def _create_initial_population(self) -> List[Expression]:
        """Create the initial population with simple lambda expressions"""
        population = []
        
        # Basic building blocks
        variable_names = ["x", "y", "z"]
        
        for _ in range(self.population_size):
            # Randomly decide what type of expression to create
            expr_type = random.choice(["var", "abs", "app"])
            
            if expr_type == "var":
                # Simple variable
                var_name = random.choice(variable_names)
                population.append(Variable(var_name))
                
            elif expr_type == "abs":
                # Lambda abstraction
                param = random.choice(variable_names)
                body = Variable(random.choice(variable_names))
                population.append(Abstraction(param, body))
                
            elif expr_type == "app":
                # Application
                func = Variable(random.choice(variable_names))
                arg = Variable(random.choice(variable_names))
                population.append(Application(func, arg))
        
        # Add some known reproducers to seed the population
        self_replicator = self._create_self_replicator()
        for _ in range(min(50, self.population_size // 20)):  # Add ~5% reproducers
            population[random.randint(0, len(population)-1)] = self_replicator
            
        return population
    
    def _create_self_replicator(self) -> Expression:
        """Create a simple self-replicating lambda expression"""
        # A simple quine in lambda calculus: ((λx.(x x)) (λx.(x x)))
        # This is not a "true" quine due to our simplified reduction model,
        # but it serves as a reproducer in our simulation
        x = "x"
        xx_body = Application(Variable(x), Variable(x))
        lx_xx = Abstraction(x, xx_body)
        return Application(lx_xx, lx_xx)
    
    def _identify_reproducers(self) -> Set[str]:
        """Identify known reproducer patterns to track"""
        reproducers = set()
        
        # Add the string representation of our known reproducers
        reproducers.add(str(self._create_self_replicator()))
        
        # Additional reproducers could be added here
        
        return reproducers
    
    def _is_reproducer(self, expr: Expression) -> bool:
        """Check if an expression is a known reproducer"""
        return str(expr) in self.reproducers
    
    def _count_reproducers(self) -> int:
        """Count the number of reproducers in the current population"""
        return sum(1 for expr in self.population if self._is_reproducer(expr))
    
    def _mutate(self, expr: Expression) -> Expression:
        """Randomly mutate an expression"""
        if random.random() > self.mutation_rate:
            return expr
        
        # Different mutation types
        mutation_type = random.choice(["change_var", "swap_parts", "new_random"])
        
        if mutation_type == "change_var" and (isinstance(expr, Variable) or 
                                            isinstance(expr, Abstraction)):
            # Change a variable name
            new_var = random.choice(["x", "y", "z"])
            if isinstance(expr, Variable):
                return Variable(new_var)
            else:  # Abstraction
                return Abstraction(new_var, expr.body)
                
        elif mutation_type == "swap_parts" and isinstance(expr, Application):
            # Swap function and argument
            return Application(expr.arg, expr.func)
            
        else:
            # Generate a new random expression
            var_names = ["x", "y", "z"]
            expr_type = random.choice(["var", "abs", "app"])
            
            if expr_type == "var":
                return Variable(random.choice(var_names))
            elif expr_type == "abs":
                return Abstraction(random.choice(var_names), 
                                Variable(random.choice(var_names)))
            else:  # app
                return Application(Variable(random.choice(var_names)), 
                                Variable(random.choice(var_names)))
    
    def simulate(self, verbose: bool = False) -> Tuple[List[int], List[int]]:
        """
        Run the lambda chemistry simulation.
        
        Args:
            verbose: Whether to print progress information
        
        Returns:
            Tuple of (reproducer_counts, population_counts)
        """
        for step in range(self.max_steps):
            # Select reaction pool
            reaction_pool = random.sample(
                self.population, 
                min(self.reaction_pool_size, len(self.population))
            )
            
            # Apply reactions: pair expressions and reduce them
            new_expressions = []
            for _ in range(max(1, len(reaction_pool) // 2)):
                if len(reaction_pool) < 2:
                    break
                
                # Select two expressions to react
                idx1, idx2 = random.sample(range(len(reaction_pool)), 2)
                expr1 = reaction_pool.pop(idx1)
                expr2 = reaction_pool.pop(idx2 if idx2 < idx1 else idx2 - 1)
                
                # Create a reaction (application)
                reaction = Application(expr1, expr2)
                
                # Reduce for a limited number of steps
                result = reaction
                for _ in range(self.max_reduction_steps):
                    new_result = result.reduce()
                    if str(new_result) == str(result):
                        break
                    result = new_result
                
                # Potentially mutate the result
                if random.random() < self.mutation_rate:
                    result = self._mutate(result)
                
                new_expressions.append(result)
            
            # Update population: remove some old expressions and add new ones
            if new_expressions:
                # Remove random expressions to make room for new ones
                removal_count = min(len(new_expressions), len(self.population))
                for _ in range(removal_count):
                    idx = random.randint(0, len(self.population) - 1)
                    self.population.pop(idx)
                
                # Add new expressions
                self.population.extend(new_expressions)
            
            # Track metrics
            reproducer_count = self._count_reproducers()
            self.reproducer_counts.append(reproducer_count)
            self.population_counts.append(len(self.population))
            
            # Print progress
            if verbose and step % (self.max_steps // 10) == 0:
                print(f"Step {step}: Population={len(self.population)}, "
                      f"Reproducers={reproducer_count} "
                      f"({reproducer_count/len(self.population)*100:.1f}%)")
        
        return self.reproducer_counts, self.population_counts

def plot_results(reproducer_counts: List[int], 
                population_counts: List[int], 
                max_steps: int):
    """
    Plot the results of the lambda chemistry simulation.
    
    Args:
        reproducer_counts: List of reproducer counts at each step
        population_counts: List of total population counts at each step
        max_steps: Total number of simulation steps
    """
    time_steps = np.arange(len(reproducer_counts))
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw counts
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, reproducer_counts, 'r-', label='Reproducers')
    plt.plot(time_steps, population_counts, 'b-', label='Total Population')
    plt.title('Lambda Chemistry Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    # Plot frequency (percentage)
    plt.subplot(2, 1, 2)
    reproducer_freq = [r/p*100 if p > 0 else 0 
                       for r, p in zip(reproducer_counts, population_counts)]
    plt.plot(time_steps, reproducer_freq, 'g-')
    plt.title('Reproducer Frequency')
    plt.xlabel('Time Steps')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

@app.command()
def run(
    population_size: int = typer.Option(500, help="Initial population size."),
    steps: int = typer.Option(200, help="Number of simulation steps."),
    reaction_pool: int = typer.Option(50, help="Size of reaction pool each step."),
    mutation_rate: float = typer.Option(0.05, help="Probability of mutation."),
    max_reductions: int = typer.Option(10, help="Maximum reduction steps per reaction."),
    seed: int = typer.Option(None, help="Random seed for reproducibility."),
    verbose: bool = typer.Option(True, help="Print progress information.")
):
    """
    Run a lambda chemistry simulation tracking reproducer frequency.
    """
    start_time = time.time()
    
    # Initialize and run simulation
    chemistry = LambdaChemistry(
        population_size=population_size,
        reaction_pool_size=reaction_pool,
        max_steps=steps,
        mutation_rate=mutation_rate,
        max_reduction_steps=max_reductions,
        seed=seed
    )
    
    reproducer_counts, population_counts = chemistry.simulate(verbose=verbose)
    
    # Print summary
    final_pop = population_counts[-1]
    final_reproducers = reproducer_counts[-1]
    final_frequency = (final_reproducers / final_pop * 100) if final_pop > 0 else 0
    
    print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds.")
    print(f"Final population: {final_pop}")
    print(f"Final reproducers: {final_reproducers} ({final_frequency:.1f}%)")
    
    # Plot the results
    plot_results(reproducer_counts, population_counts, steps)

if __name__ == "__main__":
    app()
