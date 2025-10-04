# /// script
# title = "Symbolic Regression with W&B"
# description = "Simple symbolic regression that evolves mathematical expressions and logs them to W&B"
# author = "Kyle Harrington <atrium@kyleharrington.com>"
# license = "MIT"
# version = "0.0.2"
# keywords = ["symbolic regression", "genetic programming", "machine learning", "wandb"]
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24.0",
#     "wandb>=0.15.0",
#     "scikit-learn>=1.3.0",
# ]
# ///

import numpy as np
import wandb
from sklearn.metrics import mean_squared_error, r2_score
import random
from typing import List, Callable, Tuple
import operator

# Simple symbolic regression using genetic programming
# Target: find expressions that fit data

class Expr:
    """Base class for expressions"""
    def eval(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __str__(self) -> str:
        raise NotImplementedError
    
    def complexity(self) -> int:
        return 1

class Const(Expr):
    def __init__(self, value: float):
        self.value = value
    
    def eval(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(x, self.value)
    
    def __str__(self) -> str:
        return f"{self.value:.3f}"

class Var(Expr):
    def eval(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def __str__(self) -> str:
        return "x"

class BinOp(Expr):
    def __init__(self, left: Expr, right: Expr, op: Callable, symbol: str):
        self.left = left
        self.right = right
        self.op = op
        self.symbol = symbol
    
    def eval(self, x: np.ndarray) -> np.ndarray:
        try:
            with np.errstate(all='ignore'):
                result = self.op(self.left.eval(x), self.right.eval(x))
                # Handle infinities and NaNs
                result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
                return result
        except:
            return np.zeros_like(x)
    
    def __str__(self) -> str:
        return f"({self.left} {self.symbol} {self.right})"
    
    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

def safe_div(a, b):
    """Protected division"""
    return np.where(np.abs(b) < 1e-10, 1.0, a / b)

def generate_random_expr(depth: int = 0, max_depth: int = 3) -> Expr:
    """Generate a random expression tree"""
    if depth >= max_depth or random.random() < 0.3:
        # Terminal: constant or variable
        if random.random() < 0.5:
            return Var()
        else:
            return Const(random.uniform(-5, 5))
    
    # Non-terminal: binary operation
    ops = [
        (operator.add, '+'),
        (operator.sub, '-'),
        (operator.mul, '*'),
        (safe_div, '/'),
    ]
    op, symbol = random.choice(ops)
    
    left = generate_random_expr(depth + 1, max_depth)
    right = generate_random_expr(depth + 1, max_depth)
    
    return BinOp(left, right, op, symbol)

def mutate(expr: Expr, mutation_rate: float = 0.1) -> Expr:
    """Mutate an expression"""
    if random.random() > mutation_rate:
        return expr
    
    if isinstance(expr, Const):
        return Const(expr.value + random.gauss(0, 1))
    elif isinstance(expr, Var):
        return Var() if random.random() < 0.5 else Const(random.uniform(-5, 5))
    elif isinstance(expr, BinOp):
        if random.random() < 0.5:
            return BinOp(mutate(expr.left, mutation_rate), 
                        mutate(expr.right, mutation_rate),
                        expr.op, expr.symbol)
        else:
            # Change operation
            ops = [(operator.add, '+'), (operator.sub, '-'), 
                   (operator.mul, '*'), (safe_div, '/')]
            op, symbol = random.choice(ops)
            return BinOp(expr.left, expr.right, op, symbol)
    
    return expr

def crossover(expr1: Expr, expr2: Expr) -> Tuple[Expr, Expr]:
    """Crossover two expressions"""
    # Simple crossover: swap subtrees
    if isinstance(expr1, BinOp) and isinstance(expr2, BinOp):
        if random.random() < 0.5:
            return (BinOp(expr1.left, expr2.right, expr1.op, expr1.symbol),
                   BinOp(expr2.left, expr1.right, expr2.op, expr2.symbol))
    return expr1, expr2

def fitness(expr: Expr, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate fitness (lower is better)"""
    try:
        y_pred = expr.eval(X)
        mse = mean_squared_error(y, y_pred)
        # Penalize complexity
        complexity_penalty = expr.complexity() * 0.001
        return mse + complexity_penalty
    except:
        return float('inf')

def generate_dataset(n_samples: int = 100, noise: float = 0.1, 
                     function: str = "x^2") -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset"""
    X = np.linspace(-10, 10, n_samples)
    
    if function == "x^2":
        y = X ** 2
    elif function == "sin":
        y = np.sin(X)
    elif function == "poly":
        y = 2 * X ** 2 - 3 * X + 1
    else:
        y = X ** 2
    
    y += np.random.normal(0, noise, n_samples)
    return X, y

def evolve(X: np.ndarray, y: np.ndarray,
           population_size: int = 100,
           generations: int = 50,
           mutation_rate: float = 0.1,
           wandb_run = None) -> Expr:
    """Evolve expressions to fit data"""
    
    # Initialize population
    population = [generate_random_expr() for _ in range(population_size)]
    
    best_ever_fitness = float('inf')
    best_ever_expr = None
    
    for gen in range(generations):
        # Evaluate fitness
        fitnesses = [fitness(expr, X, y) for expr in population]
        
        # Find best
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_expr = population[best_idx]
        
        # Track best ever
        if best_fitness < best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_expr = best_expr
        
        # Calculate R2 score for best
        y_pred = best_expr.eval(X)
        r2 = r2_score(y, y_pred)
        
        # Log to W&B
        if wandb_run:
            # Log numeric metrics
            wandb_run.log({
                "generation": gen,
                "best_fitness": best_fitness,
                "best_r2": r2,
                "best_complexity": best_expr.complexity(),
                "avg_fitness": np.mean(fitnesses),
            })
            
            # Log expression as HTML so it's readable
            wandb_run.log({
                "expression": wandb.Html(f"<pre>Gen {gen}: {str(best_expr)}</pre>")
            })
        
        print(f"Gen {gen}: fitness={best_fitness:.4f}, r2={r2:.4f}, expr={best_expr}")
        
        # Selection and reproduction
        # Tournament selection
        new_population = []
        for _ in range(population_size):
            # Tournament
            tournament_size = 3
            tournament_idx = random.sample(range(population_size), tournament_size)
            winner_idx = min(tournament_idx, key=lambda i: fitnesses[i])
            
            # Clone and mutate
            offspring = mutate(population[winner_idx], mutation_rate)
            new_population.append(offspring)
        
        # Crossover
        for i in range(0, population_size - 1, 2):
            if random.random() < 0.7:  # Crossover probability
                new_population[i], new_population[i+1] = crossover(
                    new_population[i], new_population[i+1]
                )
        
        population = new_population
    
    # Create table of top 10 final expressions
    if wandb_run:
        expr_data = []
        final_fitnesses = [fitness(expr, X, y) for expr in population]
        for expr, fit in sorted(zip(population, final_fitnesses), key=lambda x: x[1])[:10]:
            y_pred = expr.eval(X)
            r2 = r2_score(y, y_pred)
            expr_data.append([str(expr), fit, r2, expr.complexity()])
        
        table = wandb.Table(
            columns=["Expression", "Fitness", "R2", "Complexity"],
            data=expr_data
        )
        wandb_run.log({"top_10_expressions": table})
    
    return best_ever_expr

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Symbolic Regression with W&B")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--function", type=str, default="x^2", 
                       choices=["x^2", "sin", "poly"])
    parser.add_argument("--wandb-project", type=str, default="symbolic-regression")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()
    
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Generate dataset
    X, y = generate_dataset(args.n_samples, args.noise, args.function)
    
    # Initialize W&B
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={
            "population_size": args.population_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "n_samples": args.n_samples,
            "noise": args.noise,
            "target_function": args.function,
            "seed": args.seed,
        }
    )
    
    print(f"Target function: {args.function}")
    print(f"Dataset: {args.n_samples} samples with noise={args.noise}")
    
    # Evolve
    best_expr = evolve(X, y, 
                      args.population_size,
                      args.generations,
                      args.mutation_rate,
                      run)
    
    # Final evaluation
    y_pred = best_expr.eval(X)
    final_mse = mean_squared_error(y, y_pred)
    final_r2 = r2_score(y, y_pred)
    
    print(f"\nFinal best expression: {best_expr}")
    print(f"MSE: {final_mse:.6f}")
    print(f"R2: {final_r2:.6f}")
    print(f"Complexity: {best_expr.complexity()}")
    
    # Log final results
    run.log({
        "final_mse": final_mse,
        "final_r2": final_r2,
        "final_complexity": best_expr.complexity(),
    })
    
    # Create a text artifact for the best expression
    artifact = wandb.Artifact("best_expression", type="model")
    with artifact.new_file("expression.txt", mode="w") as f:
        f.write(f"Expression: {best_expr}\n")
        f.write(f"MSE: {final_mse:.6f}\n")
        f.write(f"R2: {final_r2:.6f}\n")
        f.write(f"Complexity: {best_expr.complexity()}\n")
    run.log_artifact(artifact)
    
    # Save to summary (visible in overview)
    run.summary["best_expression"] = str(best_expr)
    run.summary["final_r2"] = final_r2
    run.summary["final_mse"] = final_mse
    run.summary["final_complexity"] = best_expr.complexity()
    
    run.finish()
    
    print(f"\nView results at: {run.url}")

if __name__ == "__main__":
    main()