
import random
import numpy as np
import mountain_car
from typing import List, Tuple

# =====================
# Genetic Algorithm Config
# =====================
POPULATION_SIZE: int = 100
P_CROSSOVER: float = 0.9  # probability for crossover
P_MUTATION: float = 0.5   # probability for mutating an individual
MAX_GENERATIONS: int = 300
HALL_OF_FAME_SIZE: int = 4
REPLAY_INDIVIDUAL_FREQUENCY: int = 50  # how often to replay the best individual
RANDOM_SEED: int = 42

# =====================
# Environment Setup
# =====================
random.seed(RANDOM_SEED)
car = mountain_car.MountainCar(RANDOM_SEED)
car_replay = mountain_car.MountainCar(RANDOM_SEED, renderMode="human")
INDIVIDUAL_SIZE: int = len(car)

def create_individual() -> List[int]:
    """Create a random individual."""
    return [random.randint(0, 2) for _ in range(INDIVIDUAL_SIZE)]

def create_population(n: int) -> List[List[int]]:
    """Create a population of n individuals."""
    return [create_individual() for _ in range(n)]

def evaluate(individual: List[int]) -> float:
    """Evaluate an individual using the environment's scoring function."""
    return car.getScore(individual)

def tournament_selection(population: List[List[int]], fitnesses: List[float], k: int = 2) -> List[List[int]]:
    """Select individuals using tournament selection."""
    selected = []
    for _ in range(len(population)):
        aspirants = random.sample(list(zip(population, fitnesses)), k)
        selected.append(min(aspirants, key=lambda x: x[1])[0])
    return selected

def two_point_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Perform two-point crossover between two parents."""
    size = len(parent1)
    cxpoint1 = random.randint(1, size - 2)
    cxpoint2 = random.randint(cxpoint1 + 1, size - 1)
    child1 = parent1[:cxpoint1] + parent2[cxpoint1:cxpoint2] + parent1[cxpoint2:]
    child2 = parent2[:cxpoint1] + parent1[cxpoint1:cxpoint2] + parent2[cxpoint2:]
    return child1, child2

def mutate(individual: List[int], indpb: float) -> List[int]:
    """Mutate an individual with probability indpb per gene."""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(0, 2)
    return individual

def split_hall_of_fame(population: List[List[int]], fitnesses: List[float], size: int) -> List[Tuple[List[int], float]]:
    """Return the top N individuals and their fitnesses as Hall of Fame."""
    combined = list(zip(population, fitnesses))
    combined.sort(key=lambda x: x[1])  # Sort by fitness, best (lowest) first
    return combined[:size], combined[size:]

def replay_individual(individual: List[int], fitness: float):
    """Replay an individual and print its fitness."""
    print(f"\nReplaying individual (Fitness: {fitness}):")
    car_replay.replay(individual)

def print_generation_info(gen: int, population: List[List[int]], fitnesses: List[float]):
    """Print information about the current generation."""
    min_fit = np.min(fitnesses)
    avg_fit = np.mean(fitnesses)
    print(f"{gen}\t{len(population)}\t\t\t{min_fit:.2f}\t\t{avg_fit:.2f}")

def main():
    population = create_population(POPULATION_SIZE)
    fitnesses = [evaluate(ind) for ind in population]

    print("gen\tpopulation_size\t\tmin_fitness\tavg_fitness")
    print_generation_info(1, population, fitnesses)
    hall_of_fame: List[Tuple[List[int], float]] = []

    for gen in range(2, MAX_GENERATIONS + 1):

        # Update Hall of Fame
        hall_of_fame, others = split_hall_of_fame(population, fitnesses, HALL_OF_FAME_SIZE)

        # Remove Hall of Fame from the selection pool
        selection_pool = [ind for ind, _ in others]
        fitnesses_pool = [fit for _, fit in others]

        # Selection
        selected = tournament_selection(selection_pool, fitnesses_pool, k=2)
        
        # Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
            if random.random() < P_CROSSOVER:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate(ind, 1.0/INDIVIDUAL_SIZE) if random.random() < P_MUTATION else ind for ind in offspring]

        # Pass Hall of Fame to next generation
        next_population = offspring + [ind for ind, _ in hall_of_fame]

        # Evaluate
        fitnesses = [evaluate(ind) for ind in next_population]
        population = next_population

        # Print generation info
        print_generation_info(gen, population, fitnesses)

        # Replay best individual every REPLAY_INDIVIDUAL_FREQUENCY generations
        if gen % REPLAY_INDIVIDUAL_FREQUENCY == 0:
            best_ind, best_fit = hall_of_fame[0]
            replay_individual(best_ind, best_fit)

    # Best solution
    best_ind, best_fit = hall_of_fame[0]
    print("\nBest Solution = ", best_ind)
    print(f"Best Fitness = {best_fit:.2f}")

    # Save best solution for a replay
    car.saveActions(best_ind)
    car_replay.close()

if __name__ == "__main__":
    main()