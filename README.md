# Mountain Car Genetic Algorithm

## Scenario: The Mountain Car Problem

The Mountain Car environment is a classic reinforcement learning scenario. A car is stuck in a valley between two hills and must reach the flag at the top of the right hill. However, the car's engine is not strong enough to drive directly up the hill. The agent must learn to build momentum by driving back and forth to reach the goal.

- **State:** The state consists of the car's position and velocity.
- **Actions:**
  - `0`: Push left
  - `1`: No push
  - `2`: Push right
- **Goal:** Reach the flag at position `0.5` as quickly as possible.


This scenario is challenging because the optimal solution requires the agent to first move away from the goal to gain enough momentum to reach it.

### Example of a Solution

A solution is a list of 200 actions (integers 0, 1, or 2), for example:

```python
[0, 0, 2, 2, 2, 1, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 1, 2, 2, 0, ...]
```
Each number represents an action at a time step:
- `0`: Push left
- `1`: No push
- `2`: Push right

The genetic algorithm evolves these sequences to find one that gets the car to the flag as quickly as possible.

This project implements a Genetic Algorithm (GA) to solve the OpenAI Gymnasium `MountainCar-v0` environment. The goal is to evolve a sequence of actions that allows the car to reach the flag in as few steps as possible.


### Mountain Car Environment Visualization
![Mountain Car Screenshot](images/Captura%20de%20tela%202025-07-12%20181354.png)

*Screenshot: Visualization of the Mountain Car environment using pygame rendering. The car must reach the flag at the top right by building momentum.*

## Project Structure

```
   app.py                # Main script: runs the genetic algorithm
   mountain_car.py       # MountainCar environment wrapper and utilities
   requirements.txt      # Python dependencies
   data/
      mountain-car-best-solution.pickle  # Saved best solutions
```

## How It Works

### 1. Genetic Algorithm (GA)
- **Population:** Each individual is a list of 200 actions (0, 1, or 2) representing moves in the environment.
- **Fitness:** Lower is better. The score is based on how quickly the car reaches the flag, or how close it gets if it fails.
- **Selection:** Tournament selection (k=2) is used to pick parents for the next generation.
- **Crossover:** Two-point crossover combines two parents to create two children.
- **Mutation:** Each gene (action) has a small probability of being randomly changed.
- **Elitism (Hall of Fame):** The best 4 individuals are preserved in each generation and passed to the next generation unchanged.

### 2. Saving and Replaying Solutions
- The best solution (and optionally more) is saved to `data/mountain-car-best-solution.pickle` using Python's `pickle` module.
- You can replay a saved solution visually using the `MountainCar` class with `renderMode="human"`.


### 3. Main Files

#### `app.py`
- Runs the genetic algorithm for 300 generations with a population of 100.
- Prints statistics for each generation (min and average fitness).
- Replays the best solution every 50 generations.
- At the end, prints, replays and saves the best solution found.

#### `mountain_car.py`
- Wraps the Gymnasium `MountainCar-v0` environment.
- Provides methods to evaluate a solution, save actions, and replay the saved solution.
- Handles serialization/deserialization of the solution.

## Understanding the output

During execution, the program prints statistics for each generation in the following format:

```
gen    population_size    min_fitness    avg_fitness
1      100                0.7087         1.0324
2      100                0.6500         0.9800
...
```

- **gen**: The current generation number (starting from 1).
- **population_size**: Number of individuals in the population (should remain constant).
- **min_fitness**: The best (lowest) fitness value in the current generation.
- **avg_fitness**: The average fitness value of the population in the current generation.

Lower fitness values are better. As generations progress, you should see the min and average fitness decrease, indicating that the population is evolving better solutions.

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the genetic algorithm:**
   ```sh
   python app.py
   ```

3. **Replay the best solution:**
   You can run the following to see the best solution in action:
   ```sh
   python mountain_car.py
   ```
   This will load and replay the saved actions visually.

## Key Parameters
- `POPULATION_SIZE`: Number of individuals per generation (default: 100)
- `MAX_GENERATIONS`: Number of generations to run (default: 300)
- `HALL_OF_FAME_SIZE`: Number of elites preserved each generation (default: 4)
- `P_CROSSOVER`: Probability of crossover (default: 0.9)
- `P_MUTATION`: Probability of mutation (default: 0.5)
- `REPLAY_INDIVIDUAL_FREQUENCY`: Frequency (in generations) to replay the best solution (default: 50)

## Notes
- The environment is managed using the [Gymnasium](https://gymnasium.farama.org/) API.
- The project is designed for Python 3.13+.

## Troubleshooting
- If you see errors about missing dependencies, ensure you have installed everything from `requirements.txt`.
- If the replay window does not appear, make sure your system supports OpenAI Gym rendering (may require additional packages on some platforms).

## Learn More

For further details about the Mountain Car environment, visit the official Gymnasium documentation:

- [MountainCar-v0 — Gymnasium Documentation](https://gymnasium.farama.org/environments/classic_control/mountain_car/)

## License
This project is for educational purposes.