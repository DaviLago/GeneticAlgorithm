
import gymnasium as gym
import time
import pickle
import os
from typing import List, Optional

MAX_STEPS: int = 200
FLAG_LOCATION: float = 0.5
DATA_FILE_PATH: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "mountain-car-best-solution.pickle")

class MountainCar:
    """
    Wrapper for the MountainCar-v0 environment with solution evaluation, saving, and replay capabilities.
    """
    def __init__(self, randomSeed: int, renderMode: Optional[str] = None):
        self.env = gym.make('MountainCar-v0', render_mode=renderMode)
        self.randomSeed = randomSeed

    def __len__(self) -> int:
        return MAX_STEPS

    def getScore(self, actions: List[int]) -> float:
        """
        Evaluate a solution (list of actions) in the MountainCar environment.
        Lower score is better.
        Returns score: if flag reached, negative value proportional to steps used; else, distance to flag.
        """
        obs, _ = self.env.reset(seed=self.randomSeed)
        actionCounter = 0
        for action in actions:
            actionCounter += 1
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        if actionCounter < MAX_STEPS:
            score = 0 - (MAX_STEPS - actionCounter)/MAX_STEPS
        else:
            score = abs(obs[0] - FLAG_LOCATION)
        return score

    def saveActions(self, actions: List[int]) -> None:
        """
        Serialize and save a list of actions using pickle.
        """
        os.makedirs(os.path.dirname(DATA_FILE_PATH), exist_ok=True)
        with open(DATA_FILE_PATH, "wb") as f:
            pickle.dump(list(actions), f)

    def replaySavedActions(self) -> None:
        """
        Load and replay a saved list of actions.
        """
        with open(DATA_FILE_PATH, "rb") as file:
            savedActions = pickle.load(file)
        self.replay(savedActions)
        self.close()

    def replay(self, actions: List[int]) -> None:
        """
        Render and replay a list of actions.
        """
        print("Replaying saved actions...")
        self.env.reset(seed=self.randomSeed)
        for _, action in enumerate(actions, 1):
            self.env.render()
            _, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                print("Replay finished.")
                break
            else:
                time.sleep(0.02)

    def close(self) -> None:
        self.env.close()

def main():
    RANDOM_SEED = 42
    car = MountainCar(RANDOM_SEED, renderMode="human")
    car.replaySavedActions()

if __name__ == '__main__':
    main()