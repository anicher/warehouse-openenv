# TEST VERSION - Expand later
import numpy as np
from typing import Dict, Any, Tuple

class WarehouseEnv:
    def __init__(self):
        self.task = None
        self.score_val = 0.0
    
    def reset(self, task: str) -> Dict[str, Any]:
        self.task = task
        grid = np.zeros((15,15))
        grid[7,7] = 2  # robot
        self.last_obs = {'grid': grid, 'robot_pos': np.array([7,7]), 'inventory': np.zeros(5), 'pending_orders': np.zeros((3,3)), 'time_step': 0}
        return self.last_obs
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        self.last_obs['time_step'] += 1
        reward = 0.1 if action == 4 else -0.01  # pick bonus
        done = self.last_obs['time_step'] > 20
        self.score_val = min(self.last_obs['time_step'] / 50.0, 1.0)
        return self.last_obs, reward, done, {}
    
    def score(self) -> float:
        return self.score_val

# OpenEnv API
def reset(task: str) -> Dict:
    env = WarehouseEnv()
    return env.reset(task)

def step(action: int, task: str = None) -> Tuple:
    env = WarehouseEnv()
    obs = env.reset(task or "single_pick")
    return env.step(action)

def state() -> Dict:
    return {"tasks": ["single_pick", "multi_order", "efficiency_challenge"]}
