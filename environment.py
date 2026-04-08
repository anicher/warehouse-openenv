# Add to BOTTOM of your existing environment.py:

# OpenEnv API Endpoints (HTTP server compatible)
def openenv_state():
    return {
        "tasks": ["single_pick", "multi_order", "efficiency_challenge"],
        "action_space": {"type": "discrete", "size": 6},
        "observation_space": "Dict[grid:Box(15,15),robot_pos:Box(2),...]"
    }

def openenv_reset(task: str = "single_pick"):
    env = WarehouseEnv()
    obs = env.reset(task)
    return {
        "observation": obs,
        "reward": 0.0, 
        "done": False,
        "info": {"task": task}
    }

def openenv_step(action: int, task: str):
    env = WarehouseEnv()
    obs = env.reset(task)
    env.last_obs = obs
    next_obs, reward, done, info = env.step(action)
    return {
        "observation": next_obs,
        "reward": reward,
        "done": done, 
        "info": info
    }
