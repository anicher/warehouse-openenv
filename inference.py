#!/usr/bin/env python3
import asyncio
import os
from typing import List
from openai import OpenAI
from environment import WarehouseEnv, openenv_reset, openenv_step

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "mock")
TASKS = ["single_pick", "multi_order", "efficiency_challenge"]
MAX_STEPS = [50, 100, 200]

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(n: int, action: str, reward: float, done: bool, error=None):
    e = error or "null"
    print(f"[STEP] step={n} action={action} reward={reward:.2f} done={str(done).lower()} error={e}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)

def get_action(obs):
    """Reproducible heuristic"""
    grid = obs['grid']
    r, c = map(int, obs['robot_pos'])
    items = np.argwhere(grid == 3)
    if len(items):
        tr, tc = items[0]
        dr, dc = tr-r, tc-c
        if abs(dr) > abs(dc): return "0" if dr < 0 else "1"
        return "2" if dc < 0 else "3"
    return "4"

async def run_task(task: str, max_step: int):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    log_start(task, "warehouse-fulfillment", MODEL_NAME)
    
    result = openenv_reset(task)
    obs, rewards, steps, success = result["observation"], [], 0, False
    
    try:
        for step in range(1, max_step + 1):
            action_str = get_action(obs)
            action_int = int(action_str)
            
            result = openenv_step(action_int, task)
            obs, reward, done = result["observation"], result["reward"], result["done"]
            
            rewards.append(reward)
            steps = step
            log_step(step, action_str, reward, done)
            
            if done: break
        
        score = min(sum(rewards) / (max_step * 2.5), 1.0)
        success = score >= 0.5
    except Exception as e:
        log_step(steps+1, "ERROR", 0.0, True, str(e))
    
    log_end(success, steps, score, rewards)

async def main():
    for task, maxs in zip(TASKS, MAX_STEPS):
        await run_task(task, maxs)

if __name__ == "__main__":
    asyncio.run(main())
