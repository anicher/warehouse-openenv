# environment.py - FIXED VERSION
import numpy as np
from typing import Dict, Any, List, Tuple, TypedDict
from gymnasium.spaces import Discrete, Box, Dict as SpaceDict  
from dataclasses import dataclass
import json

# ... rest of your code unchanged ...

@dataclass
class WarehouseState(TypedDict):
    grid: np.ndarray      # 15x15: 0=empty,1=wall,2=robot,3=item,4=order,5=shelf
    robot_pos: Tuple[int,int]
    inventory: Dict[str,int]
    pending_orders: List[Dict[str,str]]
    completed_orders: int
    time_step: int
    collisions: int
    score: float

class WarehouseEnv:
    def __init__(self):
        self.size = 15
        self.max_steps = 200
        self.action_space = Discrete(6)  # 0↑1↓2←3→4Pick5Pack
        self.observation_space = SpaceDict({
            'grid': Box(low=0, high=5, shape=(15,15), dtype=np.uint8),
            'robot_pos': Box(low=0, high=14, shape=(2,), dtype=np.int32),
            'inventory': Box(low=0, high=10, shape=(5,), dtype=np.int32),  # 5 item types
            'pending_orders': Box(low=0, high=5, shape=(3,3), dtype=np.int32),  # order matrix
            'time_step': Box(low=0, high=200, shape=(), dtype=np.int32)
        })
        self.current_step = 0
        self.task = None
        
    def reset(self, task: str = "single_pick") -> Dict[str, Any]:
        """Reset for specific task"""
        self.task = task
        self.current_step = 0
        state = self._generate_initial_state(task)
        return self._format_observation(state)
    
    def _generate_initial_state(self, task: str) -> WarehouseState:
        """Procedurally generate layouts per task difficulty"""
        grid = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Always place walls
        grid[0:3,:] = 1; grid[-3:,:] = 1; grid[:,0:3] = 1; grid[:,-3:] = 1
        
        if task == "single_pick":
            # EASY: Single item at fixed shelf
            grid[5,5] = 5  # shelf
            grid[5,6] = 3  # single item A
            robot_pos = (10,10)
            orders = [{"A": 1}]
            
        elif task == "multi_order":
            # MEDIUM: 3 items + obstacles
            grid[4:7,4:7] = 5  # shelf block
            grid[5,5] = 3; grid[5,6] = 3  # items A,B
            grid[6,5] = 3  # item C  
            grid[8,8] = 1  # obstacle
            robot_pos = (12,12)
            orders = [{"A":1,"B":1,"C":1}]
            
        else:  # efficiency_challenge
            # HARD: Dynamic multi-order
            shelves = [(3,3),(3,10),(10,3),(10,10)]
            for i, (x,y) in enumerate(shelves):
                grid[x,y] = 5
                grid[x,y+1] = 3  # item i
            robot_pos = (12,12)
            orders = [{"A":1,"B":1}, {"C":1,"D":1}, {"A":1,"C":1}, {"B":1,"D":1}, {"A":1,"B":1,"C":1}]
        
        return WarehouseState(
            grid=grid, robot_pos=robot_pos, inventory={"A":0,"B":0,"C":0,"D":0,"E":0},
            pending_orders=orders, completed_orders=0, time_step=0, collisions=0, score=0.0
        )
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        """Execute action and compute dense reward"""
        state = self._parse_observation(self.last_obs)
        reward = 0.0
        done = False
        
        # Move robot
        new_pos = list(state['robot_pos'])
        if action == 0: new_pos[0] -= 1  # up
        elif action == 1: new_pos[0] += 1  # down  
        elif action == 2: new_pos[1] -= 1  # left
        elif action == 3: new_pos[1] += 1  # right
        
        # Bounds check + collision
        new_pos = [max(0,min(14,x)) for x in new_pos]
        if tuple(new_pos) != state['robot_pos'] and state['grid'][new_pos[0],new_pos[1]] == 1:
            reward -= 0.2  # wall collision
            state['collisions'] += 1
        else:
            state['robot_pos'] = tuple(new_pos)
        
        # Pick/Pack logic
        r,c = state['robot_pos']
        if action == 4 and state['grid'][r,c] == 3:  # PICK
            item_type = chr(ord('A') + np.where(state['grid']==3)[0][0] % 5)
            if item_type in state['inventory']:
                state['inventory'][item_type] += 1
                state['grid'][r,c] = 0
                reward += 1.0  # partial progress
        
        elif action == 5:  # PACK - check if can complete order
            for order in state['pending_orders'][:]:
                if all(state['inventory'][k] >= v for k,v in order.items()):
                    for k,v in order.items():
                        state['inventory'][k] -= v
                    state['pending_orders'].remove(order)
                    state['completed_orders'] += 1
                    reward += 2.0  # order completion
        
        state['time_step'] += 1
        self.current_step += 1
        
        # Time penalty + task completion
        reward -= 0.01  # time pressure
        if len(state['pending_orders']) == 0:
            done = True
            reward += 1.0
        
        if self.current_step >= self.max_steps:
            done = True
        
        # Update grid
        state['grid'][state['robot_pos'][0], state['robot_pos'][1]] = 2
        
        obs = self._format_observation(state)
        self.last_obs = obs
        
        info = {"task": self.task, "completed": state['completed_orders']}
        return obs, reward, done, info
    
    def _format_observation(self, state: WarehouseState) -> Dict[str, np.ndarray]:
        """Gym-compatible observation"""
        inv_array = np.array([state['inventory'].get(chr(ord('A')+i), 0) for i in range(5)])
        orders_array = np.zeros((3,3))
        for i, order in enumerate(state['pending_orders'][:3]):
            for j, (item,count) in enumerate(order.items()):
                if j < 3: orders_array[i,j] = count
        return {
            'grid': state['grid'],
            'robot_pos': np.array(state['robot_pos']),
            'inventory': inv_array,
            'pending_orders': orders_array,
            'time_step': np.array(state['time_step'])
        }
    
    def _parse_observation(self, obs: Dict) -> WarehouseState:
        """Reverse format for internal state"""
        # Simplified - reconstruct for demo
        return WarehouseState(grid=obs['grid'], robot_pos=tuple(obs['robot_pos']), 
                            inventory={}, pending_orders=[], completed_orders=0,
                            time_step=int(obs['time_step']), collisions=0, score=0.0)
    
    def score(self) -> float:
        """Safe grader - handles missing last_obs"""
        if self.task == "single_pick":
            if hasattr(self, 'last_obs') and self.last_obs is not None:
                try:
                    state = self._parse_observation(self.last_obs)
                    return 1.0 if len(state['pending_orders']) == 0 else 0.0
                except:
                    return 0.0
            return 0.0
        
    elif self.task == "multi_order":
        if hasattr(self, 'last_obs') and self.last_obs is not None:
            try:
                state = self._parse_observation(self.last_obs)
                return min(state['completed_orders'] / 1.0, 1.0)
            except:
                return 0.0
        return 0.0
        
    else:  # efficiency_challenge
        if hasattr(self, 'last_obs') and self.last_obs is not None:
            try:
                state = self._parse_observation(self.last_obs)
                return min(state['completed_orders'] / 2.5, 1.0)
            except:
                return 0.0
        return 0.0

# OpenEnv API compliance
def state() -> Dict:
    env = WarehouseEnv()
    return {"tasks": env.tasks if hasattr(env, 'tasks') else ["single_pick", "multi_order", "efficiency_challenge"]}

def reset(task: str) -> Dict:
    env = WarehouseEnv()
    return env.reset(task)

def step(action: int, task: str) -> Tuple[Dict, float, bool, Dict]:
    env = WarehouseEnv()
    obs = env.reset(task)
    env.last_obs = obs
    return env.step(action)
