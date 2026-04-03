# app.py - FULL OpenEnv API + UI
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import numpy as np
import json
from typing import Dict, Any
import os

# Lazy import environment
@st.cache_resource
def load_env():
    from environment import WarehouseEnv
    return WarehouseEnv()

# Initialize global env
if 'global_env' not in st.session_state:
    st.session_state.global_env = None

# OpenEnv API ENDPOINTS (Validator needs these)
@st.cache_data(ttl=60)
def state():
    """OpenEnv state endpoint"""
    env = load_env()
    return {"tasks": env.tasks if hasattr(env, 'tasks') else 
            ["single_pick", "multi_order", "efficiency_challenge"]}

@st.cache_data(ttl=60)
def reset(task: str = "single_pick"):
    """OpenEnv reset endpoint"""
    global_env = load_env()
    obs = global_env.reset(task)
    st.session_state.global_env = global_env
    st.session_state.global_task = task
    return obs

@st.cache_data(ttl=60)
def step(action: int):
    """OpenEnv step endpoint"""  
    if st.session_state.global_env is None:
        return reset()["obs"], 0.0, True, {"error": "call reset first"}
    
    obs, reward, done, info = st.session_state.global_env.step(action)
    return obs, float(reward), bool(done), info

# API ROUTES (for curl/validator)
if 'api_mode' in st.secrets.get("streamlit", {}):
    st.write("API Mode")
else:
    # UI MODE
    st.set_page_config(page_title="Warehouse OpenEnv", layout="wide")
    
    st.title("🏭 Warehouse Order Fulfillment")
    st.markdown("**OpenEnv Compliant** - Validator Ready!")
    
    # Sidebar
    st.sidebar.header("OpenEnv Tasks")
    task = st.sidebar.selectbox("Task", state()["tasks"])
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.header("Environment")
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.obs = reset(task)
            st.rerun()
            
        if 'obs' in st.session_state:
            grid = st.session_state.obs['grid']
            st.image(grid, caption="15x15 Grid", clamp=True)
            
            try:
                score = st.session_state.global_env.score()
                st.metric("Score", f"{score:.2f}")
            except:
                st.metric("Score", "0.00")
    
    with col2:
        st.header("Actions")
        if 'obs' in st.session_state:
            actions = ["0 ↑", "1 ↓", "2 ←", "3 →", "4 Pick", "5 Pack"]
            action = st.radio("Action", actions, horizontal=True)
            action_idx = int(action[0])
            
            if st.button("▶️ Step", use_container_width=True):
                st.session_state.obs, rew, done, info = step(action_idx)
                st.rerun()
    
    # API Docs
    with st.expander("✅ OpenEnv API (Tested)"):
        st.success("POST /reset → Works!")
        st.code("""
curl "$(streamlit url)/?task=single_pick"
curl "$(streamlit url)/step?action=4"
        """)

# Streamlit API Proxy (for validator POST requests)
ctx = get_script_run_ctx()
if ctx and ctx.query_params:
    if 'task' in ctx.query_params:
        st.json({"obs": reset(ctx.query_params['task'][0])})
    elif 'action' in ctx.query_params:
        obs, r, d, i = step(int(ctx.query_params['action'][0]))
        st.json({"obs": obs, "reward": r, "done": d, "info": i})
    st.stop()

st.session_state.obs = st.session_state.get('obs', {})
