import streamlit as st
import numpy as np

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'obs' not in st.session_state:
    st.session_state.obs = None
if 'done' not in st.session_state:
    st.session_state.done = False

st.set_page_config(page_title="Warehouse OpenEnv", layout="wide")

st.title("🏭 Warehouse Order Fulfillment")
st.markdown("**OpenEnv RL Environment** - 3 tasks Easy→Hard")

# Sidebar
st.sidebar.header("Controls")
task = st.sidebar.selectbox("Task", ["single_pick", "multi_order", "efficiency_challenge"])

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Environment")
    
    # SAFE RENDERING - No errors!
    if st.session_state.env is not None and st.session_state.obs is not None:
        grid = st.session_state.obs['grid']
        st.image(grid, caption="Grid (0=empty,1=wall,2=robot,3=item,5=shelf)", clamp=True)
        
        # SAFE METRICS
        try:
            score = st.session_state.env.score()
            st.metric("Score", f"{score:.2f}")
            st.metric("Inventory", st.session_state.obs['inventory'].tolist())
            st.metric("Time Step", st.session_state.obs['time_step'])
        except:
            st.metric("Score", "Reset to start")
    
    else:
        st.info("👆 Click **Reset** to start")

with col2:
    st.header("Actions")
    
    if st.session_state.env is not None:
        if st.session_state.done:
            st.success("✅ Episode Complete!")
            if st.button("🔄 New Episode"):
                st.session_state.env = None
                st.session_state.obs = None
                st.session_state.done = False
                st.rerun()
        else:
            action_names = ["0 ↑", "1 ↓", "2 ←", "3 →", "4 Pick", "5 Pack"]
            action = st.radio("Action:", action_names, horizontal=True)
            action_idx = int(action.split()[0])
            
            if st.button("▶️ Step", type="primary", use_container_width=True):
                obs, rew, done, info = st.session_state.env.step(action_idx)
                st.session_state.obs = obs
                st.session_state.done = done
                st.rerun()
    else:
        st.info("Reset first")

# Reset Button (Global)
if st.button("🔄 Reset Environment", use_container_width=True):
    from environment import WarehouseEnv
    st.session_state.env = WarehouseEnv()
    st.session_state.obs = st.session_state.env.reset(task)
    st.session_state.done = False
    st.rerun()

# OpenEnv API Docs
with st.expander("📖 OpenEnv API (Validator Tests)"):
    st.code("""
GET /state
→ {"tasks": ["single_pick", "multi_order", "efficiency_challenge"]}
POST /reset?task=single_pick
→ {"grid": array, "robot_pos": [x,y], ...}
POST /step?action=4
→ (obs, reward, done, info)
    """)

# Footer
st.markdown("---")
st.markdown("[inference.py baseline](https://huggingface.co/spaces/anicher/warehouse-openenv/raw/main/inference.py) | Fully offline | Docker-ready")
