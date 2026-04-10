import streamlit as st
import numpy as np

# SAFE IMPORT with error handling
try:
    from environment import WarehouseEnv
    ENV_AVAILABLE = True
except ImportError as e:
    st.error(f"Environment import failed: {e}")
    ENV_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("🏭 Warehouse OpenEnv")

if not ENV_AVAILABLE:
    st.error("❌ Fix environment.py import")
    st.info("Upload environment.py first!")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        task = st.selectbox("Task", ["single_pick", "multi_order", "efficiency_challenge"])
        if st.button("🔄 Reset"):
            env = WarehouseEnv()
            obs = env.reset(task)
            st.session_state.obs = obs
            st.session_state.env = env
            st.rerun()
    
    with col2:
        if 'obs' in st.session_state:
            st.metric("Score", st.session_state.env.score())
            st.image(st.session_state.obs['grid'])
    
    st.code("curl https://YOUR_SPACE/reset?task=single_pick")
