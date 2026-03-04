import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="[t-bound)", page_icon="📈", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## [t-bound)")
    st.markdown("*What if you had to train less?*")
    st.divider()

    api_key = st.text_input("API Key", type="password", key="api_key",
                             placeholder="tb_...")
    project_id = st.text_input("Project ID", key="project_id",
                                placeholder="my-experiment")
    st.divider()

    page = st.radio("Navigate", [
        "📈 Scaling Curve",
        "🎯 Recommendation",
        "📋 Run History",
        "💾 Compute Savings",
    ])

if not api_key or not project_id:
    st.markdown("## Welcome to [t-bound)")
    st.info("Enter your **API Key** and **Project ID** in the sidebar to get started.")
    st.markdown("""
    **Quick start:**
    ```python
    import tbound
    tbound.init(api_key="tb_...", project="my-experiment")
    tbound.log(params=342000, val_accuracy=0.847, num_steps=10000,
               learning_rate=0.001, batch_size=128)
    rec = tbound.recommend(target_accuracy=0.90)
    print(rec.optimal_n)
    ```
    """)
    st.stop()


def api_get(path):
    import requests
    try:
        r = requests.get(
            f"http://localhost:8000/v1{path}",
            headers={"X-TBound-Key": st.session_state.api_key},
            timeout=10,
        )
        if r.status_code == 401:
            st.error("❌ Invalid API key.")
            st.stop()
        return r.json()
    except Exception as e:
        st.error(f"❌ Could not reach API: {e}")
        st.stop()


# Store api_get in session so pages can import it
st.session_state["_api_get"] = api_get

if page == "📈 Scaling Curve":
    from dashboard._pages.scaling_curve import render
    render()
elif page == "🎯 Recommendation":
    from dashboard._pages.recommendation import render
    render()
elif page == "📋 Run History":
    from dashboard._pages.run_history import render
    render()
elif page == "💾 Compute Savings":
    from dashboard._pages.compute_savings import render
    render()
