import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd


def render():
    st.markdown("## 📋 Run History")

    project_id = st.session_state.get("project_id", "")
    api_get = st.session_state.get("_api_get")

    data = api_get(f"/projects/{project_id}/runs")
    runs = data.get("runs", [])
    total = data.get("total_runs", 0)

    if total == 0:
        st.info("No runs logged yet.")
        st.markdown("**Install the SDK and log your first run:**")
        st.code("""pip install httpx
import tbound
tbound.init(api_key="tb_...", project="my-experiment",
            architecture="cnn", domain="vision",
            api_url="http://localhost:8000")
tbound.log(params=342000, val_accuracy=0.847, num_steps=10000,
           learning_rate=0.001, batch_size=128)""")
        return

    df = pd.DataFrame(runs)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs", total)
    best_row = df.loc[df["val_accuracy"].idxmax()]
    c2.metric("Best Accuracy", f"{best_row['val_accuracy']:.4f}")
    c3.metric("Best at N", f"{int(best_row['params']):,}")
    c4.metric("Accuracy Range",
              f"{df['val_accuracy'].min():.3f} – {df['val_accuracy'].max():.3f}")

    st.divider()

    # Format for display
    display_df = df[["params", "val_accuracy", "learning_rate",
                      "batch_size", "dataset_fraction", "num_steps", "logged_at"]].copy()
    display_df = display_df.sort_values("params")
    display_df["params"] = display_df["params"].apply(lambda x: f"{int(x):,}")
    display_df["val_accuracy"] = display_df["val_accuracy"].apply(lambda x: f"{x:.4f}")
    display_df["learning_rate"] = display_df["learning_rate"].apply(lambda x: f"{x:.2e}")
    display_df["dataset_fraction"] = display_df["dataset_fraction"].apply(lambda x: f"{x:.0%}")
    display_df["num_steps"] = display_df["num_steps"].apply(lambda x: f"{int(x):,}")
    display_df.columns = ["Params (N)", "Val Accuracy", "Learning Rate",
                          "Batch Size", "Dataset Fraction", "Steps", "Logged At"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", data=csv,
                       file_name=f"{project_id}_runs.csv", mime="text/csv")

    if st.button("🔄 Refresh"):
        st.rerun()
