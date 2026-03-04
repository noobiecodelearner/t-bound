import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
from dashboard.components.confidence_badge import render_confidence_badge


def render():
    st.markdown("## 📈 Scaling Curve")

    project_id = st.session_state.get("project_id", "")
    api_get = st.session_state.get("_api_get")

    target_accuracy = st.slider("Target accuracy (τ)", 0.5, 0.99, 0.90, 0.01)

    data = api_get(f"/projects/{project_id}/curve?target_accuracy={target_accuracy}")

    n_runs = data.get("n_runs", 0)
    confidence = data.get("confidence", "very_low")
    alpha = data.get("alpha")

    if n_runs == 0:
        st.warning("No runs logged yet. Use the SDK to log your first training run.")
        st.code("""import tbound
tbound.init(api_key="tb_...", project="my-experiment")
tbound.log(params=342000, val_accuracy=0.847, num_steps=10000,
           learning_rate=0.001, batch_size=128)""")
        return

    params = data.get("params", [])
    accuracies = data.get("accuracies", [])
    curve_n = data.get("curve_params", [])
    curve_mean = data.get("curve_mean", [])
    ci_lower = data.get("ci_lower", [])
    ci_upper = data.get("ci_upper", [])
    n_star = data.get("n_star")

    fig = go.Figure()

    # CI band
    if ci_lower and ci_upper:
        fig.add_trace(go.Scatter(
            x=curve_n + curve_n[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            hoverinfo="skip",
        ))

    # Fitted curve
    if curve_n and curve_mean:
        fig.add_trace(go.Scatter(
            x=curve_n,
            y=curve_mean,
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Fitted curve",
        ))

    # Observed points
    fig.add_trace(go.Scatter(
        x=params,
        y=accuracies,
        mode="markers",
        marker=dict(color="#1d4ed8", size=10, symbol="circle"),
        name="Observed runs",
        hovertemplate="<b>%{x:,.0f}</b> params<br>Accuracy: %{y:.4f}<extra></extra>",
    ))

    # N* vertical line
    if n_star:
        fig.add_vline(
            x=n_star, line_dash="dash", line_color="#ef4444",
            annotation_text=f"N* = {n_star:,}",
            annotation_position="top right",
            annotation_font_color="#ef4444",
        )

    # Target accuracy horizontal line
    fig.add_hline(
        y=target_accuracy, line_dash="dash", line_color="#22c55e",
        annotation_text=f"τ = {target_accuracy}",
        annotation_position="bottom right",
        annotation_font_color="#22c55e",
    )

    fig.update_layout(
        xaxis=dict(title="Parameters (N)", type="log"),
        yaxis=dict(title="Validation Accuracy", range=[
            max(0, min(accuracies) - 0.05) if accuracies else 0, 1.0
        ]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis_gridcolor="#1e293b",
        yaxis_gridcolor="#1e293b",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Metadata row
    col1, col2, col3 = st.columns(3)
    with col1:
        if alpha is not None:
            st.metric("Scaling exponent α", f"{alpha:.3f}")
    with col2:
        st.metric("Runs logged", n_runs)
    with col3:
        if n_star:
            st.metric("Optimal N*", f"{n_star:,}")

    render_confidence_badge(confidence, n_runs)

    if st.button("🔄 Refresh"):
        st.rerun()
