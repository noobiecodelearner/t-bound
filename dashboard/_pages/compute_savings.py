import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render():
    st.markdown("## 💾 Compute Savings")

    project_id = st.session_state.get("project_id", "")
    api_get = st.session_state.get("_api_get")

    # Check if we have a fit with delta
    runs_data = api_get(f"/projects/{project_id}/runs")
    runs = runs_data.get("runs", [])

    if len(runs) < 3:
        st.info("Log at least 3 runs at different model sizes to unlock compute savings analysis.")
        st.markdown("""
        **To get the most out of this page**, log runs across a range of model sizes:
        ```python
        for params in [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
            # train model with `params` parameters
            tbound.log(params=params, val_accuracy=val_acc, ...)
        ```
        """)
        return

    target_accuracy = st.slider("Target accuracy", 0.5, 0.99, 0.90, 0.01)
    rec = api_get(f"/recommend?target_accuracy={target_accuracy}")

    optimal_n = rec.get("optimal_n", 0)
    compute_saved = rec.get("compute_saved", 0)
    energy_saved = rec.get("energy_saved_kwh", 0)
    carbon_saved = rec.get("carbon_saved_g", 0)
    confidence = rec.get("confidence", "very_low")

    # Get curve data for visualization
    curve_data = api_get(f"/projects/{project_id}/curve?target_accuracy={target_accuracy}")
    params_obs = curve_data.get("params", [])
    accs_obs = curve_data.get("accuracies", [])
    curve_n = curve_data.get("curve_params", [])
    curve_mean = curve_data.get("curve_mean", [])
    alpha = curve_data.get("alpha", 0.30)

    max_observed = max(params_obs) if params_obs else optimal_n

    # ── Savings summary ───────────────────────────────────────────────────────
    st.markdown("### Savings vs Naive Approach")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compute Saved", f"{compute_saved:.1%}")
    c2.metric("Optimal N*", f"{optimal_n:,}")
    c3.metric("Energy Saved", f"{energy_saved:.4f} kWh")
    c4.metric("Carbon Saved", f"{carbon_saved:.2f} g CO₂")

    if max_observed > optimal_n:
        st.success(
            f"Instead of training a **{max_observed:,}** param model, "
            f"train a **{optimal_n:,}** param model. "
            f"Same target accuracy. **{compute_saved:.0%} less compute.**"
        )
    else:
        st.info(
            f"Your current largest model ({max_observed:,} params) is smaller than the "
            f"recommended N* ({optimal_n:,}). Log runs with more params to see savings."
        )

    st.divider()

    # ── Scaling curve with savings highlighted ────────────────────────────────
    st.markdown("### Scaling Curve — Where to Stop")

    fig = go.Figure()

    # CI band
    ci_lower = curve_data.get("ci_lower", [])
    ci_upper = curve_data.get("ci_upper", [])
    if ci_lower and ci_upper:
        fig.add_trace(go.Scatter(
            x=curve_n + curve_n[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            hoverinfo="skip",
        ))

    # Fitted curve
    if curve_n and curve_mean:
        fig.add_trace(go.Scatter(
            x=curve_n, y=curve_mean,
            mode="lines", line=dict(color="#3b82f6", width=2),
            name="Scaling law",
        ))

    # Shade the "wasted compute" region (right of N*)
    if optimal_n and curve_n:
        waste_x = [n for n in curve_n if n >= optimal_n]
        waste_y = [curve_mean[i] for i, n in enumerate(curve_n) if n >= optimal_n]
        if waste_x:
            fig.add_trace(go.Scatter(
                x=waste_x + waste_x[::-1],
                y=waste_y + [target_accuracy] * len(waste_y),
                fill="toself",
                fillcolor="rgba(239, 68, 68, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Wasted compute",
                hoverinfo="skip",
            ))

    # Observed points
    fig.add_trace(go.Scatter(
        x=params_obs, y=accs_obs,
        mode="markers",
        marker=dict(color="#1d4ed8", size=10),
        name="Observed runs",
        hovertemplate="<b>%{x:,.0f}</b> params<br>Accuracy: %{y:.4f}<extra></extra>",
    ))

    # N* and target lines
    if optimal_n:
        fig.add_vline(x=optimal_n, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"N* = {optimal_n:,}",
                      annotation_font_color="#ef4444")
    fig.add_hline(y=target_accuracy, line_dash="dash", line_color="#22c55e",
                  annotation_text=f"τ = {target_accuracy}",
                  annotation_font_color="#22c55e")

    fig.update_layout(
        xaxis=dict(title="Parameters (N)", type="log"),
        yaxis=dict(title="Validation Accuracy",
                   range=[max(0, min(accs_obs) - 0.05) if accs_obs else 0, 1.0]),
        height=420,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        xaxis_gridcolor="#1e293b",
        yaxis_gridcolor="#1e293b",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"α = {alpha:.3f} · Confidence: {confidence.replace('_', ' ').upper()}")

    if st.button("🔄 Refresh"):
        st.rerun()
