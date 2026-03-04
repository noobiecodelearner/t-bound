import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from dashboard.components.confidence_badge import render_confidence_badge


def render():
    st.markdown("## 🎯 Recommendation")

    project_id = st.session_state.get("project_id", "")
    api_get = st.session_state.get("_api_get")

    target_accuracy = st.slider("Target accuracy", 0.5, 0.99, 0.90, 0.01)

    data = api_get(f"/recommend?target_accuracy={target_accuracy}")

    confidence = data.get("confidence", "very_low")
    n_runs = data.get("runs_used", 0)
    message = data.get("message", "")

    # Warning / success banners
    if confidence in ("very_low", "low"):
        st.warning(f"⚠️ {message or 'Log more runs to improve confidence.'}")
    elif confidence == "high":
        st.success("✅ High confidence recommendation — your data dominates the prior.")

    st.divider()

    # Row 1 — primary
    st.markdown("#### Primary Recommendation")
    c1, c2, c3 = st.columns(3)
    optimal_n = data.get("optimal_n", 0)
    expected_acc = data.get("expected_accuracy", 0)
    ci_lower = data.get("ci_lower", 0)
    ci_upper = data.get("ci_upper", 0)

    c1.metric("Optimal N*", f"{optimal_n:,}")
    c2.metric("Expected Accuracy", f"{expected_acc:.4f}",
              delta=f"CI [{ci_lower:,.0f} – {ci_upper:,.0f}]")
    c3.metric("Confidence", confidence.replace("_", " ").upper())

    st.divider()

    # Row 2 — hyperparameters
    st.markdown("#### Optimal Hyperparameters")
    c1, c2, c3 = st.columns(3)
    c1.metric("Learning Rate lr*", f"{data.get('optimal_lr', 0):.6f}")
    c2.metric("Batch Size", f"{data.get('optimal_batch', 0):,}")
    d_frac = data.get("optimal_dataset_fraction")
    c3.metric("Dataset Fraction", f"{d_frac:.0%}" if d_frac else "N/A")

    st.divider()

    # Row 3 — savings
    st.markdown("#### Compute Savings")
    c1, c2, c3 = st.columns(3)
    c1.metric("Compute Saved", f"{data.get('compute_saved', 0):.1%}")
    c2.metric("Energy Saved", f"{data.get('energy_saved_kwh', 0):.4f} kWh")
    c3.metric("Carbon Saved", f"{data.get('carbon_saved_g', 0):.2f} g CO₂")

    st.divider()

    # Row 4 — metadata
    st.markdown("#### Metadata")
    c1, c2 = st.columns(2)
    prior_weight = data.get("prior_weight", 0)
    c1.metric("Runs Used", n_runs)
    c2.metric("Prior Weight", f"{prior_weight:.0%} prior / {1-prior_weight:.0%} your data")

    render_confidence_badge(confidence, n_runs)

    st.divider()

    # Training command
    st.markdown("#### Suggested Training Command")
    st.code(
        f"python train.py --params {optimal_n:,} --lr {data.get('optimal_lr', 0):.6f} "
        f"--batch {data.get('optimal_batch', 0)}",
        language="bash",
    )
