import streamlit as st


def _get_hint(confidence: str, n_runs: int) -> str:
    if confidence == "very_low":
        return f" — log {max(0, 3 - n_runs)} more for low confidence"
    elif confidence in ("low", "medium"):
        return f" — log {max(0, 6 - n_runs)} more for high confidence"
    return ""


def render_confidence_badge(confidence: str, n_runs: int) -> None:
    colors = {"very_low": "#ff4444", "low": "#ff8c00", "medium": "#ffd700", "high": "#22c55e"}
    labels = {"very_low": "VERY LOW", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}
    color = colors.get(confidence, "#888888")
    label = labels.get(confidence, confidence.upper())
    hint = _get_hint(confidence, n_runs)
    st.markdown(
        f'<span style="color:{color}; font-weight:bold; font-size:1.1em;">● {label}</span>'
        f'<span style="color:#888; font-size:0.85em;"> ({n_runs} runs{hint})</span>',
        unsafe_allow_html=True,
    )
