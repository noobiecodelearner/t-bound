"""
[dashboard/components/confidence_badge). — confidence level display component.

DAYANCH — implement this file.

What to do:
    Write a render_confidence_badge(confidence, n_runs) function
    that displays a colored badge with the confidence level.

    confidence: "very_low" | "low" | "medium" | "high"
    n_runs: int — number of runs logged

Color mapping:
    very_low → red     (#ff4444)
    low      → orange  (#ff8c00)
    medium   → yellow  (#ffd700)
    high     → green   (#22c55e)

Display:
    Use st.markdown with HTML for colored badge.
    Show: ● HIGH (8 runs)
    Or:   ● LOW (2 runs — log 4 more for medium confidence)

Example implementation:
    def render_confidence_badge(confidence: str, n_runs: int) -> None:
        colors = {"very_low": "#ff4444", "low": "#ff8c00",
                  "medium": "#ffd700", "high": "#22c55e"}
        labels = {"very_low": "VERY LOW", "low": "LOW",
                  "medium": "MEDIUM", "high": "HIGH"}
        color = colors.get(confidence, "#888888")
        label = labels.get(confidence, confidence.upper())
        hint = _get_hint(confidence, n_runs)
        st.markdown(
            f'<span style="color:{color}; font-weight:bold;">● {label}</span> '
            f'<span style="color:#888; font-size:0.85em;">({n_runs} runs{hint})</span>',
            unsafe_allow_html=True
        )

    def _get_hint(confidence: str, n_runs: int) -> str:
        if confidence == "very_low":
            return f" — log {3 - n_runs} more for low confidence"
        elif confidence == "low":
            return f" — log {6 - n_runs} more for high confidence"
        elif confidence == "medium":
            return f" — log {6 - n_runs} more for high confidence"
        return ""
"""
