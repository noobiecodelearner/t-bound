"""
[dashboard/components/curve_plot). — reusable scaling curve plot component.

DAYANCH — implement this file.

What to do:
    Write a render_curve_plot(curve_data) function that produces
    a Plotly figure. Used by scaling_curve.py page.

    curve_data is the dict returned by GET /v1/projects/{project_id}/curve.

The plot must show:
    1. Observed data points (scatter)
    2. Fitted curve (line)
    3. 95% CI band (shaded area)
    4. N* vertical marker (dashed line)
    5. Target accuracy horizontal marker (dashed line)

Return a plotly.graph_objects.Figure, not st.plotly_chart.
Let the calling page render it with st.plotly_chart(fig, use_container_width=True).

This separation makes the component testable without Streamlit.

Styling guidelines:
    Background: white or very light grey
    Observed points: filled circles, size 8, color #3b82f6 (blue)
    Fitted curve: solid line, weight 2, color #3b82f6
    CI band: fill between ci_lower and ci_upper, color #3b82f6 at 15% opacity
    N* line: dashed, color #ef4444 (red), annotated
    Target line: dashed, color #22c55e (green), annotated
    Grid: light grey, subtle
    Font: clean sans-serif

Example signature:
    def render_curve_plot(curve_data: dict) -> go.Figure:
        ...
        return fig
"""
