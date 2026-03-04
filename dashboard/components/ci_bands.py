"""
[dashboard/components/ci_bands). — CI band rendering utilities.

DAYANCH — implement this file.

What to do:
    Write helper functions for adding CI bands to Plotly figures.
    Used by curve_plot.py and compute_savings.py.

Functions to implement:

    add_ci_band(fig, x, ci_lower, ci_upper, color="#3b82f6", opacity=0.15, name="95% CI")
        Adds a shaded CI band to an existing Plotly figure.
        Uses go.Scatter with fill='tonexty' pattern:
            - First trace: ci_upper (line, no marker, showlegend=False)
            - Second trace: ci_lower with fill='tonexty' to shade between them
        Returns the modified fig.

    add_horizontal_ci(fig, y_mean, y_lower, y_upper, x_min, x_max,
                      color="#3b82f6", name="CI")
        Adds a horizontal CI band at a fixed y value.
        Used to show the CI on a single prediction point.

    format_ci_string(lower, upper, decimals=4)
        Returns a formatted string: "[0.8312, 0.8634]"
        Used in tooltips and metric displays.

Notes:
    - All functions take and return go.Figure for composability
    - Keep opacity consistent: 0.15 for bands, 1.0 for lines
    - Test these functions independently before integrating into pages
"""
