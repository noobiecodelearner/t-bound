"""
[dashboard/pages/scaling_curve). — live scaling curve visualization.

DAYANCH — implement this page.

What to do:
    Fetch curve data from GET /v1/projects/{project_id}/curve
    Render an interactive Plotly chart showing the fitted scaling law.

API response from /v1/projects/{project_id}/curve:
    {
        "params":           [N1, N2, N3, ...],        # observed param counts
        "accuracies":       [acc1, acc2, acc3, ...],  # observed val accuracies
        "curve_params":     [N_min, ..., N_max],      # smooth x-axis for curve
        "curve_mean":       [pred1, pred2, ...],      # fitted curve values
        "ci_lower":         [lb1, lb2, ...],          # 95% CI lower band
        "ci_upper":         [ub1, ub2, ...],          # 95% CI upper band
        "n_star":           342000,                    # predicted optimal N
        "target_accuracy":  0.85,                     # the target
        "alpha":            0.31,                     # fitted exponent
        "confidence":       "high",
        "n_runs":           8
    }

Chart elements (all on one Plotly figure):
    1. Scatter points — observed (params, val_accuracy) pairs
       Color: blue dots. Tooltip: show exact params and accuracy.

    2. Fitted curve — smooth line from curve_mean
       Color: blue line.

    3. CI band — shaded region between ci_lower and ci_upper
       Color: light blue, 20% opacity.

    4. N* vertical line — dashed vertical line at n_star
       Label: "N* = {n_star:,}"

    5. Target accuracy horizontal line — dashed horizontal line at target_accuracy
       Label: "τ = {target_accuracy}"

    6. Confidence badge — show in top-right corner of chart or below it
       very_low = red, low = orange, medium = yellow, high = green

Axes:
    x-axis: "Parameters (N)" — log scale
    y-axis: "Validation Accuracy" — linear scale, range [0, 1]

Below chart:
    Show: α = {alpha:.3f}
    Show: {n_runs} runs logged
    Show: confidence badge from dashboard/components/confidence_badge.py

If no data yet (0 runs):
    Show: "Log your first run with the SDK to see your scaling curve."
    Show a placeholder empty chart with axes but no data.

Notes:
    - Use plotly.graph_objects for full control
    - Make the chart responsive (use_container_width=True)
    - Auto-refresh every 30 seconds using st.empty() + time.sleep() in a loop
      OR add a manual "Refresh" button — auto-refresh is better UX
"""
