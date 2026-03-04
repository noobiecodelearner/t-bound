"""
[dashboard/pages/compute_savings). — Chinchilla frontier visualization.

DAYANCH — implement this page.

What to do:
    Only show this page if the customer has run the N × D grid
    (i.e., fits table has a delta value for this project).

    Fetch frontier data from GET /v1/projects/{project_id}/frontier
    Display the (N, D) accuracy surface and optimal frontier.

API response from /v1/projects/{project_id}/frontier:
    {
        "has_frontier": true/false,
        "n_values":     [N1, N2, ...],          # model sizes on grid
        "d_values":     [D1, D2, ...],          # dataset fractions on grid
        "accuracy_surface": [[acc, ...], ...],  # 2D accuracy surface
        "frontier_n":   [N_f1, N_f2, ...],      # optimal frontier N values
        "frontier_d":   [D_f1, D_f2, ...],      # optimal frontier D values
        "naive_n":      2000000,                # naive approach (large model, full data)
        "naive_d":      1.0,
        "optimal_n":    342000,                 # Chinchilla optimal at compute budget
        "optimal_d":    0.8,
        "compute_saved": 0.83,
        "alpha":        0.31,
        "delta":        0.28,
    }

If has_frontier is false:
    Show: "Run the N × D grid sweep to unlock the Chinchilla frontier."
    Show instructions for how to run it.

If has_frontier is true, show three sections:

Section 1 — Accuracy surface heatmap:
    Plotly heatmap of accuracy_surface.
    X-axis: dataset fraction (D values)
    Y-axis: model parameters (N values, log scale)
    Color: accuracy (0 to 1, viridis colorscale)
    Mark the optimal point (N*, D*) with a star marker.
    Mark the naive point (large model, full data) with an X marker.

Section 2 — Optimal frontier line:
    Plotly scatter/line showing the compute-optimal frontier.
    X-axis: model parameters (log scale)
    Y-axis: dataset fraction
    Annotate the frontier with accuracy values along it.

Section 3 — Savings summary:
    Three metric cards:
        Compute saved: {compute_saved:.1%}
        Optimal model size: {optimal_n:,} params
        Optimal dataset: {optimal_d:.0%} of your data

    Interpretation text:
        "Instead of training a {naive_n:,} parameter model on 100% of your data,
         train a {optimal_n:,} parameter model on {optimal_d:.0%} of your data.
         Same target accuracy. {compute_saved:.0%} less compute."

Compute budget input:
    st.slider for compute_budget_hours (0.5 to 100 hours)
    When changed, call GET /v1/recommend with compute_budget_hours
    and update the optimal point on the chart.

Notes:
    - The heatmap is the most impressive visual in the product — make it beautiful
    - Use plotly.graph_objects for full control
    - This page is the premium differentiator — emphasize savings clearly
"""
