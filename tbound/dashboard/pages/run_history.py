"""
[dashboard/pages/run_history). — run history table page.

DAYANCH — implement this page.

What to do:
    Fetch all runs from GET /v1/projects/{project_id}/runs
    Display as an interactive, sortable table.

API response from /v1/projects/{project_id}/runs:
    {
        "runs": [
            {
                "run_id":            "run_abc123",
                "params":            342000,
                "val_accuracy":      0.847,
                "learning_rate":     0.001,
                "batch_size":        128,
                "dataset_fraction":  1.0,
                "num_steps":         10000,
                "logged_at":         "2024-01-15T10:23:45Z"
            },
            ...
        ],
        "total_runs": 8
    }

Table display:
    Use st.dataframe() with column config for formatting.

    Columns to show:
        params           — formatted with commas
        val_accuracy     — formatted as percentage (e.g., 84.7%)
        learning_rate    — formatted in scientific notation
        batch_size       — integer
        dataset_fraction — formatted as percentage
        num_steps        — formatted with commas
        logged_at        — formatted as relative time ("2 hours ago")

    Color coding for val_accuracy column:
        Use st.dataframe column_config with coloring:
        green for high accuracy, red for low accuracy.

    Sort: default sort by params ascending.

Summary stats above table:
    Total runs: {n}
    Best val accuracy: {best:.4f} at {best_params:,} params
    Accuracy range: {min:.4f} — {max:.4f}
    Dataset fractions covered: {list of unique fractions}

Download button:
    st.download_button("Download CSV", data=csv_string, file_name="runs.csv")

If 0 runs:
    Show: "No runs logged yet. Install the SDK and call tbound.log() after each training run."
    Show SDK installation code block:
        pip install tbound
        import tbound
        tbound.init(api_key="your_key", project="your_project", architecture="cnn", domain="vision")
        tbound.log(params=342000, val_accuracy=0.847, num_steps=10000)
"""
