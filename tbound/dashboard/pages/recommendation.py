"""
[dashboard/pages/recommendation). — recommendation display page.

DAYANCH — implement this page.

What to do:
    Fetch recommendation from GET /v1/recommend?project_id=X&target_accuracy=0.85
    Display it as a clean, scannable card layout.

Inputs (in sidebar or top of page):
    target_accuracy: st.slider(0.5, 1.0, 0.85, 0.01)
    compute_budget_hours: st.number_input (optional, for Chinchilla path)

Display layout:
    Row 1 — primary recommendation:
        [optimal N*).     342,000 parameters
        [expected accuracy). 0.847  [0.831 — 0.863]
        [confidence).     HIGH

    Row 2 — hyperparameters:
        [optimal lr*).    0.001
        [optimal batch*). 128
        [optimal D*).     80% of dataset   ← only if Chinchilla path

    Row 3 — savings:
        [compute saved).  91.2%
        [energy saved).   4.7 kWh
        [carbon saved).   47.3g CO₂

    Row 4 — metadata:
        Runs used: 8
        Prior weight: 3% prior / 97% your data
        Last updated: 2 minutes ago

All values in metric cards using st.metric().
CI shown as delta below the accuracy value.

Message display:
    If confidence is very_low or low:
        Show yellow warning box: "Recommendation based on limited data.
        Log more runs to improve confidence."

    If confidence is high:
        Show green success box: "High confidence recommendation."

Copy button:
    Show a code block with the recommended training command:
        python train.py --params 342000 --lr 0.001 --batch 128

Notes:
    - Update automatically when target_accuracy slider changes (Streamlit reruns on widget change)
    - Use st.columns(3) for the card layout
    - Format large numbers with commas: f"{n_star:,}"
    - Round accuracy to 4 decimal places
    - Round lr to 6 decimal places
"""
