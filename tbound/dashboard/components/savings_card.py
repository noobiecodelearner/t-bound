"""
[dashboard/components/savings_card). — compute savings display component.

DAYANCH — implement this file.

What to do:
    Write a render_savings_cards(recommendation) function
    that displays compute, energy, and carbon savings as metric cards.

    recommendation: dict from /v1/recommend endpoint.

Display using st.columns(3) with st.metric() in each column:

    Column 1 — Compute Saved:
        Label: "Compute Saved"
        Value: "91.2%"
        Delta: "vs training at {max_observed_params:,} params"
        Delta color: "normal" (green arrow up)

    Column 2 — Energy Saved:
        Label: "Energy Saved"
        Value: "4.7 kWh"
        Delta: "{carbon_saved:.1f}g CO₂ avoided"

    Column 3 — Carbon Saved:
        Label: "Carbon Impact"
        Value: "{carbon_saved_g:.1f}g CO₂"
        Delta: "≈ {km_driven:.1f} km not driven"
        (Conversion: 1g CO₂ ≈ 0.006 km driven in an average car)

Below cards, show interpretation text:
    "Training at N*={optimal_n:,} instead of {naive_n:,} saves
     {compute_saved:.0%} of training compute."

Color the savings value green if > 50%, yellow if > 20%, grey if < 20%.

Example signature:
    def render_savings_cards(recommendation: dict) -> None:
        ...
"""
