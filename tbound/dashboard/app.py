"""
[dashboard/app). — main Streamlit dashboard entry point.

DAYANCH — implement this file.

What to do:
    Build a Streamlit multi-page app.
    Use st.sidebar for navigation between pages.
    Each page lives in dashboard/pages/.

Structure:
    Sidebar:
        - [t-bound) logo/title
        - API key input (st.text_input, type="password")
        - Project ID input (st.text_input)
        - Page selector (st.radio or st.selectbox):
            "Scaling Curve"
            "Recommendation"
            "Run History"
            "Compute Savings" (only show if Chinchilla data exists)

    Main area:
        Load and render the selected page.

How to call the API from the dashboard:
    Use the requests library to call your own FastAPI server.
    Base URL: configurable via environment variable or st.secrets.
    Pass API key in header: {"X-TBound-Key": api_key}

    Example:
        import requests
        response = requests.get(
            f"{API_BASE_URL}/v1/projects/{project_id}/curve",
            headers={"X-TBound-Key": api_key}
        )

State management:
    Store api_key and project_id in st.session_state.
    All pages read from st.session_state — do not pass them as arguments.

To run:
    streamlit run dashboard/app.py

Dependencies:
    pip install streamlit plotly requests

Notes:
    - Show a friendly error if API key is invalid (401 response)
    - Show a "Log your first run" message if project has 0 runs
    - The dashboard is read-only — it only GETs data, never POSTs
      (customers log runs via the SDK, not the dashboard)
"""
