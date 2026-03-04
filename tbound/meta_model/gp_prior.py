"""
[meta_model/gp_prior). — Gaussian Process meta-model for scaling exponent prediction.

V2 — NOT implemented at launch. Stub only.

When implemented, this module will:
    Train a GP that maps architecture feature vectors to scaling exponents.
    Used to improve cold start priors for novel architectures.

Planned interface:
    class GPPrior:
        def fit(self, X: np.ndarray, y: np.ndarray) -> None
            X shape: (n_architectures, n_features)
            y shape: (n_architectures,) — one exponent value per architecture
            Called after each new customer curve is absorbed.

        def predict(self, x: np.ndarray) -> Tuple[float, float]
            x shape: (n_features,)
            Returns: (alpha_mean, alpha_std)
            alpha_std directly becomes CI width for this customer's cold start.

        def update(self, x_new: np.ndarray, y_new: float) -> None
            Add one new data point and retrain.
            Called when a customer completes 10+ runs and their alpha is extracted.

Implementation notes (for when you build this):
    - Use sklearn.gaussian_process.GaussianProcessRegressor
    - Kernel: Matern(nu=2.5) + WhiteKernel() is a good starting point
    - Normalize X features before fitting
    - Normalize y (exponents) before fitting
    - GP saturates at ~100 training points — fine for launch
    - When you have 50+ customer curves, evaluate whether to switch to neural process
"""

class GPPrior:
    def fit(self, *args, **kwargs):
        raise NotImplementedError("GPPrior is a V2 feature.")

    def predict(self, *args, **kwargs):
        raise NotImplementedError("GPPrior is a V2 feature.")

    def update(self, *args, **kwargs):
        raise NotImplementedError("GPPrior is a V2 feature.")
