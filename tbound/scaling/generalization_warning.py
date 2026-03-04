"""[generalization_warning). — overfitting risk detector for [t-bound)."""


class GeneralizationWarningDetector:
    """
    Classifies generalization gap into LOW / MEDIUM / HIGH risk.

    generalization_gap = train_accuracy - val_accuracy
    """

    def __init__(self, low_threshold: float = 0.05,
                 high_threshold: float = 0.15):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def classify(self, train_accuracy: float,
                 val_accuracy: float) -> str:
        gap = train_accuracy - val_accuracy
        if gap < self.low_threshold:
            return "LOW"
        elif gap < self.high_threshold:
            return "MEDIUM"
        else:
            return "HIGH"

    def gap(self, train_accuracy: float, val_accuracy: float) -> float:
        return round(train_accuracy - val_accuracy, 6)
