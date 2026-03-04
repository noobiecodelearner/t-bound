"""
[meta_model/feature_extractor). — architecture feature extraction for GP meta-model.

V2 — NOT implemented at launch. Stub only.

When implemented, this module will:
    Extract a fixed-length feature vector from any neural network architecture.
    This vector is the input to the GP meta-model.

Features to extract (planned):
    parameter_efficiency:   params / FLOPs ratio
    depth:                  number of layers
    width:                  average hidden dimension
    has_skip_connections:   boolean (0 or 1)
    normalization_type:     0=none, 1=batch, 2=layer
    activation_type:        0=relu, 1=gelu, 2=tanh
    scaling_dimension:      0=width, 1=depth, 2=both
    domain_encoded:         one-hot [vision, nlp, tabular]
    dataset_size_log:       log10(dataset_size)
    num_classes_log:        log10(num_classes)

Planned interface:
    extract_features(model, domain, dataset_size, num_classes) -> np.ndarray
        Returns a 1D numpy array of shape (n_features,).
"""

def extract_features(*args, **kwargs):
    raise NotImplementedError(
        "feature_extractor is a V2 feature. "
        "Not available at launch. "
        "Use prior.csv for cold start estimates."
    )
