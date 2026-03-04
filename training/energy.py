"""[energy). — energy and carbon estimation for [t-bound)."""


def estimate_energy_kwh(train_time_seconds: float,
                        gpu_wattage: float = 250.0) -> float:
    """Estimate GPU energy consumption in kWh."""
    hours = train_time_seconds / 3600.0
    return (gpu_wattage * hours) / 1000.0


def estimate_carbon_grams(energy_kwh: float,
                          carbon_intensity: float = 475.0) -> float:
    """
    Estimate CO2 emissions in grams.
    Default carbon intensity: 475 gCO2/kWh (global average grid, 2023).
    """
    return energy_kwh * carbon_intensity


def estimate_gpu_memory_mb(params: int, batch_size: int,
                           bytes_per_param: int = 4) -> float:
    """Rough GPU memory estimate in MB."""
    param_mb = (params * bytes_per_param) / (1024 ** 2)
    # activations: rough estimate, 3x params for gradients + optimizer state
    total_mb = param_mb * 4 + batch_size * 0.1
    return total_mb
