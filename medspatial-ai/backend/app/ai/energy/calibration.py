"""Bounded energy calibration."""

import math


def energy_to_plausibility(energy: float, temperature: float = 1.0, offset: float = 1.0) -> float:
    temperature = max(temperature, 1e-6)
    return 1.0 / (1.0 + math.exp((energy - offset) / temperature))
