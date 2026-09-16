"""Plausible dam-break scenario generation for the dashboard prototype."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "dam_height_m",
    "reservoir_volume_mcm",
    "water_level_m",
    "breach_width_m",
    "breach_time_min",
]


def generate_scenarios(base: dict, n_scenarios: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return diverse, physically bounded variations around a base dam condition.

    Values are intended for an early-stage demonstration, not site-specific engineering
    analysis. ``base`` uses the feature names listed in ``FEATURE_COLUMNS``.
    """
    rng = np.random.default_rng(seed)
    dam_height = float(base["dam_height_m"])
    volume = float(base["reservoir_volume_mcm"])
    water_level = float(base["water_level_m"])
    breach_width = float(base["breach_width_m"])
    breach_time = float(base["breach_time_min"])

    # Evenly spaced severity plus small variation makes the table visibly diverse.
    severity = np.linspace(-1.0, 1.0, n_scenarios)
    noise = rng.normal(0, 0.08, size=(n_scenarios, 5))
    frame = pd.DataFrame(
        {
            "dam_height_m": np.clip(dam_height * (1 + 0.08 * severity + noise[:, 0]), 5, 300),
            "reservoir_volume_mcm": np.clip(volume * (1 + 0.30 * severity + noise[:, 1]), 1, 10000),
            "water_level_m": np.clip(water_level * (1 + 0.15 * severity + noise[:, 2]), 1, dam_height * 1.05),
            "breach_width_m": np.clip(breach_width * (1 + 0.45 * severity + noise[:, 3]), 2, 500),
            "breach_time_min": np.clip(breach_time * (1 - 0.35 * severity + noise[:, 4]), 2, 360),
        }
    )
    frame["water_level_m"] = np.minimum(frame["water_level_m"], frame["dam_height_m"] * 0.99)
    frame.insert(0, "scenario", np.arange(1, n_scenarios + 1))
    return frame.round(2)
