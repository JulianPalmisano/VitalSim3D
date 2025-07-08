# drugs.py

import numpy as np

# Define the effects for each drug.
# Each drug has a description and a list of effects.
# Each effect specifies:
# - target: The cell component affected (e.g., "membrane", "mitochondria", "overall_cell").
# - parameter: The specific property of the target (e.g., "irregularity_factor", "speed_factor", "color_mod", "health").
# - magnitude: The base strength of the effect. This will be scaled by the user's dose input (0.0 to 1.0).
# - type: How the magnitude is applied ("additive" or "multiplicative").
# - cumulative: If True, the effect builds up over time (e.g., health decay).
# - reset_on_deactivation: If True, this specific effect reverts when the drug is no longer active.

DRUG_EFFECTS = {
    "Alcohol": {
        "description": "Ethanol, commonly consumed in beverages. Primarily causes cellular stress, mitochondrial dysfunction, and general damage.",
        "effects": [
            # Membrane: Increased irregularity/wobble, slight transparency loss
            {"target": "membrane", "parameter": "irregularity_factor", "magnitude": 0.08, "type": "additive", "cumulative": False, "reset_on_deactivation": True},
            {"target": "membrane", "parameter": "color_mod", "magnitude": np.array([0.1, -0.2, -0.2, -0.05]), "type": "additive", "cumulative": False, "reset_on_deactivation": True}, # Shift towards red/brown, more transparent

            # Mitochondria: Slow down, become duller, decrease in active count over time
            {"target": "mitochondria", "parameter": "speed_factor", "magnitude": 0.6, "type": "multiplicative", "cumulative": False, "reset_on_deactivation": True}, # Slows down to 60% of base speed
            {"target": "mitochondria", "parameter": "color_mod", "magnitude": np.array([-0.2, -0.3, 0.1, 0.0]), "type": "additive", "cumulative": False, "reset_on_deactivation": True}, # Shift towards duller, slightly reddish/brownish
            {"target": "mitochondria", "parameter": "active_count_mod", "magnitude": -1, "type": "additive", "cumulative": True, "reset_on_deactivation": False}, # Cumulative loss of 1 active mito per "unit of toxin" per step

            # Cytoplasm: Slow down
            {"target": "cytoplasm", "parameter": "speed_factor", "magnitude": 0.7, "type": "multiplicative", "cumulative": False, "reset_on_deactivation": True}, # Slows down to 70% of base speed

            # Overall Cell: Cumulative health decay
            {"target": "overall_cell", "parameter": "health", "magnitude": -0.005, "type": "additive", "cumulative": True, "reset_on_deactivation": False} # Cumulative health decay per simulation step (scaled by dose)
        ]
    },
    "Caffeine": {
        "description": "A common stimulant. Primarily enhances metabolic activity and cellular responsiveness.",
        "effects": [
            # Membrane: Slight increase in vibrancy/activity
            {"target": "membrane", "parameter": "irregularity_factor", "magnitude": 0.02, "type": "additive", "cumulative": False, "reset_on_deactivation": True},
            {"target": "membrane", "parameter": "color_mod", "magnitude": np.array([0.05, 0.05, 0.05, 0.0]), "type": "additive", "cumulative": False, "reset_on_deactivation": True}, # Slightly brighter/more vibrant

            # Mitochondria: Speed up, become brighter
            {"target": "mitochondria", "parameter": "speed_factor", "magnitude": 1.3, "type": "multiplicative", "cumulative": False, "reset_on_deactivation": True}, # Speeds up to 130% of base speed
            {"target": "mitochondria", "parameter": "color_mod", "magnitude": np.array([0.1, 0.1, 0.0, 0.0]), "type": "additive", "cumulative": False, "reset_on_deactivation": True}, # Brighter yellow/orange

            # Cytoplasm: Speed up
            {"target": "cytoplasm", "parameter": "speed_factor", "magnitude": 1.2, "type": "multiplicative", "cumulative": False, "reset_on_deactivation": True}, # Speeds up to 120% of base speed

            # Overall Cell: Minor temporary health boost (no cumulative effect for now)
            # {"target": "overall_cell", "parameter": "health", "magnitude": 0.001, "type": "additive", "cumulative": False, "reset_on_deactivation": True} # Optional: small temporary health boost
        ]
    }
}