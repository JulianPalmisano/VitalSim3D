# drug_manager.py

import numpy as np
from drugs import DRUG_EFFECTS

class DrugManager:
    def __init__(self):
        self.drug_definitions = DRUG_EFFECTS
        # active_drugs will store the current state for each drug:
        # {
        #    "DrugName": {
        #        "dose": float (0.0 to 1.0),
        #        "is_active": bool,
        #        "cumulative_trackers": { "parameter_name": current_cumulative_value }
        #    }
        # }
        self.active_drugs_state = {}

        # Initialize state for all defined drugs
        for drug_name in self.drug_definitions:
            self.active_drugs_state[drug_name] = {
                "dose": 0.0,
                "is_active": False,
                "cumulative_trackers": {}
            }
            # Initialize cumulative trackers based on drug definitions
            for effect in self.drug_definitions[drug_name]["effects"]:
                if effect.get("cumulative", False):
                    # For color_mod, cumulative value should also be np.array
                    if effect["parameter"].endswith("color_mod"):
                        self.active_drugs_state[drug_name]["cumulative_trackers"][effect["parameter"]] = np.array([0.0, 0.0, 0.0, 0.0])
                    else:
                        self.active_drugs_state[drug_name]["cumulative_trackers"][effect["parameter"]] = 0.0

    def set_drug_dose(self, drug_name, dose):
        """Sets the dose for a specific drug (0.0 to 1.0)."""
        if drug_name in self.active_drugs_state:
            self.active_drugs_state[drug_name]["dose"] = np.clip(dose, 0.0, 1.0)
        else:
            print(f"Warning: Drug '{drug_name}' not defined.")

    def toggle_drug_active(self, drug_name, is_active):
        """Activates or deactivates a drug. Handles reset_on_deactivation effects."""
        if drug_name in self.active_drugs_state:
            prev_active_state = self.active_drugs_state[drug_name]["is_active"]
            self.active_drugs_state[drug_name]["is_active"] = is_active

            # If drug is being deactivated, reset effects marked for reset_on_deactivation
            if prev_active_state and not is_active:
                # For cumulative effects that are NOT reset_on_deactivation, they remain.
                # For non-cumulative effects that ARE reset_on_deactivation, their contribution stops.
                # The 'get_current_net_effects' will naturally handle this by not including them.
                pass # No explicit reset needed here, as get_current_net_effects computes live effects.
                     # Cumulative effects not marked for reset_on_deactivation continue to accumulate.
                     # This manager only tracks their cumulative *value*, not resets it.
                     # The cell_model will use these values.

        else:
            print(f"Warning: Drug '{drug_name}' not defined.")

    def get_current_net_effects(self, dt=1.0):
        """
        Calculates the combined effects of all active drugs.
        dt is the time step for cumulative effects.
        """
        # Initialize an empty dictionary to accumulate all effects
        # For multiplicative effects, the base is 1.0 (no change).
        # For additive effects, the base is 0.0 (no change).
        net_effects = {
            "membrane_irregularity_factor": 0.0,
            "membrane_color_mod": np.array([0.0, 0.0, 0.0, 0.0]),
            "mitochondria_speed_factor": 1.0,  # Base for multiplicative
            "mitochondria_color_mod": np.array([0.0, 0.0, 0.0, 0.0]),
            "mitochondria_active_count_mod": 0, # Additive
            "cytoplasm_speed_factor": 1.0,     # Base for multiplicative
            "cytoplasm_color_mod": np.array([0.0, 0.0, 0.0, 0.0]),
            "cytoplasm_size_mod": 0.0,
            "nucleus_color_mod": np.array([0.0, 0.0, 0.0, 0.0]),
            "overall_cell_health_change": 0.0
        }

        for drug_name, drug_state in self.active_drugs_state.items():
            if drug_state["is_active"] and drug_state["dose"] > 0:
                definition = self.drug_definitions[drug_name]
                dose_factor = drug_state["dose"] # Dose is already 0.0 to 1.0

                for effect in definition["effects"]:
                    parameter = effect["parameter"]
                    magnitude = effect["magnitude"] * dose_factor # Scale magnitude by current dose
                    effect_type = effect["type"]
                    is_cumulative = effect.get("cumulative", False)

                    if is_cumulative:
                        # Update cumulative tracker for this specific drug and parameter
                        if parameter.endswith("color_mod"):
                             drug_state["cumulative_trackers"][parameter] += magnitude * dt
                        else:
                            drug_state["cumulative_trackers"][parameter] += magnitude * dt # Apply effect over time
                        
                        # Add cumulative value to net effects
                        if effect_type == "additive":
                            net_effects[parameter] += drug_state["cumulative_trackers"][parameter]
                        elif effect_type == "multiplicative":
                            net_effects[parameter] *= (1 + drug_state["cumulative_trackers"][parameter]) # Assuming additive change to a multiplier
                    else:
                        # Apply non-cumulative effects directly
                        if effect_type == "additive":
                            if parameter.endswith("color_mod"):
                                net_effects[parameter] += magnitude # Add array directly
                            else:
                                net_effects[parameter] += magnitude
                        elif effect_type == "multiplicative":
                            net_effects[parameter] *= magnitude # Apply factor directly

        return net_effects

    def reset_all_drugs(self):
        """Resets all drug doses, active states, and cumulative trackers."""
        for drug_name in self.active_drugs_state:
            self.active_drugs_state[drug_name]["dose"] = 0.0
            self.active_drugs_state[drug_name]["is_active"] = False
            # Reset cumulative trackers for each drug
            for param in self.active_drugs_state[drug_name]["cumulative_trackers"]:
                if param.endswith("color_mod"):
                    self.active_drugs_state[drug_name]["cumulative_trackers"][param] = np.array([0.0, 0.0, 0.0, 0.0])
                else:
                    self.active_drugs_state[drug_name]["cumulative_trackers"][param] = 0.0