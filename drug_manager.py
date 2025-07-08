# drug_manager.py
from drugs import Alcohol, Caffeine, Nicotine, Amphetamine # NEW: Import new drugs
from cell_model import NeuralCellData 

class DrugManager:
    """
    Manages the application of drug effects to the cell.
    """
    def __init__(self):
        self.available_drugs = {
            "None": None, # Option for no drug selected
            "Alcohol": Alcohol(),
            "Caffeine": Caffeine(),
            "Nicotine": Nicotine(), # NEW: Add Nicotine
            "Amphetamine": Amphetamine() # NEW: Add Amphetamine
        }
        self.active_drug_name = "None"
        self.drug_dosage = 0.0

    def set_active_drug(self, drug_name: str):
        """
        Sets the currently active drug.
        :param drug_name: The name of the drug (e.g., "Alcohol", "Caffeine", "None").
        """
        if drug_name in self.available_drugs:
            self.active_drug_name = drug_name
            self.drug_dosage = 0.0 # Reset dosage when switching drugs for a clean start
        else:
            print(f"Warning: Drug '{drug_name}' not recognized. Setting to 'None'.")
            self.active_drug_name = "None"
            self.drug_dosage = 0.0

    def get_active_drug_name(self) -> str:
        """Returns the name of the currently active drug."""
        return self.active_drug_name

    def set_drug_dosage(self, dosage: float):
        """
        Sets the dosage for the active drug.
        :param dosage: The dosage value (e.g., 0.0 to 1.0).
        """
        self.drug_dosage = max(0.0, min(1.0, dosage)) # Ensure dosage is between 0 and 1

    def apply_drug_effects(self, cell_data: NeuralCellData, time_step: float): # NEW: Pass time_step
        """
        Applies the effect of the active drug to the cell_data.
        This method is called *before* the general compound effects.
        """
        if self.active_drug_name != "None":
            drug_instance = self.available_drugs[self.active_drug_name]
            if drug_instance: # Check if it's not None
                drug_instance.apply_effect(cell_data, self.drug_dosage, time_step) # Pass time_step
