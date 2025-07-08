from abc import ABC, abstractmethod
import numpy as np # Import numpy as it's used in NeuralCellData related calculations

# Assuming NeuralCellData is available from cell_model.py
# If cell_model.py is named just 'cell_model', then:
# from cell_model import NeuralCellData 
# If it's `cell_model.py` and the class is `NeuralCellData` as provided
from cell_model import NeuralCellData


class Drug(ABC):
    """
    Abstract base class for all drugs.
    Defines the interface for applying drug effects.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply_effect(self, cell_data: NeuralCellData, dosage: float):
        """
        Applies the drug's effect to the cell components based on dosage.
        This method will modify the properties within the cell_data object directly.
        :param cell_data: The NeuralCellData object to apply effects to.
        :param dosage: The intensity of the drug (e.g., 0.0 to 1.0).
        """
        pass

class Alcohol(Drug):
    """
    Represents the effects of Alcohol on a neural cell.
    Primary effects:
    - Decreases mitochondrial metabolic efficiency.
    - Increases membrane irregularity/permeability (simulated by increased irregularity).
    - Decreases particle speed (representing slowed transport/cytoplasmic flow).
    - Can dull nucleus activity (represented by a slight decrease in the nucleus activity level).
    """
    def __init__(self):
        super().__init__("Alcohol")

    def apply_effect(self, cell_data: NeuralCellData, dosage: float):
        """
        Applies alcohol's effects based on dosage.
        Dosage: 0.0 (no effect) to 1.0 (max effect).
        """
        dosage = max(0.0, min(1.0, dosage)) # Ensure dosage is within [0, 1]

        # Metabolic Efficiency (Mitochondria): Alcohol typically slows metabolism
        # cell_data.mitochondria_color (initially (1.0, 0.7, 0.0, 0.9)) -> more red/brown
        # This will be handled by existing metabolic_level logic in simulation_logic
        # but we can make it harder for metabolic stimulator to work.
        # Direct effect on overall cell "health" related to metabolism:
        # We can simulate this by making the cell more susceptible to toxin or
        # less responsive to metabolic stimulator, but for simplicity, let's impact visual properties.

        # Let's adjust color and irregularity directly based on alcohol
        # Membrane: Increased irregularity/damage (similar to toxin, but distinct)
        # Alcohol directly adds to membrane_irregularity_factor
        cell_data.membrane_irregularity_factor += dosage * 0.15 # Add to base irregularity

        # Particle (Cytoplasm & Vesicles): Slow down transport
        # Alcohol decreases particle_speed_factor
        cell_data.particle_speed_factor -= dosage * 0.5
        cell_data.particle_speed_factor = max(0.1, cell_data.particle_speed_factor) # Don't go below min speed

        # Mitochondria count/activity (more inactive or damaged)
        # Alcohol directly reduces active mitochondria count
        # This interacts with growth/toxin logic, so ensure it doesn't over-reduce.
        cell_data.num_active_mitochondria = int(cell_data.max_mitochondria_count * (1.0 - dosage * 0.4))
        cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))

        # Color changes due to alcohol (direct modifications, not just from toxin/metabolic)
        # Membrane becomes duller/more desaturated, possibly reddish
        r, g, b, a = cell_data.membrane_color
        cell_data.membrane_color = (
            max(0, min(r + dosage * 0.2, 1)),  # More red
            max(0, min(g - dosage * 0.3, 1)),  # Less green
            max(0, min(b - dosage * 0.3, 1)),  # Less blue
            a
        )
        
        # Nucleus becomes slightly duller
        r, g, b, a = cell_data.nucleus_color
        cell_data.nucleus_color = (
            max(0, min(r + dosage * 0.1, 1)),
            max(0, min(g - dosage * 0.1, 1)),
            max(0, min(b - dosage * 0.1, 1)),
            a
        )

        # Mitochondria become less vibrant
        r, g, b, a = cell_data.mitochondria_color
        cell_data.mitochondria_color = (
            max(0, min(r - dosage * 0.1, 1)),
            max(0, min(g - dosage * 0.2, 1)),
            max(0, min(b + dosage * 0.1, 1)), # Slightly more blue/dark
            a
        )
        
        # Cytoplasm particles become less active looking
        r, g, b, a = cell_data.particle_color
        cell_data.particle_color = (
            max(0, min(r - dosage * 0.1, 1)),
            max(0, min(g - dosage * 0.1, 1)),
            max(0, min(b - dosage * 0.1, 1)),
            a
        )


class Caffeine(Drug):
    """
    Represents the effects of Caffeine on a neural cell.
    Primary effects:
    - Increases mitochondrial metabolic efficiency/activity.
    - Increases particle speed (representing increased transport/neurotransmitter release).
    - Can slightly increase nucleus activity.
    - May reduce membrane irregularity/stabilize (very slight effect).
    """
    def __init__(self):
        super().__init__("Caffeine")

    def apply_effect(self, cell_data: NeuralCellData, dosage: float):
        """
        Applies caffeine's effects based on dosage.
        Dosage: 0.0 (no effect) to 1.0 (max effect).
        """
        dosage = max(0.0, min(1.0, dosage)) # Ensure dosage is within [0, 1]

        # Metabolic Efficiency (Mitochondria): Caffeine stimulates metabolism
        # This will be handled by metabolic_level, but we can boost it.
        # Let's increase the number of active mitochondria and their vibrancy.
        cell_data.num_active_mitochondria = int(cell_data.max_mitochondria_count * (1.0 + dosage * 0.3))
        cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))
        
        # Membrane: Slightly reduce irregularity (stabilizing effect)
        cell_data.membrane_irregularity_factor -= dosage * 0.05
        cell_data.membrane_irregularity_factor = max(0.0, cell_data.membrane_irregularity_factor) # Cannot be negative

        # Particle (Cytoplasm & Vesicles): Speed up transport
        cell_data.particle_speed_factor += dosage * 0.7
        cell_data.particle_speed_factor = min(3.0, cell_data.particle_speed_factor) # Cap max speed

        # Color changes due to caffeine (direct modifications)
        # Membrane becomes slightly more vibrant green/cyan
        r, g, b, a = cell_data.membrane_color
        cell_data.membrane_color = (
            max(0, min(r - dosage * 0.1, 1)),
            max(0, min(g + dosage * 0.2, 1)),
            max(0, min(b + dosage * 0.1, 1)),
            a
        )
        
        # Nucleus becomes brighter/more active blue
        r, g, b, a = cell_data.nucleus_color
        cell_data.nucleus_color = (
            max(0, min(r - dosage * 0.1, 1)),
            max(0, min(g + dosage * 0.1, 1)),
            max(0, min(b + dosage * 0.1, 1)),
            a
        )

        # Mitochondria become brighter/more intense yellow/orange
        r, g, b, a = cell_data.mitochondria_color
        cell_data.mitochondria_color = (
            max(0, min(r + dosage * 0.1, 1)),
            max(0, min(g + dosage * 0.1, 1)),
            max(0, min(b - dosage * 0.1, 1)),
            a
        )

        # Cytoplasm particles become more vibrant
        r, g, b, a = cell_data.particle_color
        cell_data.particle_color = (
            max(0, min(r + dosage * 0.1, 1)),
            max(0, min(g + dosage * 0.1, 1)),
            max(0, min(b + dosage * 0.1, 1)),
            a
        )