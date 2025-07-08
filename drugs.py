# drugs.py
from abc import ABC, abstractmethod
import numpy as np
from cell_model import NeuralCellData

class Drug(ABC):
    """
    Abstract base class for all drugs.
    Defines the interface for applying drug effects.
    """
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply_effect(self, cell_data: NeuralCellData, dosage: float, time_step: float):
        """
        Applies the drug's effect to the cell components based on dosage and time step.
        This method will modify the properties within the cell_data object directly,
        considering homeostatic tendencies.
        :param cell_data: The NeuralCellData object to apply effects to.
        :param dosage: The intensity of the drug (e.g., 0.0 to 1.0).
        :param time_step: The time increment for this simulation step.
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

    def apply_effect(self, cell_data: NeuralCellData, dosage: float, time_step: float):
        """
        Applies alcohol's effects based on dosage and time step.
        Dosage: 0.0 (no effect) to 1.0 (max effect).
        """
        dosage = max(0.0, min(1.0, dosage)) # Ensure dosage is within [0, 1]

        # Membrane Irregularity: Alcohol increases irregularity
        target_irregularity_increase = dosage * 0.4 # Max increase due to alcohol
        cell_data.membrane_irregularity_factor += target_irregularity_increase * time_step * 0.5 # Faster onset
        cell_data.membrane_irregularity_factor = min(cell_data.membrane_irregularity_factor, 0.8) # Clamp max irregularity

        # Particle Speed: Alcohol decreases speed
        target_particle_speed_decrease = dosage * 0.8 # Max decrease
        cell_data.particle_speed_factor -= target_particle_speed_decrease * time_step * 0.5 # Apply decrease
        cell_data.particle_speed_factor = max(0.1, cell_data.particle_speed_factor) # Don't go below min speed

        # Active Mitochondria: Alcohol reduces active count
        target_active_mito_ratio_decrease = dosage * 0.6 # Max reduction
        # Calculate change based on max possible reduction, then apply gradually
        change_in_mito = int(target_active_mito_ratio_decrease * cell_data.max_mitochondria_count * time_step * 0.5)
        cell_data.num_active_mitochondria -= change_in_mito
        cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))

        # Cell Scale: Alcohol slightly reduces scale
        target_scale_decrease = dosage * 0.1
        cell_data.current_scale -= target_scale_decrease * time_step * 0.2
        cell_data.current_scale = max(0.6, cell_data.current_scale) # Don't shrink too much

        # Color changes (gradual nudging towards "damaged" colors)
        r, g, b, a = cell_data.membrane_color
        cell_data.membrane_color = (
            max(0, min(r + dosage * 0.2 * time_step, 1)),  # More red
            max(0, min(g - dosage * 0.3 * time_step, 1)),  # Less green
            max(0, min(b - dosage * 0.3 * time_step, 1)),  # Less blue
            a
        )
        cell_data.nucleus_color = (
            max(0, min(cell_data.nucleus_color[0] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[1] - dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[2] - dosage * 0.1 * time_step, 1)),
            cell_data.nucleus_color[3]
        )
        cell_data.mitochondria_color = (
            max(0, min(cell_data.mitochondria_color[0] - dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[1] - dosage * 0.2 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[2] + dosage * 0.1 * time_step, 1)),
            cell_data.mitochondria_color[3]
        )
        cell_data.particle_color = (
            max(0, min(cell_data.particle_color[0] - dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.particle_color[1] - dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.particle_color[2] - dosage * 0.1 * time_step, 1)),
            cell_data.particle_color[3]
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

    def apply_effect(self, cell_data: NeuralCellData, dosage: float, time_step: float):
        """
        Applies caffeine's effects based on dosage and time step.
        Dosage: 0.0 (no effect) to 1.0 (max effect).
        """
        dosage = max(0.0, min(1.0, dosage)) # Ensure dosage is within [0, 1]

        # Active Mitochondria: Caffeine increases active count
        target_active_mito_ratio_increase = dosage * 0.4 # Max increase
        change_in_mito = int(target_active_mito_ratio_increase * cell_data.max_mitochondria_count * time_step * 0.5)
        cell_data.num_active_mitochondria += change_in_mito
        cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))

        # Membrane Irregularity: Caffeine slightly reduces irregularity (stabilizing)
        target_irregularity_decrease = dosage * 0.1
        cell_data.membrane_irregularity_factor -= target_irregularity_decrease * time_step * 0.5
        cell_data.membrane_irregularity_factor = max(0.0, cell_data.membrane_irregularity_factor) # Cannot be negative

        # Particle Speed: Caffeine increases speed
        target_particle_speed_increase = dosage * 1.5
        cell_data.particle_speed_factor += target_particle_speed_increase * time_step * 0.5
        cell_data.particle_speed_factor = min(3.0, cell_data.particle_speed_factor) # Cap max speed

        # Cell Scale: Caffeine can slightly increase scale (growth effect)
        target_scale_increase = dosage * 0.05
        cell_data.current_scale += target_scale_increase * time_step * 0.2
        cell_data.current_scale = min(1.5, cell_data.current_scale)

        # Color changes (gradual nudging towards "active" colors)
        r, g, b, a = cell_data.membrane_color
        cell_data.membrane_color = (
            max(0, min(r - dosage * 0.1 * time_step, 1)),
            max(0, min(g + dosage * 0.2 * time_step, 1)),
            max(0, min(b + dosage * 0.1 * time_step, 1)),
            a
        )
        cell_data.nucleus_color = (
            max(0, min(cell_data.nucleus_color[0] - dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[1] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[2] + dosage * 0.1 * time_step, 1)),
            cell_data.nucleus_color[3]
        )
        cell_data.mitochondria_color = (
            max(0, min(cell_data.mitochondria_color[0] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[1] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[2] - dosage * 0.1 * time_step, 1)),
            cell_data.mitochondria_color[3]
        )
        cell_data.particle_color = (
            max(0, min(cell_data.particle_color[0] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.particle_color[1] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.particle_color[2] + dosage * 0.1 * time_step, 1)),
            cell_data.particle_color[3]
        )

class Nicotine(Drug):
    """
    Represents the effects of Nicotine on a neural cell.
    Primary effects:
    - Increases particle speed/vesicle transport (neurotransmitter release).
    - Slight increase in membrane irregularity (due to increased activity/permeability).
    - Slight boost to mitochondrial activity.
    - Colors become more vibrant, but less intense than amphetamine.
    """
    def __init__(self):
        super().__init__("Nicotine")

    def apply_effect(self, cell_data: NeuralCellData, dosage: float, time_step: float):
        dosage = max(0.0, min(1.0, dosage))

        # Particle Speed: Nicotine increases speed (neurotransmitter release)
        target_particle_speed_increase = dosage * 0.8
        cell_data.particle_speed_factor += target_particle_speed_increase * time_step * 0.5
        cell_data.particle_speed_factor = min(2.5, cell_data.particle_speed_factor) # Cap speed

        # Membrane Irregularity: Slight increase due to increased activity/permeability
        target_irregularity_increase = dosage * 0.1
        cell_data.membrane_irregularity_factor += target_irregularity_increase * time_step * 0.5
        cell_data.membrane_irregularity_factor = min(cell_data.membrane_irregularity_factor, 0.6)

        # Active Mitochondria: Slight boost
        target_active_mito_ratio_increase = dosage * 0.2
        change_in_mito = int(target_active_mito_ratio_increase * cell_data.max_mitochondria_count * time_step * 0.5)
        cell_data.num_active_mitochondria += change_in_mito
        cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))

        # Cell Scale: Neutral or very slight increase
        # No significant direct effect on scale for simplicity

        # Color changes: Generally more vibrant, yellowish/greenish tint for activity
        r, g, b, a = cell_data.membrane_color
        cell_data.membrane_color = (
            max(0, min(r + dosage * 0.05 * time_step, 1)), # Slightly more red/yellow
            max(0, min(g + dosage * 0.15 * time_step, 1)), # More green
            max(0, min(b - dosage * 0.05 * time_step, 1)), # Less blue
            a
        )
        cell_data.nucleus_color = (
            max(0, min(cell_data.nucleus_color[0] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[1] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[2] + dosage * 0.05 * time_step, 1)),
            cell_data.nucleus_color[3]
        )
        cell_data.mitochondria_color = (
            max(0, min(cell_data.mitochondria_color[0] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[1] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[2] - dosage * 0.05 * time_step, 1)),
            cell_data.mitochondria_color[3]
        )
        cell_data.particle_color = (
            max(0, min(cell_data.particle_color[0] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.particle_color[1] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.particle_color[2] + dosage * 0.05 * time_step, 1)),
            cell_data.particle_color[3]
        )

class Amphetamine(Drug):
    """
    Represents the effects of Amphetamine on a neural cell.
    Primary effects:
    - Significant increase in particle speed/vesicle transport (strong neurotransmitter surge).
    - Significant increase in mitochondrial activity (high energy demand).
    - Increased nucleus activity.
    - Colors become very vibrant, potentially with a "hotter" tone.
    """
    def __init__(self):
        super().__init__("Amphetamine")

    def apply_effect(self, cell_data: NeuralCellData, dosage: float, time_step: float):
        dosage = max(0.0, min(1.0, dosage))

        # Particle Speed: Significant increase
        target_particle_speed_increase = dosage * 2.5
        cell_data.particle_speed_factor += target_particle_speed_increase * time_step * 0.5
        cell_data.particle_speed_factor = min(5.0, cell_data.particle_speed_factor) # Higher cap for amphetamine

        # Active Mitochondria: Significant increase
        target_active_mito_ratio_increase = dosage * 0.6
        change_in_mito = int(target_active_mito_ratio_increase * cell_data.max_mitochondria_count * time_step * 0.5)
        cell_data.num_active_mitochondria += change_in_mito
        cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))

        # Nucleus activity: Increased (represented by color vibrancy)
        # No direct property for nucleus activity level, but colors will reflect this.

        # Membrane Irregularity: Slight increase at very high doses/stress
        target_irregularity_increase = dosage * 0.15 # More pronounced than nicotine
        cell_data.membrane_irregularity_factor += target_irregularity_increase * time_step * 0.5
        cell_data.membrane_irregularity_factor = min(cell_data.membrane_irregularity_factor, 0.7)

        # Cell Scale: Neutral or slight decrease (stress)
        # For simplicity, let's assume no significant scale change from amphetamine.

        # Color changes: Very vibrant, reddish/orange hue (high activity, potential stress)
        r, g, b, a = cell_data.membrane_color
        cell_data.membrane_color = (
            max(0, min(r + dosage * 0.2 * time_step, 1)), # More red
            max(0, min(g + dosage * 0.1 * time_step, 1)), # Slightly more green
            max(0, min(b - dosage * 0.2 * time_step, 1)), # Less blue
            a
        )
        cell_data.nucleus_color = (
            max(0, min(cell_data.nucleus_color[0] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[1] + dosage * 0.1 * time_step, 1)),
            max(0, min(cell_data.nucleus_color[2] - dosage * 0.1 * time_step, 1)), # Shift towards red/yellow
            cell_data.nucleus_color[3]
        )
        cell_data.mitochondria_color = (
            max(0, min(cell_data.mitochondria_color[0] + dosage * 0.15 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[1] + dosage * 0.05 * time_step, 1)),
            max(0, min(cell_data.mitochondria_color[2] - dosage * 0.1 * time_step, 1)),
            cell_data.mitochondria_color[3]
        )
        cell_data.particle_color = (
            max(0, min(cell_data.particle_color[0] + dosage * 0.15 * time_step, 1)),
            max(0, min(cell_data.particle_color[1] + dosage * 0.15 * time_step, 1)),
            max(0, min(cell_data.particle_color[2] - dosage * 0.1 * time_step, 1)),
            cell_data.particle_color[3]
        )