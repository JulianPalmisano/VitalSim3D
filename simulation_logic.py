# simulation_logic.py
import numpy as np
# Import both cell types
from cell_model import NeuralCellData, AstrocyteCellData
from drug_manager import DrugManager 

# Update the type hint to accept either NeuralCellData or AstrocyteCellData
def apply_compound_effects(cell_data: 'NeuralCellData | AstrocyteCellData', growth_level: float, toxin_level: float, 
                           metabolic_level: float, time_step: float, drug_manager: DrugManager) -> 'NeuralCellData | AstrocyteCellData':
    """
    Apply the effects of various compounds and drugs to the cell's state and returns the updated CellData.
    Includes homeostatic recovery mechanisms, adapted for different cell types.

    Args:
        cell_data (NeuralCellData | AstrocyteCellData): The current state of the cell's data.
        growth_level (float): Intensity of the growth factor (0.0 to 1.0).
        toxin_level (float): Intensity of the toxin (0.0 to 1.0).
        metabolic_level (float): Intensity of the metabolic stimulator (0.0 to 1.0).
        time_step (float): The time increment for this simulation step.
        drug_manager (DrugManager): The DrugManager instance to apply drug effects.

    Returns:
        NeuralCellData | AstrocyteCellData: The updated CellData object.
    """

    # --- 0. Apply Homeostatic Recovery FIRST ---
    # These properties will tend towards their healthy baseline if not actively pushed otherwise.
    
    # Scale recovery (Common to both, but specific rates might differ)
    if cell_data.current_scale > cell_data.homeo_scale:
        cell_data.current_scale -= cell_data.recovery_rate_scale * time_step
    elif cell_data.current_scale < cell_data.homeo_scale:
        cell_data.current_scale += cell_data.recovery_rate_scale * time_step
    cell_data.current_scale = max(0.5, min(cell_data.current_scale, 1.5)) # Keep within reasonable bounds

    # Type-specific recovery
    if isinstance(cell_data, NeuralCellData):
        # Neural Cell specific recoveries
        if cell_data.membrane_irregularity_factor > cell_data.homeo_membrane_irregularity:
            cell_data.membrane_irregularity_factor -= cell_data.recovery_rate_membrane_irregularity * time_step
        cell_data.membrane_irregularity_factor = max(cell_data.homeo_membrane_irregularity, cell_data.membrane_irregularity_factor)
        
        if cell_data.particle_speed_factor != cell_data.homeo_particle_speed:
            diff_speed = cell_data.homeo_particle_speed - cell_data.particle_speed_factor
            cell_data.particle_speed_factor += diff_speed * cell_data.recovery_rate_particle_speed * time_step
        cell_data.particle_speed_factor = max(0.1, min(cell_data.particle_speed_factor, 3.0)) # Keep within min/max

    elif isinstance(cell_data, AstrocyteCellData):
        # Astrocyte specific recoveries
        if cell_data.process_density_factor < cell_data.homeo_process_density:
            cell_data.process_density_factor += cell_data.recovery_rate_process_density * time_step
        elif cell_data.process_density_factor > cell_data.homeo_process_density:
            cell_data.process_density_factor -= cell_data.recovery_rate_process_density * time_step
        cell_data.process_density_factor = max(0.0, min(cell_data.process_density_factor, 1.0))

    # Active Mitochondria recovery (Common, but rates/counts differ per cell_data)
    target_active_mito_count = int(cell_data.max_mitochondria_count * cell_data.homeo_active_mitochondria_ratio)
    if cell_data.num_active_mitochondria < target_active_mito_count:
        cell_data.num_active_mitochondria += int(cell_data.recovery_rate_mitochondria * cell_data.max_mitochondria_count * time_step)
    elif cell_data.num_active_mitochondria > target_active_mito_count:
        cell_data.num_active_mitochondria -= int(cell_data.recovery_rate_mitochondria * cell_data.max_mitochondria_count * time_step)
    cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))
    
    # Color Recovery (simple interpolation towards original colors)
    recovery_rate_color = 0.05 # How fast colors revert
    
    def lerp_color(current, target, alpha):
        """Linear interpolation for colors."""
        # Ensure 'current' is a list/tuple for element-wise operation
        return tuple(c + (t - c) * alpha for c, t in zip(current, target))

    cell_data.membrane_color = lerp_color(cell_data.membrane_color, cell_data.initial_membrane_color, recovery_rate_color * time_step)
    cell_data.nucleus_color = lerp_color(cell_data.nucleus_color, cell_data.initial_nucleus_color, recovery_rate_color * time_step)
    cell_data.mitochondria_color = lerp_color(cell_data.mitochondria_color, cell_data.initial_mitochondria_color, recovery_rate_color * time_step)
    cell_data.particle_color = lerp_color(cell_data.particle_color, cell_data.initial_particle_color, recovery_rate_color * time_step)
    
    # Astrocyte specific color recovery for processes
    if isinstance(cell_data, AstrocyteCellData):
        cell_data.process_color = lerp_color(cell_data.process_color, cell_data.initial_process_color, recovery_rate_color * time_step)


    # --- 1. Apply Drug Effects ---
    drug_manager.apply_drug_effects(cell_data, time_step)


    # --- 2. Apply Compound Effects (Growth Factor, Toxin, Metabolic Stimulator) ---

    # Cell Scale (Growth/Toxin) - Common adjustment
    scale_change_rate = (growth_level * 0.01) - (toxin_level * 0.01)
    cell_data.current_scale += scale_change_rate * time_step
    cell_data.current_scale = max(0.5, min(cell_data.current_scale, 1.5))

    cell_data.membrane_points = cell_data.base_membrane_points * cell_data.current_scale
    cell_data.nucleus_points = cell_data.base_nucleus_points * cell_data.current_scale

    # Type-specific compound effects
    if isinstance(cell_data, NeuralCellData):
        # Membrane Irregularity (Toxin for Neuron)
        cell_data.membrane_irregularity_factor += toxin_level * 0.2 * time_step
        cell_data.membrane_irregularity_factor = max(0.0, min(cell_data.membrane_irregularity_factor, 0.5)) 

        if cell_data.membrane_irregularity_factor > 0:
            # Re-apply noise for visualization based on current factor
            norms = np.linalg.norm(cell_data.membrane_points, axis=1, keepdims=True)
            norms[norms == 0] = 1 # Avoid division by zero
            normals = cell_data.membrane_points / norms
            noise_magnitude = np.random.normal(0, cell_data.membrane_irregularity_factor, cell_data.membrane_points.shape[0])
            cell_data.membrane_points += normals * noise_magnitude[:, np.newaxis]

        # Particle speed (Metabolic stimulates further for Neuron)
        cell_data.particle_speed_factor += metabolic_level * 0.5 * time_step
        cell_data.particle_speed_factor = max(0.1, min(cell_data.particle_speed_factor, 5.0))
        
    elif isinstance(cell_data, AstrocyteCellData):
        # Astrocyte Process Density (Toxin reduces, Growth stimulates)
        process_density_change = (growth_level * cell_data.process_growth_rate) - (toxin_level * cell_data.process_decay_rate)
        cell_data.process_density_factor += process_density_change * time_step
        cell_data.process_density_factor = max(0.0, min(cell_data.process_density_factor, 1.0))
        
        # Re-generate astrocyte processes based on new density and scale
        cell_data.generate_processes(
            current_scale=cell_data.current_scale,
            density_factor=cell_data.process_density_factor
        )

        # Astrocyte particle speed is simpler for now, maybe tied to metabolic_level
        cell_data.particle_speed_factor = cell_data.homeo_particle_speed + metabolic_level * 0.2
        cell_data.particle_speed_factor = max(0.1, min(cell_data.particle_speed_factor, 2.0))


    # --- 3. Update Cell Colors (Further modified by compounds) ---
    # Adjustments based on compound effects for both cell types

    # Membrane color: Toxin makes it more yellow/brownish for both
    r_m, g_m, b_m, a_m = cell_data.membrane_color
    r_m += toxin_level * 0.4 * time_step
    g_m -= toxin_level * 0.5 * time_step
    b_m -= toxin_level * 0.7 * time_step
    cell_data.membrane_color = (max(0, min(r_m, 1)), max(0, min(g_m, 1)), max(0, min(b_m, 1)), a_m)

    # Nucleus color: Toxin slightly dulls for both
    n_r, n_g, n_b, n_a = cell_data.nucleus_color
    n_r += toxin_level * 0.1 * time_step
    n_g -= toxin_level * 0.1 * time_step
    cell_data.nucleus_color = (max(0, min(n_r, 1)), max(0, min(n_g, 1)), max(0, min(n_b, 1)), n_a)

    # Particle color: Metabolic stimulator brightens, Toxin dulls/reddens for both
    p_r, p_g, p_b, p_a = cell_data.particle_color
    p_r += (metabolic_level * 0.1 + toxin_level * 0.1) * time_step
    p_g += (metabolic_level * 0.1 - toxin_level * 0.1) * time_step
    p_b += (metabolic_level * 0.1 - toxin_level * 0.1) * time_step
    cell_data.particle_color = (max(0, min(p_r, 1)), max(0, min(p_g, 1)), max(0, min(p_b, 1)), p_a)

    # Astrocyte process color affected by compounds
    if isinstance(cell_data, AstrocyteCellData):
        pr_r, pr_g, pr_b, pr_a = cell_data.process_color
        pr_r += toxin_level * 0.2 * time_step
        pr_g -= toxin_level * 0.3 * time_step
        cell_data.process_color = (max(0, min(pr_r, 1)), max(0, min(pr_g, 1)), max(0, min(pr_b, 1)), pr_a)


    # --- 4. Update Mitochondria (Number/Activity) ---
    # Common logic for mitochondria count changes, but depends on cell_data's max_mitochondria_count
    change_mito_count = (growth_level * 0.5 - toxin_level * 0.8) * cell_data.max_mitochondria_count * time_step
    cell_data.num_active_mitochondria += int(change_mito_count)
    cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, cell_data.max_mitochondria_count))

    current_mitochondria_render_list = [] 
    for i in range(cell_data.num_active_mitochondria):
        if i < len(cell_data.base_mitochondria_points_list):
            jiggle_amplitude = (metabolic_level * 0.05) + (drug_manager.drug_dosage * 0.02 if drug_manager.active_drug_name == "Caffeine" else 0)
            jiggle_amplitude = max(0, jiggle_amplitude)
            
            jiggle = (np.random.rand(*cell_data.base_mitochondria_points_list[i].shape) - 0.5) * jiggle_amplitude
            scaled_mito = cell_data.base_mitochondria_points_list[i] * cell_data.current_scale + jiggle
            current_mitochondria_render_list.append(scaled_mito)
        else:
            current_mitochondria_render_list.append(np.empty((0,3)))

    cell_data.mitochondria_points = current_mitochondria_render_list

    # Mitochondria color: Metabolic stimulates, Toxin degrades for both
    m_r, m_g, m_b, m_a = cell_data.mitochondria_color
    m_r -= toxin_level * 0.2 * time_step
    m_g += (metabolic_level * 0.2 - toxin_level * 0.4) * time_step
    m_b += toxin_level * 0.1 * time_step
    cell_data.mitochondria_color = (max(0, min(m_r, 1)), max(0, min(m_g, 1)), max(0, min(m_b, 1)), m_a)

    # --- 5. Update Cytoplasm Particles (Movement and Boundary Checks) ---
    # Particle movement applies to both, but speed factor logic is type-dependent
    particle_movement = (np.random.rand(*cell_data.cytoplasm_particles.shape) - 0.5) * 0.2 * cell_data.particle_speed_factor * time_step
    cell_data.cytoplasm_particles += particle_movement

    # Boundary checks are generally similar, but based on cell's current scale
    soma_x_radius = 4.0 * cell_data.current_scale
    soma_y_radius = 5.0 * cell_data.current_scale
    soma_z_radius = 4.0 * cell_data.current_scale
    nucleus_radius = 1.8 * cell_data.current_scale # Still based on neuron for now, adjust if astrocyte has different nucleus scaling

    # Apply boundary checks for cytoplasm particles
    for i in range(len(cell_data.cytoplasm_particles)):
        p = cell_data.cytoplasm_particles[i]
        
        # Check if particle is outside soma
        if (p[0]**2 / soma_x_radius**2) + (p[1]**2 / soma_y_radius**2) + (p[2]**2 / soma_z_radius**2) >= 1.0:
            # Teleport particle back to a random position within a smaller, central region of soma
            cell_data.cytoplasm_particles[i] = np.random.uniform(-soma_x_radius * 0.7, soma_x_radius * 0.7, 3) 
            # Ensure it's outside the nucleus if it lands too close
            if np.linalg.norm(cell_data.cytoplasm_particles[i]) < nucleus_radius:
                cell_data.cytoplasm_particles[i] += (np.random.rand(3) - 0.5) * 0.5 # Nudge it
        
        # Check if particle is inside nucleus
        elif np.linalg.norm(p) < nucleus_radius:
            # Push particle away from the nucleus
            direction = p / np.linalg.norm(p) if np.linalg.norm(p) > 0 else np.array([0,0,1]) # Handle zero vector
            cell_data.cytoplasm_particles[i] = direction * nucleus_radius * 1.05 # Place just outside nucleus

    return cell_data