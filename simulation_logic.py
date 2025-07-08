import numpy as np
from cell_model import CellData # Import the CellData class

def apply_compound_effects(cell_data: CellData, growth_level: float, toxin_level: float, metabolic_level: float, time_step: float) -> CellData:
    """
    Applies the effects of various compounds to the cell's state and returns the updated CellData.

    Args:
        cell_data (CellData): The current state of the cell's data.
        growth_level (float): Intensity of the growth factor (0.0 to 1.0).
        toxin_level (float): Intensity of the toxin (0.0 to 1.0).
        metabolic_level (float): Intensity of the metabolic stimulator (0.0 to 1.0).
        time_step (float): The time increment for this simulation step.

    Returns:
        CellData: The updated CellData object.
    """

    # --- 1. Update Cell Scale (Growth/Toxin) ---
    # Growth increases scale, toxin decreases.
    # We'll make the change proportional to the time_step and compound levels.
    # A small factor (e.g., 0.01) controls the speed of scaling.
    scale_change_rate = (growth_level * 0.01) - (toxin_level * 0.01)
    cell_data.current_scale += scale_change_rate * time_step
    # Clamp scale to reasonable biological limits (e.g., 0.5 to 1.5 times base size)
    cell_data.current_scale = max(0.5, min(cell_data.current_scale, 1.5))

    # Apply scaling to all base points to get current points
    cell_data.membrane_points = cell_data.base_membrane_points * cell_data.current_scale
    cell_data.nucleus_points = cell_data.base_nucleus_points * cell_data.current_scale

    # --- 2. Update Membrane Irregularity (Toxin) ---
    # Toxin increases irregularity. Max irregularity factor 0.2 (arbitrary visual effect)
    cell_data.membrane_irregularity_factor = toxin_level * 0.2

    if cell_data.membrane_irregularity_factor > 0:
        # Apply noise based on original normal vectors to push points in/out
        # This creates a "jagged" or "damaged" look for the membrane
        # Calculate approximate normals (vector from origin to point)
        normals = cell_data.membrane_points / np.linalg.norm(cell_data.membrane_points, axis=1, keepdims=True)
        # Add random noise along the normal direction
        noise_magnitude = np.random.normal(0, cell_data.membrane_irregularity_factor, cell_data.membrane_points.shape[0])
        cell_data.membrane_points += normals * noise_magnitude[:, np.newaxis] # Apply noise along normal

    # --- 3. Update Cell Colors (Toxin for membrane, Metabolic for particles) ---
    # Membrane color: from healthy green (low toxin) to damaged red (high toxin)
    r = toxin_level * 0.8 + (1 - toxin_level) * 0.2 # More red with toxin
    g = (1 - toxin_level) * 0.8 + toxin_level * 0.2 # Less green with toxin
    b = 0.2 # Keep some blue
    a = 0.8 # Alpha (transparency)
    cell_data.membrane_color = (r, g, b, a)

    # Nucleus color: slightly affected by toxin (e.g., darker)
    n_r = cell_data.nucleus_color[0] + toxin_level * 0.1
    n_g = cell_data.nucleus_color[1] - toxin_level * 0.1
    n_b = cell_data.nucleus_color[2]
    cell_data.nucleus_color = (max(0, min(n_r, 1)), max(0, min(n_g, 1)), n_b, cell_data.nucleus_color[3])


    # Particle color: from yellow (normal) to brighter/more active (metabolic stimulator)
    p_r = 1.0 # Always red component
    p_g = 1.0 - metabolic_level * 0.5 # Less green with stimulator (more orange/red)
    p_b = 0.0 # No blue
    p_a = 0.7 # Alpha
    cell_data.particle_color = (p_r, p_g, p_b, p_a)

    # --- 4. Update Mitochondria (Number/Size/Activity) ---
    # Number of active mitochondria based on growth/toxin
    # Max mitochondria count is fixed by the number of base_mitochondria_points_list
    max_mito_count = len(cell_data.base_mitochondria_points_list)
    cell_data.num_active_mitochondria = int(max_mito_count * (1.0 + growth_level * 0.5 - toxin_level * 0.5))
    cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, max_mito_count))

    # Scale and update positions for active mitochondria
    cell_data.mitochondria_points = []
    for i in range(cell_data.num_active_mitochondria):
        # Apply scaling and a subtle "jiggle" for activity
        jiggle_amplitude = metabolic_level * 0.05 # Jiggle more with stimulator
        jiggle = (np.random.rand(*cell_data.base_mitochondria_points_list[i].shape) - 0.5) * jiggle_amplitude
        scaled_mito = cell_data.base_mitochondria_points_list[i] * cell_data.current_scale + jiggle
        cell_data.mitochondria_points.append(scaled_mito)

    # Mitochondria color can also change with metabolic stimulator
    m_r = cell_data.mitochondria_color[0] + metabolic_level * 0.1
    m_g = cell_data.mitochondria_color[1] - metabolic_level * 0.1
    cell_data.mitochondria_color = (max(0, min(m_r, 1)), max(0, min(m_g, 1)), cell_data.mitochondria_color[2], cell_data.mitochondria_color[3])


    # --- 5. Update Cytoplasm Particles (Movement) ---
    # Particle speed based on metabolic stimulator
    cell_data.particle_speed_factor = 1.0 + metabolic_level * 0.5 # Base speed + stimulator effect

    # Simple random walk for particles
    particle_movement = (np.random.rand(*cell_data.cytoplasm_particles.shape) - 0.5) * 0.2 * cell_data.particle_speed_factor
    cell_data.cytoplasm_particles += particle_movement

    # Keep particles roughly within the current cell bounds (simplified spherical clamp)
    current_cell_radius = 5.0 * cell_data.current_scale # Approximate radius
    distances_from_center = np.linalg.norm(cell_data.cytoplasm_particles, axis=1)
    # If a particle moves too far, reset it closer to the center or within bounds
    # A simple clamp is sufficient for visual effect
    cell_data.cytoplasm_particles = np.clip(cell_data.cytoplasm_particles,
                                            -current_cell_radius, current_cell_radius)


    return cell_data

