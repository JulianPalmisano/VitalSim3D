import numpy as np
# Corrected import to use the new class name
from cell_model import NeuralCellData 

def apply_compound_effects(cell_data: NeuralCellData, growth_level: float, toxin_level: float, metabolic_level: float, time_step: float) -> NeuralCellData:
    """
    Applies the effects of various compounds to the neural cell's state and returns the updated NeuralCellData.

    Args:
        cell_data (NeuralCellData): The current state of the neural cell's data.
        growth_level (float): Intensity of the growth factor (0.0 to 1.0).
        toxin_level (float): Intensity of the toxin (0.0 to 1.0).
        metabolic_level (float): Intensity of the metabolic stimulator (0.0 to 1.0).
        time_step (float): The time increment for this simulation step.

    Returns:
        NeuralCellData: The updated NeuralCellData object.
    """

    # --- 1. Update Cell Scale (Growth/Toxin) ---
    scale_change_rate = (growth_level * 0.01) - (toxin_level * 0.01)
    cell_data.current_scale += scale_change_rate * time_step
    cell_data.current_scale = max(0.5, min(cell_data.current_scale, 1.5))

    cell_data.membrane_points = cell_data.base_membrane_points * cell_data.current_scale
    cell_data.nucleus_points = cell_data.base_nucleus_points * cell_data.current_scale

    # --- 2. Update Membrane Irregularity (Toxin) ---
    cell_data.membrane_irregularity_factor = toxin_level * 0.2

    if cell_data.membrane_irregularity_factor > 0:
        norms = np.linalg.norm(cell_data.membrane_points, axis=1, keepdims=True)
        norms[norms == 0] = 1 
        normals = cell_data.membrane_points / norms
        
        noise_magnitude = np.random.normal(0, cell_data.membrane_irregularity_factor, cell_data.membrane_points.shape[0])
        cell_data.membrane_points += normals * noise_magnitude[:, np.newaxis]

    # --- 3. Update Cell Colors ---

    # Membrane color: Healthy (light green/cyan) to Damaged (more yellow/brownish)
    # Start: (0.3, 0.8, 0.7, 0.15)
    r_m = 0.3 + toxin_level * 0.4  # Increase red towards yellow/brown
    g_m = 0.8 - toxin_level * 0.5  # Decrease green
    b_m = 0.7 - toxin_level * 0.7  # Decrease blue, making it less cyan
    a_m = 0.15 # Keep alpha consistent

    cell_data.membrane_color = (max(0, min(r_m, 1)), max(0, min(g_m, 1)), max(0, min(b_m, 1)), a_m)

    # Nucleus color: Healthy (blue) to slightly affected (darker blue/purple)
    # Start: (0.3, 0.3, 0.8, 0.7)
    n_r = 0.3 + toxin_level * 0.1 # Slight red tint with toxin
    n_g = 0.3 - toxin_level * 0.1 # Slight green reduction
    n_b = 0.8 # Keep blue dominant
    n_a = 0.7 # Alpha

    cell_data.nucleus_color = (max(0, min(n_r, 1)), max(0, min(n_g, 1)), max(0, min(n_b, 1)), n_a)

    # Particle color: Healthy (gray) to more active/damaged (brighter, potentially redder)
    # Start: (0.8, 0.8, 0.8, 0.4)
    # Metabolic stimulator makes them brighter/more active. Toxin could make them duller/redder.
    p_r = 0.8 + metabolic_level * 0.1 + toxin_level * 0.1 # Brighten and add red with toxin
    p_g = 0.8 + metabolic_level * 0.1 - toxin_level * 0.1 # Brighten, but less green with toxin
    p_b = 0.8 + metabolic_level * 0.1 - toxin_level * 0.1 # Brighten, but less blue with toxin
    p_a = 0.4 # Alpha

    cell_data.particle_color = (max(0, min(p_r, 1)), max(0, min(p_g, 1)), max(0, min(p_b, 1)), p_a)

    # --- 4. Update Mitochondria (Number/Size/Activity) ---
    max_mito_count = len(cell_data.base_mitochondria_points_list)
    cell_data.num_active_mitochondria = int(max_mito_count * (1.0 + growth_level * 0.5 - toxin_level * 0.8))
    cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, max_mito_count))

    current_mitochondria_render_list = [] 
    for i in range(cell_data.num_active_mitochondria):
        if i < len(cell_data.base_mitochondria_points_list):
            jiggle_amplitude = metabolic_level * 0.05
            jiggle = (np.random.rand(*cell_data.base_mitochondria_points_list[i].shape) - 0.5) * jiggle_amplitude
            scaled_mito = cell_data.base_mitochondria_points_list[i] * cell_data.current_scale + jiggle
            current_mitochondria_render_list.append(scaled_mito)
        else:
            current_mitochondria_render_list.append(np.empty((0,3)))

    cell_data.mitochondria_points = current_mitochondria_render_list

    # Mitochondria color: Healthy (yellow/orange) to stressed/damaged (duller, more red/brown)
    # Start: (1.0, 0.7, 0.0, 0.9)
    m_r = 1.0 - toxin_level * 0.2 # Red slightly down with toxin
    m_g = 0.7 + metabolic_level * 0.2 - toxin_level * 0.4 # Green up with metabolic, significantly down with toxin
    m_b = 0.0 + toxin_level * 0.1 # Introduce a little blue/darkness with toxin

    cell_data.mitochondria_color = (max(0, min(m_r, 1)), max(0, min(m_g, 1)), max(0, min(m_b, 1)), cell_data.mitochondria_color[3])

    # --- 5. Update Cytoplasm Particles (Movement) ---
    cell_data.particle_speed_factor = 1.0 + metabolic_level * 0.5
    particle_movement = (np.random.rand(*cell_data.cytoplasm_particles.shape) - 0.5) * 0.2 * cell_data.particle_speed_factor * time_step
    cell_data.cytoplasm_particles += particle_movement

    soma_x_radius = 4.0 * cell_data.current_scale
    soma_y_radius = 5.0 * cell_data.current_scale
    soma_z_radius = 4.0 * cell_data.current_scale
    nucleus_radius = 1.8 * cell_data.current_scale

    for i in range(len(cell_data.cytoplasm_particles)):
        p = cell_data.cytoplasm_particles[i]
        
        if (p[0]**2 / soma_x_radius**2) + (p[1]**2 / soma_y_radius**2) + (p[2]**2 / soma_z_radius**2) >= 1.0:
            cell_data.cytoplasm_particles[i] = np.random.uniform(-soma_x_radius * 0.8, soma_x_radius * 0.8, 3) 
            if np.linalg.norm(cell_data.cytoplasm_particles[i]) < nucleus_radius:
                cell_data.cytoplasm_particles[i] += (np.random.rand(3) - 0.5) * 0.5 
        
        elif np.linalg.norm(p) < nucleus_radius:
            direction = p / np.linalg.norm(p) if np.linalg.norm(p) > 0 else np.array([0,0,1])
            cell_data.cytoplasm_particles[i] = direction * nucleus_radius * 1.1
            
    return cell_data