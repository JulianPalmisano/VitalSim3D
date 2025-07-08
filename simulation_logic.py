import numpy as np
from cell_model import NeuralCellData
# NEW: Import DrugManager
from drug_manager import DrugManager 

# NEW: Pass DrugManager instance
def apply_compound_effects(cell_data: NeuralCellData, growth_level: float, toxin_level: float, 
                           metabolic_level: float, time_step: float, drug_manager: DrugManager) -> NeuralCellData:
    """
    Applies the effects of various compounds and drugs to the neural cell's state and returns the updated NeuralCellData.

    Args:
        cell_data (NeuralCellData): The current state of the neural cell's data.
        growth_level (float): Intensity of the growth factor (0.0 to 1.0).
        toxin_level (float): Intensity of the toxin (0.0 to 1.0).
        metabolic_level (float): Intensity of the metabolic stimulator (0.0 to 1.0).
        time_step (float): The time increment for this simulation step.
        drug_manager (DrugManager): The DrugManager instance to apply drug effects.

    Returns:
        NeuralCellData: The updated NeuralCellData object.
    """

    # --- NEW: Apply Drug Effects FIRST ---
    # Drugs will directly modify cell_data properties.
    # This might slightly re-normalize properties that were set by 'reset_simulation'
    # or the start of a simulation, so we need to ensure the order is logical.
    
    # IMPORTANT: To ensure drugs can modify initial states and not just add to existing levels,
    # we should first reset the dynamic properties to their "homeostatic" state (without drugs or compounds),
    # then apply drugs, then apply compounds.
    # This means moving some 'resetting' from main_app.reset_simulation into here or a dedicated
    # CellData reset function called at the beginning of each simulation step.

    # For simplicity and to fit your existing structure, let's assume drugs
    # are applied on top of the 'current' state, and their effects are persistent
    # until the drug level changes or the simulation is reset.
    # A more robust system might 'undo' previous effects or reset to a clean slate each step.

    # Option 1 (Simpler but less precise for cumulative effects): Apply drugs on current state
    # This means if toxin already made membrane irregular, alcohol adds *more* irregularity.
    drug_manager.apply_drug_effects(cell_data)

    # Option 2 (More robust, but requires exposing a 'reset_dynamic_properties' in NeuralCellData):
    # cell_data.reset_dynamic_simulation_properties() # New method to reset properties affected by compounds/drugs
    # drug_manager.apply_drug_effects(cell_data)
    # Then apply compound effects.
    # For now, let's stick with Option 1 for minimal changes, knowing there might be some interaction complexities.


    # --- 1. Update Cell Scale (Growth/Toxin) ---
    # These base calculations are still relevant. Drugs can pre-modify the cell_data
    # properties that these calculations then use (e.g., if alcohol makes cell smaller,
    # then growth factor has to work harder).
    scale_change_rate = (growth_level * 0.01) - (toxin_level * 0.01)
    cell_data.current_scale += scale_change_rate * time_step
    cell_data.current_scale = max(0.5, min(cell_data.current_scale, 1.5))

    cell_data.membrane_points = cell_data.base_membrane_points * cell_data.current_scale
    cell_data.nucleus_points = cell_data.base_nucleus_points * cell_data.current_scale

    # --- 2. Update Membrane Irregularity (Toxin) ---
    # The membrane_irregularity_factor might have been affected by drugs (e.g., alcohol).
    # Toxin then adds to this base.
    cell_data.membrane_irregularity_factor += toxin_level * 0.2 # Toxin adds its part
    cell_data.membrane_irregularity_factor = max(0.0, min(cell_data.membrane_irregularity_factor, 0.5)) # Cap irregularity

    if cell_data.membrane_irregularity_factor > 0:
        norms = np.linalg.norm(cell_data.membrane_points, axis=1, keepdims=True)
        norms[norms == 0] = 1 
        normals = cell_data.membrane_points / norms
        
        noise_magnitude = np.random.normal(0, cell_data.membrane_irregularity_factor, cell_data.membrane_points.shape[0])
        cell_data.membrane_points += normals * noise_magnitude[:, np.newaxis]

    # --- 3. Update Cell Colors ---
    # Colors were modified by drugs. Now, toxin/metabolic levels can *further* modify them.
    # This means the color blending will be cumulative.

    # Membrane color: Healthy (light green/cyan) to Damaged (more yellow/brownish)
    # Start: (0.3, 0.8, 0.7, 0.15) (affected by alcohol/caffeine in apply_drug_effects)
    # Now, add toxin effect on top of drug effect.
    r_m, g_m, b_m, a_m = cell_data.membrane_color # Get current color after potential drug effects
    
    r_m += toxin_level * 0.4 
    g_m -= toxin_level * 0.5 
    b_m -= toxin_level * 0.7 
    # a_m remains constant as per your original logic

    cell_data.membrane_color = (max(0, min(r_m, 1)), max(0, min(g_m, 1)), max(0, min(b_m, 1)), a_m)

    # Nucleus color: Healthy (blue) to slightly affected (darker blue/purple)
    # Start: (0.3, 0.3, 0.8, 0.7)
    n_r, n_g, n_b, n_a = cell_data.nucleus_color
    n_r += toxin_level * 0.1
    n_g -= toxin_level * 0.1 
    # n_b remains dominant but can be influenced by drugs
    cell_data.nucleus_color = (max(0, min(n_r, 1)), max(0, min(n_g, 1)), max(0, min(n_b, 1)), n_a)

    # Particle color: Healthy (gray) to more active/damaged (brighter, potentially redder)
    # Start: (0.8, 0.8, 0.8, 0.4)
    p_r, p_g, p_b, p_a = cell_data.particle_color
    p_r += metabolic_level * 0.1 + toxin_level * 0.1 
    p_g += metabolic_level * 0.1 - toxin_level * 0.1 
    p_b += metabolic_level * 0.1 - toxin_level * 0.1 
    # p_a remains constant
    cell_data.particle_color = (max(0, min(p_r, 1)), max(0, min(p_g, 1)), max(0, min(p_b, 1)), p_a)

    # --- 4. Update Mitochondria (Number/Size/Activity) ---
    max_mito_count = len(cell_data.base_mitochondria_points_list)
    # The num_active_mitochondria could have been altered by drugs.
    # Now, growth/toxin further modify this *new* number.
    # A cleaner approach might be to calculate the 'target' num based on all factors combined.
    # For now, let's make it additive on top of drug effects.
    cell_data.num_active_mitochondria += int(growth_level * 0.5 * max_mito_count - toxin_level * 0.8 * max_mito_count)
    cell_data.num_active_mitochondria = max(0, min(cell_data.num_active_mitochondria, max_mito_count))

    current_mitochondria_render_list = [] 
    for i in range(cell_data.num_active_mitochondria):
        if i < len(cell_data.base_mitochondria_points_list):
            # Jiggle amplitude is also influenced by metabolic level
            jiggle_amplitude = (metabolic_level * 0.05) + (drug_manager.drug_dosage * 0.02 if drug_manager.active_drug_name == "Caffeine" else 0)
            jiggle_amplitude = max(0, jiggle_amplitude) # Ensure positive
            
            jiggle = (np.random.rand(*cell_data.base_mitochondria_points_list[i].shape) - 0.5) * jiggle_amplitude
            scaled_mito = cell_data.base_mitochondria_points_list[i] * cell_data.current_scale + jiggle
            current_mitochondria_render_list.append(scaled_mito)
        else:
            current_mitochondria_render_list.append(np.empty((0,3)))

    cell_data.mitochondria_points = current_mitochondria_render_list

    # Mitochondria color: Healthy (yellow/orange) to stressed/damaged (duller, more red/brown)
    m_r, m_g, m_b, m_a = cell_data.mitochondria_color
    m_r -= toxin_level * 0.2 
    m_g += metabolic_level * 0.2 - toxin_level * 0.4 
    m_b += toxin_level * 0.1 

    cell_data.mitochondria_color = (max(0, min(m_r, 1)), max(0, min(m_g, 1)), max(0, min(m_b, 1)), m_a)

    # --- 5. Update Cytoplasm Particles (Movement) ---
    # Particle speed factor can be influenced by drugs (e.g., caffeine/alcohol)
    cell_data.particle_speed_factor += metabolic_level * 0.5
    cell_data.particle_speed_factor = max(0.1, min(cell_data.particle_speed_factor, 3.0)) # Cap speed

    particle_movement = (np.random.rand(*cell_data.cytoplasm_particles.shape) - 0.5) * 0.2 * cell_data.particle_speed_factor * time_step
    cell_data.cytoplasm_particles += particle_movement

    soma_x_radius = 4.0 * cell_data.current_scale
    soma_y_radius = 5.0 * cell_data.current_scale
    soma_z_radius = 4.0 * cell_data.current_scale
    nucleus_radius = 1.8 * cell_data.current_scale

    # Boundary checks for cytoplasm particles
    for i in range(len(cell_data.cytoplasm_particles)):
        p = cell_data.cytoplasm_particles[i]
        
        # Check if outside soma (using ellipsoid equation)
        if (p[0]**2 / soma_x_radius**2) + (p[1]**2 / soma_y_radius**2) + (p[2]**2 / soma_z_radius**2) >= 1.0:
            # If outside, re-randomize within a slightly smaller region inside soma
            cell_data.cytoplasm_particles[i] = np.random.uniform(-soma_x_radius * 0.7, soma_x_radius * 0.7, 3) 
            # If newly placed particle lands inside nucleus, push it out
            if np.linalg.norm(cell_data.cytoplasm_particles[i]) < nucleus_radius:
                cell_data.cytoplasm_particles[i] += (np.random.rand(3) - 0.5) * 0.5 
        
        # Check if inside nucleus
        elif np.linalg.norm(p) < nucleus_radius:
            # Push particles out of nucleus
            direction = p / np.linalg.norm(p) if np.linalg.norm(p) > 0 else np.array([0,0,1]) # Handle origin case
            cell_data.cytoplasm_particles[i] = direction * nucleus_radius * 1.05 # Push just outside nucleus boundary
            
    return cell_data