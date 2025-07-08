# cell_model.py
import numpy as np
from math import sin, cos, pi

class NeuralCellData:
    """
    Encapsulates all data related to the neural cell's 3D structure and visual properties.
    This class handles the generation of the initial (homeostasis) neural cell state.
    """
    def __init__(self):
        # Raw point data for each component (original, untransformed state)
        self.base_membrane_points = None
        self.base_nucleus_points = None
        self.base_mitochondria_points_list = [] # List of numpy arrays, one for each mito
        self.base_cytoplasm_particles = None

        # Current (transformed) point data for rendering.
        # These will be updated in a simulation/rendering loop, not here.
        self.membrane_points = None
        self.nucleus_points = None
        self.mitochondria_points = [] # List of numpy arrays
        self.cytoplasm_particles = None

        # Visual properties (colors, sizes, and LABELS for the legend)
        self.membrane_color = (0.3, 0.8, 0.7, 0.15) # RGBA: Soft cyan/green, highly transparent
        self.membrane_point_size = 4
        self.membrane_label = "Neuron Soma Membrane"

        self.nucleus_color = (0.3, 0.3, 0.8, 0.7) # RGBA: Slightly darker blue, good visibility
        self.nucleus_point_size = 8
        self.nucleus_label = "Nucleus" 

        self.mitochondria_color = (1.0, 0.7, 0.0, 0.9) # RGBA: Brighter orange/yellow
        self.mitochondria_point_size = 5
        self.mitochondria_label = "Mitochondria" 

        self.max_mitochondria_count = 20
        self.num_active_mitochondria = self.max_mitochondria_count # Start with all active

        self.particle_color = (0.8, 0.8, 0.8, 0.4) # RGBA: Off-white/gray, good contrast with new membrane
        self.particle_size = 2.5
        self.particle_label = "Cytoplasm & Vesicles"

        # Dynamic properties that will be modified by simulation_logic (initially set)
        self.current_scale = 1.0 
        self.membrane_irregularity_factor = 0.0 
        self.particle_speed_factor = 1.0 
        
        # --- NEW: Homeostatic/Target Values and Recovery Rates ---
        self.homeo_scale = 1.0
        self.homeo_membrane_irregularity = 0.0
        self.homeo_particle_speed = 1.0
        self.homeo_active_mitochondria_ratio = 1.0 # 1.0 means all are active initially

        self.recovery_rate_scale = 0.01 # How fast scale reverts to homeo
        self.recovery_rate_membrane_irregularity = 0.05 # How fast irregularity reduces
        self.recovery_rate_particle_speed = 0.05 # How fast particle speed reverts
        self.recovery_rate_mitochondria = 0.02 # How fast active mito count recovers

        # Store initial colors for recovery in simulation_logic
        self.initial_membrane_color = (0.3, 0.8, 0.7, 0.15)
        self.initial_nucleus_color = (0.3, 0.3, 0.8, 0.7)
        self.initial_mitochondria_color = (1.0, 0.7, 0.0, 0.9)
        self.initial_particle_color = (0.8, 0.8, 0.8, 0.4)

        # --- Neural Cell Specific Properties (for future simulation logic) ---
        self.tau_tangle_level = 0.0 
        self.amyloid_beta_level = 0.0 
        self.synaptic_health = 1.0 

        # Generate the initial state when the object is created
        self.generate_initial_state()

    def generate_initial_state(self):
        """
        Generates the base 3D points for all cell components in their
        homeostasis (initial) state. This is called on init and on reset.
        """
        # --- POINT COUNT ADJUSTMENTS FOR OPTIMIZATION ---
        self.num_membrane_points = 2500
        self.num_nucleus_points = 800
        self.num_mitochondria_points_per_mito = 100
        self.num_cytoplasm_particles = 1000

        # --- Membrane (deformed ovoid/soma-like) ---
        soma_x_radius = 4.0
        soma_y_radius = 5.0
        soma_z_radius = 4.0
        
        points_on_sphere = np.random.randn(self.num_membrane_points, 3)
        points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1)[:, np.newaxis]
        
        points_on_sphere[:,0] *= soma_x_radius
        points_on_sphere[:,1] *= soma_y_radius
        points_on_sphere[:,2] *= soma_z_radius

        self.base_membrane_points = points_on_sphere
        
        initial_noise = np.random.normal(0, 0.4, self.base_membrane_points.shape)
        self.base_membrane_points += initial_noise
        self.membrane_points = np.copy(self.base_membrane_points)

        # --- Nucleus (typically prominent and centrally located in neuron soma) ---
        nucleus_radius = 1.8
        points_on_sphere_nucleus = np.random.randn(self.num_nucleus_points, 3)
        points_on_sphere_nucleus /= np.linalg.norm(points_on_sphere_nucleus, axis=1)[:, np.newaxis]
        self.base_nucleus_points = points_on_sphere_nucleus * nucleus_radius
        self.nucleus_points = np.copy(self.base_nucleus_points)

        # --- Mitochondria (numerous, distributed, potentially elongated) ---
        self.base_mitochondria_points_list = []
        # Ensure mitochondria_points has the correct initial size
        self.mitochondria_points = [None] * self.max_mitochondria_count 

        soma_max_dim = max(soma_x_radius, soma_y_radius, soma_z_radius)
        
        for i in range(self.max_mitochondria_count):
            while True:
                mito_buffer = 1.5
                center_x = np.random.uniform(-soma_x_radius + mito_buffer, soma_x_radius - mito_buffer)
                center_y = np.random.uniform(-soma_y_radius + mito_buffer, soma_y_radius - mito_buffer)
                center_z = np.random.uniform(-soma_z_radius + mito_buffer, soma_z_radius - mito_buffer)
                
                mito_center = np.array([center_x, center_y, center_z])
                
                dist_from_nucleus_center = np.linalg.norm(mito_center)
                
                if dist_from_nucleus_center > nucleus_radius * 1.2 and \
                   abs(center_x) < soma_x_radius - 0.5 and \
                   abs(center_y) < soma_y_radius - 0.5 and \
                   abs(center_z) < soma_z_radius - 0.5:
                    break

            x_m = np.random.normal(0, 0.4, self.num_mitochondria_points_per_mito)
            y_m = np.random.normal(0, 0.2, self.num_mitochondria_points_per_mito)
            z_m = np.random.normal(0, 0.2, self.num_mitochondria_points_per_mito)
            mito_shape = np.vstack([x_m, y_m, z_m]).T
            self.base_mitochondria_points_list.append(mito_shape + mito_center)
            # Ensure a deep copy for each mitochondrion's points
            self.mitochondria_points[i] = np.copy(self.base_mitochondria_points_list[i])

        self.num_active_mitochondria = self.max_mitochondria_count # Reset to full count

        # --- Cytoplasm Particles (representing vesicles, ions, small proteins) ---
        num_target_particles = self.num_cytoplasm_particles
        self.base_cytoplasm_particles = np.empty((num_target_particles, 3)) # Pre-allocate array
        
        # Define the bounding box for initial random generation to be slightly larger than the soma
        max_gen_x = soma_x_radius * 1.2
        max_gen_y = soma_y_radius * 1.2
        max_gen_z = soma_z_radius * 1.2

        count = 0
        while count < num_target_particles:
            # Generate a small batch of candidate particles
            # Generating in small batches is often faster than one-by-one due to NumPy overhead
            batch_size = min(num_target_particles - count, 100) # Process in batches of up to 100
            candidate_particles = np.random.uniform(
                [-max_gen_x, -max_gen_y, -max_gen_z],
                [max_gen_x, max_gen_y, max_gen_z],
                (batch_size, 3)
            )
            
            for p in candidate_particles:
                # Check if within ovoid soma
                is_within_soma = (p[0]**2 / soma_x_radius**2) + \
                                 (p[1]**2 / soma_y_radius**2) + \
                                 (p[2]**2 / soma_z_radius**2) < 1.0
                
                # Check if outside nucleus
                is_outside_nucleus = np.linalg.norm(p) > nucleus_radius
                
                if is_within_soma and is_outside_nucleus:
                    self.base_cytoplasm_particles[count] = p
                    count += 1
                    if count >= num_target_particles:
                        break # All particles collected

        self.cytoplasm_particles = np.copy(self.base_cytoplasm_particles)

        # Reset dynamic properties to default homeostasis values
        # These are the *initial* values, which will then recover towards homeo_ values.
        self.current_scale = self.homeo_scale
        self.membrane_irregularity_factor = self.homeo_membrane_irregularity
        self.particle_speed_factor = self.homeo_particle_speed
        self.num_active_mitochondria = int(self.max_mitochondria_count * self.homeo_active_mitochondria_ratio)

        self.tau_tangle_level = 0.0
        self.amyloid_beta_level = 0.0
        self.synaptic_health = 1.0 

        # Ensure colors are reset to the new, more transparent values when generate_initial_state is called
        self.membrane_color = self.initial_membrane_color # Use stored initial color
        self.nucleus_color = self.initial_nucleus_color
        self.mitochondria_color = self.initial_mitochondria_color
        self.particle_color = self.initial_particle_color

    # --- NEW: Method to reset only dynamic properties, preserving base geometries ---
    def reset_dynamic_properties(self):
        """
        Resets only the dynamic simulation properties to their homeostatic defaults,
        without regenerating the base 3D point data.
        """
        self.current_scale = self.homeo_scale
        self.membrane_irregularity_factor = self.homeo_membrane_irregularity
        self.particle_speed_factor = self.homeo_particle_speed
        self.num_active_mitochondria = int(self.max_mitochondria_count * self.homeo_active_mitochondria_ratio)

        self.tau_tangle_level = 0.0
        self.amyloid_beta_level = 0.0
        self.synaptic_health = 1.0

        # Reset transformed points to base points (scaled by homeo_scale)
        self.membrane_points = self.base_membrane_points * self.homeo_scale
        self.nucleus_points = self.base_nucleus_points * self.homeo_scale
        
        # Ensure deep copies when resetting mitochondria and cytoplasm points
        self.mitochondria_points = [np.copy(b_m) for b_m in self.base_mitochondria_points_list]
        self.cytoplasm_particles = np.copy(self.base_cytoplasm_particles)

        # Reset colors to initial values
        self.membrane_color = self.initial_membrane_color 
        self.nucleus_color = self.initial_nucleus_color
        self.mitochondria_color = self.initial_mitochondria_color
        self.particle_color = self.initial_particle_color

