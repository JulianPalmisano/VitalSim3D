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

class AstrocyteCellData:
    """
    Encapsulates all data related to the astrocyte cell's 3D structure and visual properties.
    This class handles the generation of the initial (homeostasis) astrocyte state,
    including its characteristic star-like morphology with processes.
    """
    def __init__(self):
        # Raw point data for each component (original, untransformed state)
        self.base_membrane_points = None # Central soma membrane
        self.base_nucleus_points = None
        self.base_mitochondria_points_list = []
        self.base_cytoplasm_particles = None
        self.base_process_particles = None # For radiating astrocytic processes

        # Current (transformed) point data for rendering.
        self.membrane_points = None
        self.nucleus_points = None
        self.mitochondria_points = []
        self.cytoplasm_particles = None
        self.process_particles = None

        # Visual properties (colors, sizes, and LABELS for the legend)
        self.membrane_color = (0.7, 0.5, 0.9, 0.2) # RGBA: Soft purple, transparent
        self.membrane_point_size = 3
        self.membrane_label = "Astrocyte Soma Membrane"

        self.nucleus_color = (0.5, 0.2, 0.7, 0.8) # RGBA: Darker purple
        self.nucleus_point_size = 7
        self.nucleus_label = "Astrocyte Nucleus"

        self.mitochondria_color = (0.9, 0.4, 0.1, 0.8) # RGBA: Orange/brown
        self.mitochondria_point_size = 4
        self.mitochondria_label = "Astrocyte Mitochondria"

        self.max_mitochondria_count = 15 # Fewer than neuron, but still important
        self.num_active_mitochondria = self.max_mitochondria_count

        self.particle_color = (0.6, 0.6, 0.6, 0.5) # RGBA: Grayish, internal soma particles
        self.particle_size = 2
        self.particle_label = "Astrocyte Cytoplasm"
        
        self.process_color = (0.7, 0.5, 0.9, 0.3) # RGBA: Similar to membrane, but slightly less opaque
        self.process_particle_size = 2.5
        self.process_label = "Astrocyte Processes"


        # Dynamic properties (initially set, modified by simulation_logic)
        self.current_scale = 1.0
        self.process_density_factor = 1.0 # New dynamic property for astrocytes
        self.cytoplasm_activity_factor = 1.0 # Like particle_speed_factor, but maybe more about "fuzziness"
        
        # --- Homeostatic/Target Values and Recovery Rates ---
        self.homeo_scale = 1.0
        self.homeo_process_density = 1.0
        self.homeo_cytoplasm_activity = 1.0
        self.homeo_active_mitochondria_ratio = 1.0

        self.recovery_rate_scale = 0.015 # Astrocytes might swell/shrink differently
        self.recovery_rate_process_density = 0.08 # How fast processes might grow/shrink back
        self.recovery_rate_cytoplasm_activity = 0.04
        self.recovery_rate_mitochondria = 0.03

        # Store initial colors for recovery
        self.initial_membrane_color = (0.7, 0.5, 0.9, 0.2)
        self.initial_nucleus_color = (0.5, 0.2, 0.7, 0.8)
        self.initial_mitochondria_color = (0.9, 0.4, 0.1, 0.8)
        self.initial_particle_color = (0.6, 0.6, 0.6, 0.5)
        self.initial_process_color = (0.7, 0.5, 0.9, 0.3)

        # --- Astrocyte Specific Properties (for future simulation logic) ---
        self.gfap_level = 0.0 # Glial Fibrillary Acidic Protein, a marker of reactivity
        self.glutamate_uptake_efficiency = 1.0 # Astrocytes regulate glutamate
        self.calcium_activity_level = 0.0 # Astrocytes also have Ca2+ signaling

        # Generate the initial state
        self.generate_initial_state()

    def generate_initial_state(self):
        """
        Generates the base 3D points for all astrocyte components in their
        homeostasis (initial) state, including its characteristic processes.
        """
        # --- POINT COUNT ADJUSTMENTS ---
        self.num_soma_membrane_points = 1500
        self.num_nucleus_points = 500
        self.num_mitochondria_points_per_mito = 80
        self.num_cytoplasm_particles = 700
        self.num_process_particles = 3000 # More particles for processes due to their extent

        # --- Soma Membrane (roughly spherical/ovoid) ---
        soma_radius = 2.5 # Astrocytes have smaller somas than neuron soma
        
        points_on_soma = np.random.randn(self.num_soma_membrane_points, 3)
        points_on_soma /= np.linalg.norm(points_on_soma, axis=1)[:, np.newaxis]
        self.base_membrane_points = points_on_soma * soma_radius
        
        # Add some initial slight irregularity
        initial_noise_soma = np.random.normal(0, 0.2, self.base_membrane_points.shape)
        self.base_membrane_points += initial_noise_soma
        self.membrane_points = np.copy(self.base_membrane_points)

        # --- Nucleus ---
        nucleus_radius = 1.0
        points_on_sphere_nucleus = np.random.randn(self.num_nucleus_points, 3)
        points_on_sphere_nucleus /= np.linalg.norm(points_on_sphere_nucleus, axis=1)[:, np.newaxis]
        self.base_nucleus_points = points_on_sphere_nucleus * nucleus_radius
        self.nucleus_points = np.copy(self.base_nucleus_points)

        # --- Mitochondria ---
        self.base_mitochondria_points_list = []
        self.mitochondria_points = [None] * self.max_mitochondria_count

        for i in range(self.max_mitochondria_count):
            while True:
                mito_buffer = 0.8
                center_x = np.random.uniform(-soma_radius + mito_buffer, soma_radius - mito_buffer)
                center_y = np.random.uniform(-soma_radius + mito_buffer, soma_radius - mito_buffer)
                center_z = np.random.uniform(-soma_radius + mito_buffer, soma_radius - mito_buffer)
                
                mito_center = np.array([center_x, center_y, center_z])
                
                dist_from_nucleus_center = np.linalg.norm(mito_center)
                
                if dist_from_nucleus_center > nucleus_radius * 1.2 and \
                   np.linalg.norm(mito_center) < soma_radius - 0.2: # Ensure within soma, not too close to edge
                    break

            x_m = np.random.normal(0, 0.3, self.num_mitochondria_points_per_mito)
            y_m = np.random.normal(0, 0.15, self.num_mitochondria_points_per_mito)
            z_m = np.random.normal(0, 0.15, self.num_mitochondria_points_per_mito)
            mito_shape = np.vstack([x_m, y_m, z_m]).T # Elongated shape
            self.base_mitochondria_points_list.append(mito_shape + mito_center)
            self.mitochondria_points[i] = np.copy(self.base_mitochondria_points_list[i])

        self.num_active_mitochondria = self.max_mitochondria_count # Reset to full count

        # --- Cytoplasm Particles (within soma) ---
        num_target_particles_soma = self.num_cytoplasm_particles
        self.base_cytoplasm_particles = np.empty((num_target_particles_soma, 3))
        
        count = 0
        while count < num_target_particles_soma:
            batch_size = min(num_target_particles_soma - count, 50)
            candidate_particles = np.random.uniform(
                [-soma_radius, -soma_radius, -soma_radius],
                [soma_radius, soma_radius, soma_radius],
                (batch_size, 3)
            )
            
            for p in candidate_particles:
                is_within_soma = np.linalg.norm(p) < soma_radius
                is_outside_nucleus = np.linalg.norm(p) > nucleus_radius
                
                if is_within_soma and is_outside_nucleus:
                    self.base_cytoplasm_particles[count] = p
                    count += 1
                    if count >= num_target_particles_soma:
                        break
        self.cytoplasm_particles = np.copy(self.base_cytoplasm_particles)


        # --- Astrocyte Processes (radiating particles) ---
        # The key defining feature of an astrocyte visually
        num_target_process_particles = self.num_process_particles
        self.base_process_particles = np.empty((num_target_process_particles, 3))

        # Define an outer boundary for processes
        process_outer_radius = 10.0 # Astrocytes can have very large domains, covering synapses and blood vessels
        process_inner_radius = soma_radius * 1.2 # Processes start slightly outside the main soma to avoid overlap

        count = 0
        while count < num_target_process_particles:
            batch_size = min(num_target_process_particles - count, 100)
            # Generate points within a spherical shell
            candidate_particles = np.random.uniform(
                [-process_outer_radius, -process_outer_radius, -process_outer_radius],
                [process_outer_radius, process_outer_radius, process_outer_radius],
                (batch_size, 3)
            )

            for p in candidate_particles:
                dist = np.linalg.norm(p)
                # Particles must be outside the main soma's immediate vicinity, but within the process domain
                if dist >= process_inner_radius and dist <= process_outer_radius:
                    # To create a more "spiky" or "arm-like" appearance,
                    # we can bias points towards certain radial directions or
                    # generate them along predefined "rays".
                    # For simplicity and robust random generation: generate points
                    # in a wider sphere and scale them radially to emphasize outer reach
                    
                    # A slight radial bias: points are pushed outwards from center, but with noise
                    radial_factor = np.random.uniform(process_inner_radius, process_outer_radius) / dist
                    p_biased = p * radial_factor + np.random.normal(0, 0.5, 3) # Add some noise
                    
                    # Ensure it's still within the intended bounds after bias
                    if np.linalg.norm(p_biased) <= process_outer_radius:
                        self.base_process_particles[count] = p_biased
                        count += 1
                        if count >= num_target_process_particles:
                            break
        self.process_particles = np.copy(self.base_process_particles)


        # Reset dynamic properties to default homeostasis values
        self.current_scale = self.homeo_scale
        self.process_density_factor = self.homeo_process_density
        self.cytoplasm_activity_factor = self.homeo_cytoplasm_activity
        self.num_active_mitochondria = int(self.max_mitochondria_count * self.homeo_active_mitochondria_ratio)

        self.gfap_level = 0.0
        self.glutamate_uptake_efficiency = 1.0
        self.calcium_activity_level = 0.0

        # Ensure colors are reset
        self.membrane_color = self.initial_membrane_color
        self.nucleus_color = self.initial_nucleus_color
        self.mitochondria_color = self.initial_mitochondria_color
        self.particle_color = self.initial_particle_color
        self.process_color = self.initial_process_color


    def reset_dynamic_properties(self):
        """
        Resets only the dynamic simulation properties of the astrocyte to their homeostatic defaults,
        without regenerating the base 3D point data.
        """
        self.current_scale = self.homeo_scale
        self.process_density_factor = self.homeo_process_density
        self.cytoplasm_activity_factor = self.homeo_cytoplasm_activity
        self.num_active_mitochondria = int(self.max_mitochondria_count * self.homeo_active_mitochondria_ratio)

        self.gfap_level = 0.0
        self.glutamate_uptake_efficiency = 1.0
        self.calcium_activity_level = 0.0

        # Reset transformed points to base points (scaled by homeo_scale)
        # Apply current_scale to all components, and process_density_factor to processes
        self.membrane_points = self.base_membrane_points * self.current_scale
        self.nucleus_points = self.base_nucleus_points * self.current_scale
        
        self.mitochondria_points = [np.copy(b_m) for b_m in self.base_mitochondria_points_list]
        self.cytoplasm_particles = np.copy(self.base_cytoplasm_particles)
        
        # Apply process_density_factor to processes
        self.process_particles = np.copy(self.base_process_particles) * self.process_density_factor * self.current_scale # Also affected by overall scale

        # Reset colors to initial values
        self.membrane_color = self.initial_membrane_color
        self.nucleus_color = self.initial_nucleus_color
        self.mitochondria_color = self.initial_mitochondria_color
        self.particle_color = self.initial_particle_color
        self.process_color = self.initial_process_color
