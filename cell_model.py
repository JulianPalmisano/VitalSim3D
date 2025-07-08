import numpy as np
from math import sin, cos, pi

class NeuralCellData: # Renamed CellData to NeuralCellData
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
        # --- ADJUSTED ALPHA FOR TRANSPARENCY & SLIGHTLY REDUCED POINT SIZES ---
        # Neural cells can vary in color; let's go with a subtle reddish-pink for cell body
        # NEW: Membrane - Translucent light green/cyan
        self.membrane_color = (0.3, 0.8, 0.7, 0.15) # RGBA: Soft cyan/green, highly transparent
        self.membrane_point_size = 4
        self.membrane_label = "Neuron Soma Membrane" # Updated label for neural cell

        # Nucleus: prominent in neurons (retained existing good color)
        self.nucleus_color = (0.3, 0.3, 0.8, 0.7) # RGBA: Slightly darker blue, good visibility
        self.nucleus_point_size = 8
        self.nucleus_label = "Nucleus" 

        # Mitochondria: even more crucial and numerous in neurons (retained existing yellow/orange)
        self.mitochondria_color = (1.0, 0.7, 0.0, 0.9) # RGBA: Brighter orange/yellow
        self.mitochondria_point_size = 5 # Slightly smaller to allow more
        self.mitochondria_label = "Mitochondria" 

        # Neural cells have many mitochondria, especially in the soma
        self.max_mitochondria_count = 20 # Significantly increased for neurons
        self.num_active_mitochondria = self.max_mitochondria_count # Start with all active

        # Cytoplasm Particles: can represent vesicles, proteins, neurotransmitters (retained gray, adjusted alpha slightly)
        self.particle_color = (0.8, 0.8, 0.8, 0.4) # RGBA: Off-white/gray, good contrast with new membrane
        self.particle_size = 2.5 # Slightly smaller particles
        self.particle_label = "Cytoplasm & Vesicles" # Updated label

        # Dynamic properties that will be modified by simulation_logic (initially set)
        self.current_scale = 1.0 # Overall scale of the cell soma
        self.membrane_irregularity_factor = 0.0 # For potential shape changes/degeneration
        self.particle_speed_factor = 1.0 # For cytoplasmic streaming or vesicle movement

        # --- Neural Cell Specific Properties (for future simulation logic) ---
        self.tau_tangle_level = 0.0 # 0.0 (healthy) to 1.0 (severe tangles)
        self.amyloid_beta_level = 0.0 # 0.0 (clear) to 1.0 (high plaque load)
        self.synaptic_health = 1.0 # Represents overall health of neuronal connections

        # Generate the initial state when the object is created
        self.generate_initial_state()

    def generate_initial_state(self):
        """
        Generates the base 3D points for all cell components in their
        homeostasis (initial) state. This is called on init and on reset.
        """
        # --- POINT COUNT ADJUSTMENTS FOR OPTIMIZATION ---
        # Adjust counts for a neuron's soma, balancing detail and performance
        self.num_membrane_points = 2500    # Slightly more points for potentially complex soma shape
        self.num_nucleus_points = 800      # Proportionally more nucleus points
        self.num_mitochondria_points_per_mito = 100 # Reduced per mito, but more total mito
        self.num_cytoplasm_particles = 1000 # Increased for neural activity hints

        # --- Membrane (deformed ovoid/soma-like) ---
        # Neurons are not perfect spheres. We'll start with an ovoid base.
        # Let's make it slightly elongated along one axis
        soma_x_radius = 4.0
        soma_y_radius = 5.0
        soma_z_radius = 4.0
        
        points_on_sphere = np.random.randn(self.num_membrane_points, 3)
        points_on_sphere /= np.linalg.norm(points_on_sphere, axis=1)[:, np.newaxis]
        
        # Scale by ovoid radii
        points_on_sphere[:,0] *= soma_x_radius
        points_on_sphere[:,1] *= soma_y_radius
        points_on_sphere[:,2] *= soma_z_radius

        self.base_membrane_points = points_on_sphere
        
        # Add some initial noise to give it a more organic, irregular shape
        initial_noise = np.random.normal(0, 0.4, self.base_membrane_points.shape) # Increased noise
        self.base_membrane_points += initial_noise
        self.membrane_points = np.copy(self.base_membrane_points)

        # --- Nucleus (typically prominent and centrally located in neuron soma) ---
        nucleus_radius = 1.8 # Proportionate to new soma size
        points_on_sphere_nucleus = np.random.randn(self.num_nucleus_points, 3)
        points_on_sphere_nucleus /= np.linalg.norm(points_on_sphere_nucleus, axis=1)[:, np.newaxis]
        self.base_nucleus_points = points_on_sphere_nucleus * nucleus_radius
        self.nucleus_points = np.copy(self.base_nucleus_points)

        # --- Mitochondria (numerous, distributed, potentially elongated) ---
        self.base_mitochondria_points_list = []
        self.mitochondria_points = [None] * self.max_mitochondria_count

        # Define bounds for mitochondria placement (within soma, outside nucleus)
        soma_max_dim = max(soma_x_radius, soma_y_radius, soma_z_radius)
        
        for i in range(self.max_mitochondria_count):
            # Find a random position for each mitochondrion within soma bounds
            # but outside the nucleus, considering nucleus at origin.
            while True:
                mito_buffer = 1.5
                center_x = np.random.uniform(-soma_x_radius + mito_buffer, soma_x_radius - mito_buffer)
                center_y = np.random.uniform(-soma_y_radius + mito_buffer, soma_y_radius - mito_buffer)
                center_z = np.random.uniform(-soma_z_radius + mito_buffer, soma_z_radius - mito_buffer)
                
                mito_center = np.array([center_x, center_y, center_z])
                
                dist_from_nucleus_center = np.linalg.norm(mito_center)
                
                # Ensure mitochondria are somewhat outside nucleus and inside soma
                # Simple check assuming roughly spherical bounds for the sake of placement
                if dist_from_nucleus_center > nucleus_radius * 1.2 and \
                   abs(center_x) < soma_x_radius - 0.5 and \
                   abs(center_y) < soma_y_radius - 0.5 and \
                   abs(center_z) < soma_z_radius - 0.5:
                    break # Valid position found

            # Generate points for this mitochondrion (slightly elongated shape)
            x_m = np.random.normal(0, 0.4, self.num_mitochondria_points_per_mito)
            y_m = np.random.normal(0, 0.2, self.num_mitochondria_points_per_mito)
            z_m = np.random.normal(0, 0.2, self.num_mitochondria_points_per_mito)
            mito_shape = np.vstack([x_m, y_m, z_m]).T
            self.base_mitochondria_points_list.append(mito_shape + mito_center)
            self.mitochondria_points[i] = np.copy(self.base_mitochondria_points_list[i])

        self.num_active_mitochondria = self.max_mitochondria_count

        # --- Cytoplasm Particles (representing vesicles, ions, small proteins) ---
        # Distribute throughout the soma, avoiding the nucleus
        num_target_particles = self.num_cytoplasm_particles
        
        # Generate particles within the ovoid cell body
        particles = np.random.uniform(-soma_max_dim * 1.1, soma_max_dim * 1.1, (num_target_particles * 2, 3)) # Generate more than needed
        
        # Filter points to be within the ovoid soma and outside the nucleus
        valid_particles = []
        for p in particles:
            # Check if within ovoid soma (approximation)
            if (p[0]**2 / soma_x_radius**2) + (p[1]**2 / soma_y_radius**2) + (p[2]**2 / soma_z_radius**2) < 1.0:
                # Check if outside nucleus
                if np.linalg.norm(p) > nucleus_radius:
                    valid_particles.append(p)
            if len(valid_particles) >= num_target_particles:
                break
        
        self.base_cytoplasm_particles = np.array(valid_particles[:num_target_particles])
        
        while len(self.base_cytoplasm_particles) < num_target_particles:
            remaining = num_target_particles - len(self.base_cytoplasm_particles)
            new_particles = np.random.uniform(-soma_max_dim * 1.1, soma_max_dim * 1.1, (remaining * 2, 3))
            
            new_valid_particles = []
            for p in new_particles:
                if (p[0]**2 / soma_x_radius**2) + (p[1]**2 / soma_y_radius**2) + (p[2]**2 / soma_z_radius**2) < 1.0:
                    if np.linalg.norm(p) > nucleus_radius:
                        new_valid_particles.append(p)
                if len(new_valid_particles) >= remaining:
                    break

            if len(new_valid_particles) > 0:
                self.base_cytoplasm_particles = np.vstack([
                    self.base_cytoplasm_particles,
                    np.array(new_valid_particles)
                ])
                # Truncate if we accidentally added too many
                self.base_cytoplasm_particles = self.base_cytoplasm_particles[:num_target_particles]
            else:
                # Fallback to simple random uniform within a cube if ovoid rejection is too high
                # This should ideally not happen with factor * 2 and enough space
                print("Warning: Could not fill all cytoplasm particles within ovoid/nucleus bounds. Using simpler generation.")
                self.base_cytoplasm_particles = np.random.uniform(-soma_max_dim, soma_max_dim, (num_target_particles, 3))
                # Simple exclusion of nucleus for fallback
                distances = np.linalg.norm(self.base_cytoplasm_particles, axis=1)
                self.base_cytoplasm_particles = self.base_cytoplasm_particles[distances > nucleus_radius][:num_target_particles]

        self.cytoplasm_particles = np.copy(self.base_cytoplasm_particles)

        # Reset dynamic properties to default homeostasis values
        self.current_scale = 1.0
        self.membrane_irregularity_factor = 0.0
        self.particle_speed_factor = 1.0
        self.tau_tangle_level = 0.0
        self.amyloid_beta_level = 0.0
        self.synaptic_health = 1.0 # Reset synaptic health

        # Ensure colors are reset to the new, more transparent values when generate_initial_state is called
        # NEW: Reset to the chosen new initial colors
        self.membrane_color = (0.3, 0.8, 0.7, 0.15) 
        self.nucleus_color = (0.3, 0.3, 0.8, 0.7)
        self.mitochondria_color = (1.0, 0.7, 0.0, 0.9)
        self.particle_color = (0.8, 0.8, 0.8, 0.4)