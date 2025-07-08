import numpy as np
from math import sin, cos, pi

class CellData:
    """
    Encapsulates all data related to the cell's 3D structure and visual properties.
    This class handles the generation of the initial (homeostasis) cell state.
    """
    def __init__(self):
        # Raw point data for each component
        self.base_membrane_points = None
        self.base_nucleus_points = None
        self.base_mitochondria_points_list = [] # List of numpy arrays, one for each mito
        self.base_cytoplasm_particles = None

        # Current (transformed) point data for rendering
        self.membrane_points = None
        self.nucleus_points = None
        self.mitochondria_points = [] # List of numpy arrays
        self.cytoplasm_particles = None

        # Visual properties (colors, sizes)
        self.membrane_color = (0.2, 0.8, 0.2, 0.8) # RGBA: Greenish, semi-transparent
        self.membrane_point_size = 5

        self.nucleus_color = (0.5, 0.5, 0.8, 0.8) # RGBA: Blueish, semi-transparent
        self.nucleus_point_size = 7

        self.mitochondria_color = (0.8, 0.4, 0.2, 0.9) # RGBA: Orangeish
        self.mitochondria_point_size = 6
        self.num_active_mitochondria = 5 # How many mitochondria are currently visible

        self.particle_color = (1.0, 1.0, 0.0, 0.7) # RGBA: Yellow, semi-transparent
        self.particle_size = 3

        # Dynamic properties that will be modified by simulation_logic
        self.current_scale = 1.0
        self.membrane_irregularity_factor = 0.0 # Controls how jagged the membrane is
        self.particle_speed_factor = 1.0

        # Generate the initial state when the object is created
        self.generate_initial_state()

    def generate_initial_state(self):
        """
        Generates the base 3D points for all cell components in their
        homeostasis (initial) state. This is called on init and on reset.
        """
        num_membrane_points = 1500
        num_nucleus_points = 500
        num_mitochondria_points_per_mito = 150
        num_cytoplasm_particles = 700
        max_mitochondria_count = 5 # Max number of mitochondria to generate base data for

        # --- Membrane (deformed sphere/ellipsoid) ---
        # Generate points on a sphere
        phi = np.linspace(0, 2 * pi, num_membrane_points // 2)
        theta = np.linspace(0, pi, num_membrane_points // (2 * 2)) # Half points for theta
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        # Scale to a base size and add some initial random noise for organic look
        self.base_membrane_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T * 5.0
        initial_noise = np.random.normal(0, 0.2, self.base_membrane_points.shape)
        self.base_membrane_points += initial_noise
        self.membrane_points = np.copy(self.base_membrane_points) # Initialize current points

        # --- Nucleus (smaller sphere) ---
        x_n = np.sin(theta) * np.cos(phi)
        y_n = np.sin(theta) * np.sin(phi)
        z_n = np.cos(theta)
        self.base_nucleus_points = np.vstack([x_n.ravel(), y_n.ravel(), z_n.ravel()]).T * 2.0
        self.nucleus_points = np.copy(self.base_nucleus_points) # Initialize current points

        # --- Mitochondria (elongated point clouds) ---
        self.base_mitochondria_points_list = []
        for i in range(max_mitochondria_count):
            center = np.random.uniform(-3, 3, 3) # Random position within cell bounds
            # Elongated shape (e.g., stretched sphere-like point cloud)
            x_m = np.random.normal(0, 0.5, num_mitochondria_points_per_mito)
            y_m = np.random.normal(0, 0.3, num_mitochondria_points_per_mito)
            z_m = np.random.normal(0, 0.3, num_mitochondria_points_per_mito)
            mito_shape = np.vstack([x_m, y_m, z_m]).T
            self.base_mitochondria_points_list.append(mito_shape + center)
        # Initialize current mitochondria points (copy of base)
        self.mitochondria_points = [np.copy(mp) for mp in self.base_mitochondria_points_list]
        self.num_active_mitochondria = max_mitochondria_count # All visible initially

        # --- Cytoplasm Particles (random points within a sphere) ---
        # Generate points in a cube and filter for a sphere to get initial distribution
        initial_particles = np.random.uniform(-4.5, 4.5, (num_cytoplasm_particles * 2, 3)) # Generate more and filter
        distances = np.linalg.norm(initial_particles, axis=1)
        self.base_cytoplasm_particles = initial_particles[distances < 4.5][:num_cytoplasm_particles]
        self.cytoplasm_particles = np.copy(self.base_cytoplasm_particles) # Initialize current points

        # Reset dynamic properties to default homeostasis values
        self.current_scale = 1.0
        self.membrane_irregularity_factor = 0.0
        self.particle_speed_factor = 1.0
        self.membrane_color = (0.2, 0.8, 0.2, 0.8)
        self.nucleus_color = (0.5, 0.5, 0.8, 0.8)
        self.mitochondria_color = (0.8, 0.4, 0.2, 0.9)
        self.particle_color = (1.0, 1.0, 0.0, 0.7)

