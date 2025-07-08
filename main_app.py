import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

import pyqtgraph.opengl as gl
import pyqtgraph as pg

# --- CHANGE 1: Import NeuralCellData instead of CellData ---
from cell_model import NeuralCellData 
from simulation_logic import apply_compound_effects

class VitalSimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VitalSim: Interactive Cell Dynamics")
        self.setGeometry(100, 100, 1200, 800)

        self.simulation_running = False
        self.current_time = 0.0
        self.time_step = 0.05
        self.animation_interval_ms = 50

        # --- CHANGE 2: Instantiate NeuralCellData instead of CellData ---
        self.cell_data = NeuralCellData() 

        self.growth_factor_level = 0.0
        self.toxin_level = 0.0
        self.metabolic_stimulator_level = 0.0

        self.growth_slider_widget = None
        self.toxin_slider_widget = None
        self.metabolic_slider_widget = None
        self.speed_slider_widget = None

        self.setup_ui()

        # --- Initialize Cell Visualization Items ---
        self.membrane_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.membrane_point_size)
        self.nucleus_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.nucleus_point_size)
        
        self.mitochondria_items = []
        for i in range(self.cell_data.max_mitochondria_count):
            mito_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.mitochondria_point_size)
            self.mitochondria_items.append(mito_item)
            
        self.cytoplasm_particles_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.particle_size)
        
        # Add all items to the GLViewWidget
        self.gl_widget.addItem(self.membrane_item)
        self.gl_widget.addItem(self.nucleus_item)
        self.gl_widget.addItem(self.cytoplasm_particles_item)

        for item in self.mitochondria_items:
            self.gl_widget.addItem(item)

        self.timer = QTimer(self)
        self.timer.setInterval(self.animation_interval_ms)
        self.timer.timeout.connect(self.update_simulation_step)

        self.reset_simulation()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # --- 3D Visualization Area (Left Side) ---
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.opts['distance'] = 25
        self.gl_widget.opts['elevation'] = 30
        self.gl_widget.opts['azimuth'] = 45
        self.gl_widget.setBackgroundColor('#202020') # Dark background for good contrast
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid = gl.GLGridItem()
        grid.scale(2,2,2)
        self.gl_widget.addItem(grid)

        overlay_widget = QWidget(self.gl_widget) 
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        overlay_layout.setContentsMargins(10, 10, 10, 10)

        self.legend_frame = QFrame(overlay_widget)
        self.legend_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(40, 40, 40, 180);
                border: 1px solid rgba(80, 80, 80, 200);
                border-radius: 8px;
            }
            QLabel {
                color: white;
                font-size: 12px;
                padding: 2px 5px;
            }
            QLabel.title {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 5px;
            }
        """)
        legend_content_layout = QVBoxLayout(self.legend_frame)
        legend_content_layout.setContentsMargins(10, 10, 10, 10)
        legend_content_layout.setSpacing(5)

        title_label = QLabel("Cell Components Key")
        title_label.setProperty("class", "title")
        legend_content_layout.addWidget(title_label)

        # --- MODIFIED: Helper function to get a readable color name ---
        def get_color_name(r, g, b):
            # Normalizing to 0-1 range if inputs are 0-255, though your current code uses 0-1
            # threshold = 0.7 # A stricter threshold for "dominant"
            
            # Check for specific colors first, especially yellow/orange
            # Yellow: High Red and High Green, Low Blue
            if r > 0.8 and g > 0.6 and b < 0.2: # Tuned for (1.0, 0.7, 0.0) -> Yellow/Orange
                return "Yellow" 
            # Cyan: High Green and High Blue, Low Red
            elif g > 0.7 and b > 0.6 and r < 0.4: # Tuned for (0.3, 0.8, 0.7) -> Cyan/Light Green
                return "Cyan"
            # Blue: High Blue, relatively low Red and Green
            elif b > 0.6 and r < 0.5 and g < 0.5: # Tuned for (0.3, 0.3, 0.8) -> Blue
                return "Blue"
            # Gray: All components roughly equal
            elif abs(r - g) < 0.1 and abs(g - b) < 0.1 and r > 0.5: # Tuned for (0.8, 0.8, 0.8) -> Gray
                return "Gray"
            # Fallback to dominant single color if not caught by specific combinations
            elif r > g and r > b:
                return "Red"
            elif g > r and g > b:
                return "Green"
            elif b > r and b > g:
                return "Blue"
            
            return "Other" # Fallback for unknown colors

        # --- MODIFIED: Helper function to create a colored dot label for the legend ---
        def create_legend_item(label_text, color_rgba):
            r, g, b, _ = color_rgba # Extract RGB, ignore alpha for name
            rgb_hex = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            color_name = get_color_name(r, g, b) # Use the improved function
            
            # HTML to draw a colored square next to the text, and include color name
            html_text = (f"<span style='background-color: {rgb_hex}; display: inline-block; "
                         f"width: 12px; height: 12px; border-radius: 3px; vertical-align: middle; "
                         f"margin-right: 5px;'></span>"
                         f"{label_text} (<span style='color: {rgb_hex}; font-weight: bold;'>{color_name}</span>)")
            
            label = QLabel(html_text)
            return label

        # Add legend items dynamically using the helper function and CellData properties
        legend_content_layout.addWidget(create_legend_item(self.cell_data.membrane_label, self.cell_data.membrane_color))
        legend_content_layout.addWidget(create_legend_item(self.cell_data.nucleus_label, self.cell_data.nucleus_color))
        legend_content_layout.addWidget(create_legend_item(self.cell_data.mitochondria_label, self.cell_data.mitochondria_color))
        legend_content_layout.addWidget(create_legend_item(self.cell_data.particle_label, self.cell_data.particle_color))

        legend_content_layout.addStretch(1)

        overlay_layout.addWidget(self.legend_frame)
        overlay_layout.addStretch(1)

        main_layout.addWidget(self.gl_widget, 3)

        # --- Control Panel (Right Side) ---
        control_panel_layout = QVBoxLayout()
        control_panel_layout.setContentsMargins(15, 15, 15, 15)
        control_panel_layout.setSpacing(15)

        title_label = QLabel("VitalSim Controls")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(title_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        control_panel_layout.addWidget(line)

        slider_layout, self.growth_slider_widget = self._create_slider("Growth Factor Intensity:", self.set_growth_factor_level)
        control_panel_layout.addLayout(slider_layout)

        slider_layout, self.toxin_slider_widget = self._create_slider("Toxin Intensity:", self.set_toxin_level)
        control_panel_layout.addLayout(slider_layout)

        slider_layout, self.metabolic_slider_widget = self._create_slider("Metabolic Stimulator Intensity:", self.set_metabolic_stimulator_level)
        control_panel_layout.addLayout(slider_layout)

        slider_layout, self.speed_slider_widget = self._create_slider("Animation Speed:", self.set_animation_speed, min_val=10, max_val=200, inverted=True)
        control_panel_layout.addLayout(slider_layout)

        control_panel_layout.addStretch(1)

        self.start_stop_button = QPushButton("Start Simulation")
        self.start_stop_button.clicked.connect(self.toggle_simulation)
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                padding: 12px;
                font-size: 18px;
                background-color: #4CAF50; /* Green */
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
        """)
        control_panel_layout.addWidget(self.start_stop_button)

        self.reset_button = QPushButton("Reset to Homeostasis")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.reset_button.setStyleSheet("""
            QPushButton {
                padding: 12px;
                font-size: 18px;
                background-color: #f44336; /* Red */
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c00c00;
            }
        """)
        control_panel_layout.addWidget(self.reset_button)

        main_layout.addLayout(control_panel_layout, 1)

    def _create_slider(self, label_text, callback_func, min_val=0, max_val=100, inverted=False):
        layout = QVBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 14px; font-weight: bold; color: #555;")
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(int(self._get_initial_slider_value(label_text.split(':')[0], min_val, max_val, inverted)))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #ddd;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #fff;
                border: 1px solid #777;
                width: 18px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 9px;
            }
        """)
        if not inverted:
            slider.valueChanged.connect(lambda val: callback_func(val / 100.0))
        else:
            slider.valueChanged.connect(callback_func)
        layout.addWidget(slider)
        return layout, slider

    def _get_initial_slider_value(self, slider_name, min_val, max_val, inverted):
        if "Growth Factor" in slider_name:
            return self.growth_factor_level * 100
        elif "Toxin" in slider_name:
            return self.toxin_level * 100
        elif "Metabolic Stimulator" in slider_name:
            return self.metabolic_stimulator_level * 100
        elif "Animation Speed" in slider_name:
            return self.animation_interval_ms
        return 0

    def set_growth_factor_level(self, level):
        self.growth_factor_level = level
        if not self.simulation_running:
            self.update_visualization()

    def set_toxin_level(self, level):
        self.toxin_level = level
        if not self.simulation_running:
            self.update_visualization()

    def set_metabolic_stimulator_level(self, level):
        self.metabolic_stimulator_level = level
        if not self.simulation_running:
            self.update_visualization()

    def set_animation_speed(self, value_ms):
        self.animation_interval_ms = value_ms
        self.timer.setInterval(self.animation_interval_ms)

    def toggle_simulation(self):
        if self.simulation_running:
            self.timer.stop()
            self.simulation_running = False
            self.start_stop_button.setText("Start Simulation")
            self.start_stop_button.setStyleSheet("padding: 12px; font-size: 18px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; font-weight: bold;")
        else:
            self.timer.start()
            self.simulation_running = True
            self.start_stop_button.setText("Stop Simulation")
            self.start_stop_button.setStyleSheet("padding: 12px; font-size: 18px; background-color: #FF9800; color: white; border: none; border-radius: 8px; font-weight: bold;")

    def reset_simulation(self):
        if self.simulation_running:
            self.timer.stop()
            self.simulation_running = False
            self.start_stop_button.setText("Start Simulation")
            self.start_stop_button.setStyleSheet("padding: 12px; font-size: 18px; background-color: #4CAF50; color: white; border: none; border-radius: 8px; font-weight: bold;")

        self.growth_factor_level = 0.0
        self.toxin_level = 0.0
        self.metabolic_stimulator_level = 0.0

        self.growth_slider_widget.setValue(0)
        self.toxin_slider_widget.setValue(0)
        self.metabolic_slider_widget.setValue(0)
        self.speed_slider_widget.setValue(self.animation_interval_ms)

        self.cell_data.generate_initial_state()
        self.update_visualization()

    def update_simulation_step(self):
        self.current_time += self.time_step

        self.cell_data = apply_compound_effects(
            self.cell_data,
            self.growth_factor_level,
            self.toxin_level,
            self.metabolic_stimulator_level,
            self.time_step
        )
        self.update_visualization()

    def update_visualization(self):
        # Update Membrane
        self.membrane_item.setData(pos=self.cell_data.membrane_points,
                                   color=self.cell_data.membrane_color,
                                   size=self.cell_data.membrane_point_size)

        # Update Nucleus
        self.nucleus_item.setData(pos=self.cell_data.nucleus_points,
                                   color=self.cell_data.nucleus_color,
                                   size=self.cell_data.nucleus_point_size)

        # Update Mitochondria
        for i, item in enumerate(self.mitochondria_items):
            if i < self.cell_data.num_active_mitochondria:
                item.setData(pos=self.cell_data.mitochondria_points[i],
                             color=self.cell_data.mitochondria_color,
                             size=self.cell_data.mitochondria_point_size)
            else:
                item.setData(pos=np.empty((0,3)))

        # Update Cytoplasm Particles
        self.cytoplasm_particles_item.setData(pos=self.cell_data.cytoplasm_particles,
                                               color=self.cell_data.particle_color,
                                               size=self.cell_data.particle_size)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VitalSimApp()
    window.show()
    sys.exit(app.exec_())