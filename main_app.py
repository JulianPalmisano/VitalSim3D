import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSizePolicy, QFrame # QFrame imported correctly now
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph.opengl as gl
import pyqtgraph as pg # Main pyqtgraph import

from cell_model import CellData
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

        self.cell_data = CellData()

        self.growth_factor_level = 0.0
        self.toxin_level = 0.0
        self.metabolic_stimulator_level = 0.0

        # These will now store the actual QSlider objects
        self.growth_slider_widget = None
        self.toxin_slider_widget = None
        self.metabolic_slider_widget = None
        self.speed_slider_widget = None

        self.setup_ui()

        self.membrane_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=5)
        self.nucleus_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=7)
        self.mitochondria_items = []
        for _ in range(5):
            mito_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=6)
            self.mitochondria_items.append(mito_item)
        self.cytoplasm_particles_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=3)

        self.gl_widget.addItem(self.membrane_item)
        self.gl_widget.addItem(self.nucleus_item)
        for item in self.mitochondria_items:
            self.gl_widget.addItem(item)
        self.gl_widget.addItem(self.cytoplasm_particles_item)

        self.timer = QTimer(self)
        self.timer.setInterval(self.animation_interval_ms)
        self.timer.timeout.connect(self.update_simulation_step)

        self.reset_simulation() # This will now work correctly after setup_ui initializes sliders

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

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

        # Store the slider widgets themselves, not just their layouts
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

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.opts['distance'] = 25
        self.gl_widget.opts['elevation'] = 30
        self.gl_widget.opts['azimuth'] = 45
        self.gl_widget.setBackgroundColor('#202020')
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid = gl.GLGridItem()
        grid.scale(2,2,2)
        self.gl_widget.addItem(grid)

        main_layout.addWidget(self.gl_widget, 3)

    def _create_slider(self, label_text, callback_func, min_val=0, max_val=100, inverted=False):
        """Helper method to create a QLabel + QSlider pair in a QVBoxLayout.
           Returns both the layout AND the slider widget.
        """
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
        return layout, slider # <--- NOW RETURNS BOTH THE LAYOUT AND THE SLIDER

    def _get_initial_slider_value(self, slider_name, min_val, max_val, inverted):
        if "Growth Factor" in slider_name:
            return self.growth_factor_level * 100
        elif "Toxin" in slider_name:
            return self.toxin_level * 100
        elif "Metabolic Stimulator" in slider_name:
            return self.metabolic_stimulator_level * 100
        elif "Animation Speed" in slider_name:
            # Animation speed slider is inverted, so max_val is slowest, min_val is fastest.
            # The initial value is self.animation_interval_ms, which is 50.
            # We want to return the slider value that corresponds to 50ms.
            # If min_val=10 (fastest) and max_val=200 (slowest)
            # The slider value *is* the ms interval, so no conversion needed.
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

        # Now setting values directly on the stored QSlider objects
        self.growth_slider_widget.setValue(0)
        self.toxin_slider_widget.setValue(0)
        self.metabolic_slider_widget.setValue(0)
        self.speed_slider_widget.setValue(self.animation_interval_ms) # Resets to default speed

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
        self.membrane_item.setData(pos=self.cell_data.membrane_points,
                                   color=self.cell_data.membrane_color,
                                   size=self.cell_data.membrane_point_size)

        self.nucleus_item.setData(pos=self.cell_data.nucleus_points,
                                  color=self.cell_data.nucleus_color,
                                  size=self.cell_data.nucleus_point_size)

        for i, item in enumerate(self.mitochondria_items):
            if i < len(self.cell_data.mitochondria_points):
                item.setData(pos=self.cell_data.mitochondria_points[i],
                             color=self.cell_data.mitochondria_color,
                             size=self.cell_data.mitochondria_point_size)
            else:
                item.setData(pos=np.empty((0,3)))

        self.cytoplasm_particles_item.setData(pos=self.cell_data.cytoplasm_particles,
                                              color=self.cell_data.particle_color,
                                              size=self.cell_data.particle_size)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VitalSimApp()
    window.show()
    sys.exit(app.exec_())
