import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSizePolicy, QFrame, QComboBox, QSpacerItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

import pyqtgraph.opengl as gl
import pyqtgraph as pg

# Import both NeuralCellData and AstrocyteCellData
from cell_model import NeuralCellData, AstrocyteCellData
from simulation_logic import apply_compound_effects
from drug_manager import DrugManager

class VitalSimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VitalSim3D: Interactive Cell Dynamics")
        self.setGeometry(100, 100, 1400, 800)

        self.simulation_running = False
        self.current_time = 0.0
        self.time_step = 0.05
        self.animation_interval_ms = 50

        # Initialize with NeuralCellData as default
        self.current_cell_type_name = "Neuron"
        self.cell_data = NeuralCellData()
        self.drug_manager = DrugManager()

        self.growth_factor_level = 0.0
        self.toxin_level = 0.0
        self.metabolic_stimulator_level = 0.0

        self.active_drug = "None"
        self.drug_dosage = 0.0
        
        self.growth_slider_widget = None
        self.toxin_slider_widget = None
        self.metabolic_slider_widget = None
        self.speed_slider_widget = None
        self.drug_selection_combo = None
        self.drug_dosage_slider_widget = None
        self.cell_type_combo = None # Reference to the new cell type dropdown

        self.setup_ui()

        # Initialize all possible GL items. We will show/hide or set data based on cell type.
        self.membrane_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.membrane_point_size)
        self.nucleus_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.nucleus_point_size)
        
        self.mitochondria_items = []
        # Pre-create enough mito items for the maximum possible count (currently from NeuralCellData)
        max_mito_count = max(NeuralCellData().max_mitochondria_count, AstrocyteCellData().max_mitochondria_count)
        for i in range(max_mito_count):
            mito_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=5) # Default size
            self.mitochondria_items.append(mito_item)
            
        self.cytoplasm_particles_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=self.cell_data.particle_size)
        
        # NEW: Astrocyte specific item
        self.process_particles_item = gl.GLScatterPlotItem(pos=np.empty((0,3)), color=(0,0,0,0), size=3) # Placeholder for astrocyte processes

        # Add all items to the GL widget
        self.gl_widget.addItem(self.membrane_item)
        self.gl_widget.addItem(self.nucleus_item)
        self.gl_widget.addItem(self.cytoplasm_particles_item)
        self.gl_widget.addItem(self.process_particles_item) # Add the new item
        for item in self.mitochondria_items:
            self.gl_widget.addItem(item)

        self.timer = QTimer(self)
        self.timer.setInterval(self.animation_interval_ms)
        self.timer.timeout.connect(self.update_simulation_step)

        # Initial setup: Call reset_simulation, which will call update_visualization
        self.reset_simulation()
        self.update_legend() # Ensure legend is correct initially

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #1e1e1e;")

        main_layout = QHBoxLayout(central_widget)

        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.opts['distance'] = 25
        self.gl_widget.opts['elevation'] = 30
        self.gl_widget.opts['azimuth'] = 45
        self.gl_widget.setBackgroundColor('#202020')
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
                background-color: rgba(30, 30, 30, 200); /* Darker, more opaque */
                border: 1px solid rgba(60, 60, 60, 220); /* Darker border */
                border-radius: 8px;
            }
            QLabel {
                color: #e0e0e0; /* Lighter text for contrast */
                font-size: 12px;
                padding: 2px 5px;
            }
            QLabel.title {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 5px;
                color: #ffffff; /* Pure white for title */
            }
        """)
        self.legend_content_layout = QVBoxLayout(self.legend_frame) # Store reference to content layout
        self.legend_content_layout.setContentsMargins(10, 10, 10, 10)
        self.legend_content_layout.setSpacing(5)

        overlay_layout.addWidget(self.legend_frame)
        overlay_layout.addStretch(1)

        main_layout.addWidget(self.gl_widget, 3)

        # --- Control Panel (Right Side) ---
        control_panel_widget = QWidget()
        control_panel_widget.setStyleSheet("background-color: #282828; border-radius: 10px;")
        control_panel_layout = QVBoxLayout(control_panel_widget)
        control_panel_layout.setContentsMargins(20, 20, 20, 20)
        control_panel_layout.setSpacing(18)

        title_label = QLabel("VitalSim3D Controls")
        title_label.setStyleSheet("font-size: 26px; font-weight: bold; color: #ffffff; margin-bottom: 15px;")
        title_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(title_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #444444; background-color: #444444;")
        control_panel_layout.addWidget(line)

        # NEW: Cell Type Selection Dropdown
        cell_type_layout = QHBoxLayout()
        cell_type_label = QLabel("Select Cell Type:")
        cell_type_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        cell_type_layout.addWidget(cell_type_label)

        self.cell_type_combo = QComboBox()
        self.cell_type_combo.addItems(["Neuron", "Astrocyte"])
        self.cell_type_combo.setCurrentText(self.current_cell_type_name) # Set initial selection
        self.cell_type_combo.currentIndexChanged.connect(self.switch_cell_type)
        self.cell_type_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 15px;
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #3a3a3a;
                color: #ffffff;
                selection-background-color: #666666;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAFCAYAAACNbyblAAAAAXNSR0IArs4c6QAAADhJREFUCB1jYGBg+M8ABWggoAYMghMGBgYQBBgAkgEwMBgYGBgQEAAUGBgYYB/Q/wEYGBhBGPcAADX8Bv3p8K/gAAAAAElFTkSuQmCC);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: #ffffff;
                selection-background-color: #666666;
            }
        """)
        cell_type_layout.addWidget(self.cell_type_combo)
        control_panel_layout.addLayout(cell_type_layout)

        # Separator for Cell Type vs. Simulation Parameters
        cell_type_line = QFrame()
        cell_type_line.setFrameShape(QFrame.HLine)
        cell_type_line.setFrameShadow(QFrame.Sunken)
        cell_type_line.setStyleSheet("color: #444444; background-color: #444444;")
        control_panel_layout.addWidget(cell_type_line)

        # Existing sliders
        slider_layout, self.growth_slider_widget = self._create_slider("Growth Factor Intensity:", self.set_growth_factor_level)
        control_panel_layout.addLayout(slider_layout)

        slider_layout, self.toxin_slider_widget = self._create_slider("Toxin Intensity:", self.set_toxin_level)
        control_panel_layout.addLayout(slider_layout)

        slider_layout, self.metabolic_slider_widget = self._create_slider("Metabolic Stimulator Intensity:", self.set_metabolic_stimulator_level)
        control_panel_layout.addLayout(slider_layout)

        slider_layout, self.speed_slider_widget = self._create_slider("Animation Speed:", self.set_animation_speed, min_val=10, max_val=200, inverted=True)
        control_panel_layout.addLayout(slider_layout)

        # Drug Control Panel Section
        drug_section_label = QLabel("Drug Administration")
        drug_section_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; margin-top: 20px;")
        control_panel_layout.addWidget(drug_section_label)

        drug_line = QFrame()
        drug_line.setFrameShape(QFrame.HLine)
        drug_line.setFrameShadow(QFrame.Sunken)
        drug_line.setStyleSheet("color: #444444; background-color: #444444;")
        control_panel_layout.addWidget(drug_line)

        # Drug selection dropdown
        drug_select_layout = QHBoxLayout()
        drug_select_label = QLabel("Select Drug:")
        drug_select_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e0e0e0;")
        drug_select_layout.addWidget(drug_select_label)

        self.drug_selection_combo = QComboBox()
        self.drug_selection_combo.addItems(list(self.drug_manager.available_drugs.keys()))
        self.drug_selection_combo.currentIndexChanged.connect(self.set_active_drug)
        self.drug_selection_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 15px;
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #3a3a3a;
                color: #ffffff;
                selection-background-color: #666666;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox::down-arrow {
                image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAFCAYAAACNbyblAAAAAXNSR0IArs4c6QAAADhJREFUCB1jYGBg+M8ABWggoAYMghMGBgYQBBgAkgEwMBgYGBgQEAAUGBgYYB/Q/wEYGBhBGPcAADX8Bv3p8K/gAAAAAElFTkSuQmCC);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: #ffffff;
                selection-background-color: #666666;
            }
        """)
        drug_select_layout.addWidget(self.drug_selection_combo)
        control_panel_layout.addLayout(drug_select_layout)

        drug_slider_layout, self.drug_dosage_slider_widget = self._create_slider("Drug Dosage:", self.set_drug_dosage)
        control_panel_layout.addLayout(drug_slider_layout)
        self.drug_dosage_slider_widget.setEnabled(False)

        control_panel_layout.addStretch(1)

        self.start_stop_button = QPushButton("Start Simulation")
        self.start_stop_button.clicked.connect(self.toggle_simulation)
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                padding: 14px;
                font-size: 19px;
                background-color: #27ae60; /* Green (kept for accent) */
                color: white;
                border: none;
                border-radius: 9px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        control_panel_layout.addWidget(self.start_stop_button)

        self.reset_button = QPushButton("Reset to Homeostasis")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.reset_button.setStyleSheet("""
            QPushButton {
                padding: 14px;
                font-size: 19px;
                background-color: #c0392b; /* Red (kept for accent) */
                color: white;
                border: none;
                border-radius: 9px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #e74c3c;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        control_panel_layout.addWidget(self.reset_button)

        main_layout.addWidget(control_panel_widget, 1)

    def _create_slider(self, label_text, callback_func, min_val=0, max_val=100, inverted=False):
        layout = QVBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
        layout.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(int(self._get_initial_slider_value(label_text.split(':')[0], min_val, max_val, inverted)))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444444; /* Mid-dark grey border */
                background: #3a3a3a; /* Darker grey groove */
                height: 12px;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #ffffff; /* Light grey handle */
                border: 1px solid #888888; /* Mid-grey handle border */
                width: 20px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #666666; /* A good contrasting grey for the filled part */
                border-radius: 6px;
            }
        """)
        if not inverted:
            slider.valueChanged.connect(lambda val: callback_func(val / 100.0))
        else:
            slider.valueChanged.connect(callback_func)
        layout.addWidget(slider)
        return layout, slider

    def _get_initial_slider_value(self, slider_name, min_val, max_val, inverted):
        # This now needs to be aware of the *current* cell_data if it influences initial values
        # For now, it mostly impacts simulation speed, others are 0 by default.
        if "Drug Dosage" in slider_name:
            return self.drug_dosage * 100
        elif "Growth Factor" in slider_name:
            return self.growth_factor_level * 100
        elif "Toxin" in slider_name:
            return self.toxin_level * 100
        elif "Metabolic Stimulator" in slider_name:
            return self.metabolic_stimulator_level * 100
        elif "Animation Speed" in slider_name:
            return self.animation_interval_ms
        return 0

    # Helper for legend colors
    def _get_color_name(self, r, g, b):
        r, g, b = r * 255, g * 255, b * 255
        if r > 200 and g > 150 and b < 50: return "Orange" # Mitochondria
        if r < 100 and g > 150 and b > 150: return "Cyan"   # Neuron Membrane
        if r < 100 and g < 100 and b > 150: return "Blue"   # Nucleus
        if r > 150 and g < 100 and b > 150: return "Purple" # Astrocyte Membrane/Processes
        if r > 150 and g > 150 and b > 150: return "White"  # Cytoplasm
        return "Other"

    def _create_legend_item_label(self, label_text, color_rgba):
        r, g, b, _ = color_rgba
        rgb_hex = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        color_name = self._get_color_name(r, g, b)
        html_text = (f"<span style='background-color: {rgb_hex}; display: inline-block; "
                     f"width: 12px; height: 12px; border-radius: 3px; vertical-align: middle; "
                     f"margin-right: 5px;'></span>"
                     f"{label_text} (<span style='color: {rgb_hex}; font-weight: bold;'>{color_name}</span>)")
        label = QLabel(html_text)
        return label

    def update_legend(self):
        # Clear existing legend items
        while self.legend_content_layout.count():
            item = self.legend_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add title back
        title_label = QLabel("Cell Components Key")
        title_label.setProperty("class", "title")
        self.legend_content_layout.addWidget(title_label)

        # Add items based on current cell_data
        self.legend_content_layout.addWidget(self._create_legend_item_label(self.cell_data.membrane_label, self.cell_data.membrane_color))
        self.legend_content_layout.addWidget(self._create_legend_item_label(self.cell_data.nucleus_label, self.cell_data.nucleus_color))
        self.legend_content_layout.addWidget(self._create_legend_item_label(self.cell_data.mitochondria_label, self.cell_data.mitochondria_color))
        self.legend_content_layout.addWidget(self._create_legend_item_label(self.cell_data.particle_label, self.cell_data.particle_color))
        
        # Add astrocyte-specific label if it's an astrocyte
        if isinstance(self.cell_data, AstrocyteCellData):
            self.legend_content_layout.addWidget(self._create_legend_item_label(self.cell_data.process_label, self.cell_data.process_color))

        self.legend_content_layout.addStretch(1)


    def switch_cell_type(self, index):
        new_cell_type = self.cell_type_combo.currentText()
        if new_cell_type == self.current_cell_type_name:
            return # No change needed

        self.current_cell_type_name = new_cell_type
        
        # Stop simulation if running
        if self.simulation_running:
            self.toggle_simulation()

        # Instantiate new cell data object
        if new_cell_type == "Neuron":
            self.cell_data = NeuralCellData()
        elif new_cell_type == "Astrocyte":
            self.cell_data = AstrocyteCellData()
        
        # Reset dynamic properties and update visualization for the new cell
        self.reset_simulation()
        self.update_legend() # Update legend for the new cell type

    def set_active_drug(self, index):
        drug_name = self.drug_selection_combo.currentText()
        self.drug_manager.set_active_drug(drug_name)
        self.active_drug = drug_name
        self.drug_dosage_slider_widget.setValue(0)
        self.drug_dosage_slider_widget.setEnabled(drug_name != "None")
        if not self.simulation_running:
            self.update_visualization()

    def set_drug_dosage(self, level):
        self.drug_dosage = level
        self.drug_manager.set_drug_dosage(level)
        if not self.simulation_running:
            self.update_visualization()

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
            self.start_stop_button.setStyleSheet("""
                QPushButton {
                    padding: 14px;
                    font-size: 19px;
                    background-color: #27ae60;
                    color: white;
                    border: none;
                    border-radius: 9px;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                }
                QPushButton:pressed {
                    background-color: #229954;
                }
            """)
        else:
            self.timer.start()
            self.simulation_running = True
            self.start_stop_button.setText("Stop Simulation")
            self.start_stop_button.setStyleSheet("""
                QPushButton {
                    padding: 14px;
                    font-size: 19px;
                    background-color: #e67e22; /* Orange-red for Stop */
                    color: white;
                    border: none;
                    border-radius: 9px;
                    font-weight: bold;
                    letter-spacing: 0.5px;
                }
                QPushButton:hover {
                    background-color: #f39c12;
                }
                QPushButton:pressed {
                    background-color: #d35400;
                }
            """)

    def reset_simulation(self):
        if self.simulation_running:
            self.toggle_simulation() # Stop simulation if running

        self.growth_factor_level = 0.0
        self.toxin_level = 0.0
        self.metabolic_stimulator_level = 0.0
        
        self.active_drug = "None"
        self.drug_dosage = 0.0
        self.drug_manager.set_active_drug("None")
        self.drug_manager.set_drug_dosage(0.0)

        # Reset UI sliders
        self.growth_slider_widget.setValue(0)
        self.toxin_slider_widget.setValue(0)
        self.metabolic_slider_widget.setValue(0)
        self.speed_slider_widget.setValue(self.animation_interval_ms)
        
        self.drug_selection_combo.setCurrentText("None")
        self.drug_dosage_slider_widget.setValue(0)
        self.drug_dosage_slider_widget.setEnabled(False)

        # Reset the current cell_data to its homeostatic properties
        self.cell_data.reset_dynamic_properties()
        self.update_visualization() # Update visualization after reset

    def update_simulation_step(self):
        self.current_time += self.time_step

        self.cell_data = apply_compound_effects(
            self.cell_data,
            self.growth_factor_level,
            self.toxin_level,
            self.metabolic_stimulator_level,
            self.time_step,
            self.drug_manager
        )
        self.update_visualization()

    def update_visualization(self):
        # Update common components
        self.membrane_item.setData(pos=self.cell_data.membrane_points,
                                   color=self.cell_data.membrane_color,
                                   size=self.cell_data.membrane_point_size)

        self.nucleus_item.setData(pos=self.cell_data.nucleus_points,
                                   color=self.cell_data.nucleus_color,
                                   size=self.cell_data.nucleus_point_size)

        for i, item in enumerate(self.mitochondria_items):
            if i < self.cell_data.num_active_mitochondria and i < len(self.cell_data.mitochondria_points):
                # Ensure points array is not empty before setting data
                if self.cell_data.mitochondria_points[i] is not None and self.cell_data.mitochondria_points[i].size > 0:
                    item.setData(pos=self.cell_data.mitochondria_points[i],
                                 color=self.cell_data.mitochondria_color,
                                 size=self.cell_data.mitochondria_point_size)
                else: # Clear data if no points or None
                    item.setData(pos=np.empty((0,3)))
            else: # Clear data for inactive or non-existent mitochondria
                item.setData(pos=np.empty((0,3)))

        self.cytoplasm_particles_item.setData(pos=self.cell_data.cytoplasm_particles,
                                               color=self.cell_data.particle_color,
                                               size=self.cell_data.particle_size)

        # Handle Astrocyte-specific processes
        if isinstance(self.cell_data, AstrocyteCellData):
            self.process_particles_item.setData(pos=self.cell_data.process_particles,
                                                color=self.cell_data.process_color,
                                                size=self.cell_data.process_particle_size)
            self.process_particles_item.setVisible(True) # Ensure visible
        else:
            self.process_particles_item.setData(pos=np.empty((0,3))) # Clear points
            self.process_particles_item.setVisible(False) # Hide item


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VitalSimApp()
    window.show()
    sys.exit(app.exec_())