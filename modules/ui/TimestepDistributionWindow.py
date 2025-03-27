# timestep_distribution_window.py

import random

import torch
import matplotlib
matplotlib.use("QtAgg")  # Use Qt backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QScrollArea, QFrame, QPushButton
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.ui import components
from modules.util.ui.UIState import UIState

# This class replicates your MGDS-based TimestepGenerator logic
class TimestepGenerator:
    def __init__(
        self,
        timestep_distribution: TimestepDistribution,
        min_noising_strength: float,
        max_noising_strength: float,
        noising_weight: float,
        noising_bias: float,
        timestep_shift: float,
        dynamic_timestep_shifting: bool,
        latent_width: int,
        latent_height: int,
    ):
        self.timestep_distribution = timestep_distribution
        self.min_noising_strength = min_noising_strength
        self.max_noising_strength = max_noising_strength
        self.noising_weight = noising_weight
        self.noising_bias = noising_bias
        self.timestep_shift = timestep_shift
        self.dynamic_timestep_shifting = dynamic_timestep_shifting
        self.latent_width = latent_width
        self.latent_height = latent_height

    def generate(self) -> torch.Tensor:
        from modules.util.config.TrainConfig import TrainConfig
        from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin

        # we define a local mixin object
        class NoiseHelper(ModelSetupNoiseMixin):
            pass

        helper = NoiseHelper()
        generator = torch.Generator().manual_seed(random.randint(0, 2**30))

        config = TrainConfig.default_values()
        config.timestep_distribution = self.timestep_distribution
        config.min_noising_strength = self.min_noising_strength
        config.max_noising_strength = self.max_noising_strength
        config.noising_weight = self.noising_weight
        config.noising_bias = self.noising_bias
        config.timestep_shift = self.timestep_shift
        config.dynamic_timestep_shifting = self.dynamic_timestep_shifting

        return helper._get_timestep_discrete(
            num_train_timesteps=1000,
            deterministic=False,
            generator=generator,
            batch_size=1_000_000,
            config=config,
            latent_width=self.latent_width,
            latent_height=self.latent_height,
        )


class TimestepDistributionWindow(QDialog):
    def __init__(
        self,
        parent,
        config: TrainConfig,
        ui_state: UIState,
        *args,
        **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.config = config
        self.ui_state = ui_state

        self.setWindowTitle("Timestep Distribution")
        self.resize(900, 600)
        self.setModal(False)

        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area, 0, 0, 1, 1)

        self.container = QFrame()
        self.container_layout = QGridLayout(self.container)
        self.container_layout.setContentsMargins(5, 5, 5, 5)
        self.container_layout.setSpacing(5)
        self.container.setLayout(self.container_layout)
        self.scroll_area.setWidget(self.container)

        self.ok_button = components.button(self, 1, 0, "ok", command=self.__ok)
        main_layout.addWidget(self.ok_button, 1, 0, alignment=Qt.AlignRight)

        self.__content_frame(self.container)

    def __content_frame(self, master: QFrame):
        # row=0 => Timestep Distribution
        components.label(master, 0, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps.",
                         wide_tooltip=True)
        components.options(
            master, 0, 1,
            [str(x) for x in list(TimestepDistribution)],
            self.ui_state,
            "timestep_distribution"
        )

        # min noising strength
        components.label(master, 1, 0, "Min Noising Strength")
        components.entry(master, 1, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(master, 2, 0, "Max Noising Strength")
        components.entry(master, 2, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(master, 3, 0, "Noising Weight")
        components.entry(master, 3, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(master, 4, 0, "Noising Bias")
        components.entry(master, 4, 1, self.ui_state, "noising_bias")

        # timestep shift
        components.label(master, 5, 0, "Timestep Shift")
        components.entry(master, 5, 1, self.ui_state, "timestep_shift")

        # dynamic timestep shifting
        components.label(master, 6, 0, "Dynamic Timestep Shifting")
        components.switch(master, 6, 1, self.ui_state, "dynamic_timestep_shifting")

        # We'll embed a matplotlib figure in row=0..8 col=2
        # Create the figure
        self.fig = Figure(figsize=(4, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.container_layout.addWidget(self.canvas, 0, 2, 8, 1)

        self.__update_preview_button = components.button(
            master, 8, 2, "Update Preview", command=self.__update_preview
        )

        self.__update_preview()

    def __update_preview(self):
        resolution = random.randint(512, 1024)
        generator = TimestepGenerator(
            timestep_distribution=self.config.timestep_distribution,
            min_noising_strength=self.config.min_noising_strength,
            max_noising_strength=self.config.max_noising_strength,
            noising_weight=self.config.noising_weight,
            noising_bias=self.config.noising_bias,
            timestep_shift=self.config.timestep_shift,
            dynamic_timestep_shifting=self.config.dynamic_timestep_shifting,
            latent_width=resolution // 8,
            latent_height=resolution // 8,
        )

        self.ax.clear()
        data = generator.generate()
        self.ax.hist(data.numpy(), bins=1000, range=(0, 999))
        self.canvas.draw()

    def __ok(self):
        self.close()
