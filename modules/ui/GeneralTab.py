from PySide6.QtWidgets import QScrollArea, QFrame, QGridLayout, QWidget
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class GeneralTab(QScrollArea):
    def __init__(self, ui_state: UIState):
        super().__init__()
        self.ui_state: UIState = ui_state

        # Configure the scroll area
        self.setWidgetResizable(True)

        # Create the container widget with a grid layout
        container = QFrame()
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(5)
        container.setLayout(container_layout)
        self.setWidget(container)

        #This should be called components.createlabel()
        components.label(container, 0, 0, "Workspace Directory",
                         tooltip="The directory where all files of this training run are saved")
        components.dir_entry(container, 0, 1, self.ui_state, "workspace_dir")

        components.label(container, 1, 0, "Cache Directory",
                         tooltip="The directory where cached data is saved")
        components.dir_entry(container, 1, 1, self.ui_state, "cache_dir")

        components.label(container, 2, 0, "Continue from last backup",
                         tooltip="Automatically continues training from the last backup saved in <workspace>/backup")
        components.switch(container, 2, 1, self.ui_state, "continue_last_backup")

        components.label(container, 3, 0, "Only Cache",
                         tooltip="Only populate the cache, without any training")
        components.switch(container, 3, 1, self.ui_state, "only_cache")

        components.label(container, 4, 0, "Debug mode",
                         tooltip="Save debug information during the training into the debug directory")
        components.switch(container, 4, 1, self.ui_state, "debug_mode")

        components.label(container, 5, 0, "Debug Directory",
                         tooltip="The directory where debug data is saved")
        components.dir_entry(container, 5, 1, self.ui_state, "debug_dir")

        components.label(container, 6, 0, "Tensorboard",
                         tooltip="Starts the Tensorboard Web UI during training")
        components.switch(container, 6, 1, self.ui_state, "tensorboard")

        components.label(container, 7, 0, "Expose Tensorboard",
                         tooltip="Exposes Tensorboard Web UI to all network interfaces (makes it accessible from the network)")
        components.switch(container, 7, 1, self.ui_state, "tensorboard_expose")
        components.label(container, 7, 2, "Tensorboard Port",
                         tooltip="Port to use for Tensorboard link")
        components.entry(container, 7, 3, self.ui_state, "tensorboard_port")

        components.label(container, 8, 0, "Validation",
                         tooltip="Enable validation steps and add new graph in tensorboard")
        components.switch(container, 8, 1, self.ui_state, "validation")

        components.label(container, 9, 0, "Validate after",
                         tooltip="The interval used when validate training")
        components.time_entry(container, 9, 1, self.ui_state, "validate_after", "validate_after_unit")

        components.label(container, 10, 0, "Dataloader Threads",
                         tooltip="Number of threads used for the data loader. Increase if your GPU has room during caching, decrease if it's going out of memory during caching.")
        components.entry(container, 10, 1, self.ui_state, "dataloader_threads")

        components.label(container, 11, 0, "Train Device",
                         tooltip='The device used for training. E.g. "cuda", "cuda:0", etc.')
        components.entry(container, 11, 1, self.ui_state, "train_device")

        components.label(container, 12, 0, "Temp Device",
                         tooltip='The device used to temporarily offload models while they are not used. Default: "cpu"')
        components.entry(container, 12, 1, self.ui_state, "temp_device")
