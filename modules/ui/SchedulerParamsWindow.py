
# These classes get acivated under the Training tab, if you want to set parameters for a
# "Custom" scheduler, instead of the usual LINEAR, COSINE, etc.


from typing import Any, Callable

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QFrame, QPushButton, QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import Qt

from modules.ui.OTConfigFrame import OTConfigFrame

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.ui import components
from modules.util.ui.UIState import UIState


class KvParamsFrame(OTConfigFrame):

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="scheduler_params",
            from_external_file=False,
            add_button_text="add parameter",
            is_full_width=True
        )

    def refresh_ui(self):
        self.__create_element_list()

    # called by super.__add_element()
    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        try:
            w = KvParamsWidget(element, i, open_command, remove_command, clone_command, save_command)
        except Exception as e:
            print(f"KvParamsFrame.create_widget: Error creating KvParamswidget: {e}")
            w = None
        
        return w

    def create_new_element(self) -> dict[str, str]:
        return {"key": "", "value": ""}


    def open_element_window(self, i, ui_state):
        # we dont need a deeper "edit" popup. 
        # But superclass forces us to implement this function
        pass

# class that displays one or more items, in the larger frame
class KvParamsWidget(QFrame):
    def __init__(
            self,
            element, 
            i: int, 
            open_command: Callable[..., None], 
            remove_command: Callable[..., None], 
            clone_command: Callable[..., None], 
            save_command: Callable[..., None]
    ):
        super().__init__()
        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        qglayout = QGridLayout(self)
        qglayout.setContentsMargins(0,0,0,0)
        qglayout.setSpacing(5)

        # "X" close button
        close_button = components.button(self, 0, 0, "X",
                                         command=lambda: remove_command(self.i),
                                         tooltip="Remove this parameter")
                
        # Key
        tooltip_key = "Key name for an argument in your scheduler"
        self.key_entry = components.entry(
            self, 0, 1, self.ui_state, "key", tooltip=tooltip_key, wide_tooltip=True,width=0
        )
        self.key_entry.editingFinished.connect(self.save_command)

        # Value
        tooltip_val = (
            "Value for an argument in your scheduler. Some special values can be used, wrapped in percent signs: "
            "LR, EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS, SCHEDULER_STEPS."
        )
        self.value_entry = components.entry(
            self, 0, 2, self.ui_state, "value", tooltip=tooltip_val,wide_tooltip=True,width=0
        )
        self.value_entry.editingFinished.connect(self.save_command)

        qglayout.setColumnStretch(0,0)
        qglayout.setColumnStretch(1,1) 
        qglayout.setColumnStretch(2,1)


    def place_in_list(self):
        self.layout.setRowMinimumHeight(self.i, 30)
        self.layout.setRowStretch(self.i, 0)
        self.layout.setColumnStretch(2, 1)
        self.grid_layout_parent = self.parentWidget().layout()
        if self.grid_layout_parent:
            self.grid_layout_parent.addWidget(self, self.i, 0, 1, 1)


# This is what gets called from the TrainingTab class
class SchedulerParamsWindow(QDialog):
    def __init__(self, parent, train_config: TrainConfig, ui_state, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

        self.setWindowTitle("Learning Rate Scheduler Settings")
        self.resize(800, 400)
        self.setModal(False)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5,5,5,5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)
        
        self.frame = QFrame(self)
        self.frame_layout = QHBoxLayout(self.frame)
        self.frame_layout.setContentsMargins(5,5,5,5)
        self.frame_layout.setSpacing(5)
        
        if self.train_config.learning_rate_scheduler == LearningRateScheduler.CUSTOM:
            components.label(self.frame, 0, 0, "Class Name",
                             tooltip="Python class module and name for the custom scheduler class.")
            components.entry(self.frame, 0, 1, self.ui_state, "custom_learning_rate_scheduler")
            #self.frame_layout.setStretch(0,0)
            #self.frame_layout.setStretch(1,1)
            self.frame_layout.addStretch()

        main_layout.addWidget(self.frame)

        main_layout.addWidget(KvParamsFrame(self, self.train_config, self.ui_state))

        main_layout.addStretch()

        # "ok" button
        self.ok_button = components.button(self, 1, 0, "ok", self.on_window_close)
        main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)


 

    def on_window_close(self):
        self.close()
