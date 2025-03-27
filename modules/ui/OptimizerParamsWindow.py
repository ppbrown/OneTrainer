
# Subwindow for the Training Tab, optimizer section

import contextlib

from PySide6.QtWidgets import (
    QDialog, QGridLayout, QPushButton, QScrollArea, QFrame
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.Optimizer import Optimizer
from modules.util.optimizer_util import (
    OPTIMIZER_DEFAULT_PARAMETERS,
    change_optimizer,
    load_optimizer_defaults,
    update_optimizer_config,
)
from modules.util.ui import components


class OptimizerParamsWindow(QDialog):
    def __init__(
            self,
            parent,
            train_config: TrainConfig,
            ui_state,
            *args,
            **kwargs,
    ):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state
        self.optimizer_ui_state = ui_state.get_var("optimizer")

        self.setWindowTitle("Optimizer Settings")
        self.resize(800, 500)
        self.setModal(False)

        main_layout = QGridLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area, 0, 0, 1, 1)

        self.frame = QFrame()
        self.frame_layout = QGridLayout(self.frame)
        self.frame_layout.setContentsMargins(5, 5, 5, 5)
        self.frame_layout.setSpacing(5)
        self.frame.setLayout(self.frame_layout)
        self.scroll_area.setWidget(self.frame)

        self.ok_button = QPushButton("ok", self)
        self.ok_button.clicked.connect(self.on_window_close)
        main_layout.addWidget(self.ok_button, 1, 0, alignment=Qt.AlignRight)

        self.main_frame(self.frame)

    def main_frame(self, master):
        components.label(master, 0, 0, "Optimizer", tooltip="The type of optimizer")

        components.options(
            master, 0, 1,
            [str(x) for x in list(Optimizer)],
            self.optimizer_ui_state,
            "optimizer",
            command=self.on_optimizer_change
        )

        components.label(master, 0, 3, "Optimizer Defaults", tooltip="Load default settings for the selected optimizer")
        components.button(master, 0, 4, "Load Defaults", self.load_defaults, tooltip="Load default settings")

        self.create_dynamic_ui(master)

    def clear_dynamic_ui(self, master):
        with contextlib.suppress(Exception):
            for widget in master.children():
                info = widget.property("grid_info")
                if info and info.get("row", 0) >= 1:
                    widget.setParent(None)

    def create_dynamic_ui(self, master):
        KEY_DETAIL_MAP = {
            'adam_w_mode': {'title': 'Adam W Mode', 'tooltip': 'Whether to use weight decay correction for Adam optimizer.', 'type': 'bool'},
            'alpha': {'title': 'Alpha', 'tooltip': 'Smoothing parameter for RMSprop and others.', 'type': 'float'},
            'amsgrad': {'title': 'AMSGrad', 'tooltip': 'Whether to use the AMSGrad variant for Adam.', 'type': 'bool'},
            'beta1': {'title': 'Beta1', 'tooltip': 'optimizer_momentum term.', 'type': 'float'},
            'beta2': {'title': 'Beta2', 'tooltip': 'Coefficients for computing running averages of gradient.', 'type': 'float'},
            'beta3': {'title': 'Beta3', 'tooltip': 'Coefficient for computing the Prodigy stepsize.', 'type': 'float'},
            'bias_correction': {'title': 'Bias Correction', 'tooltip': 'Use bias correction in Adam-like optimizers.', 'type': 'bool'},
            'block_wise': {'title': 'Block Wise', 'tooltip': 'Block-wise model update.', 'type': 'bool'},
            'capturable': {'title': 'Capturable', 'tooltip': 'Whether the optimizer can be captured.', 'type': 'bool'},
            'centered': {'title': 'Centered', 'tooltip': 'Center gradient before scaling.', 'type': 'bool'},
            'clip_threshold': {'title': 'Clip Threshold', 'tooltip': 'Clipping value for gradients.', 'type': 'float'},
            'd0': {'title': 'Initial D', 'tooltip': 'Initial D estimate for D-adaptation.', 'type': 'float'},
            'd_coef': {'title': 'D Coefficient', 'tooltip': 'Coefficient in the expression for the estimate of d.', 'type': 'float'},
            'dampening': {'title': 'Dampening', 'tooltip': 'Dampening for optimizer_momentum.', 'type': 'float'},
            'decay_rate': {'title': 'Decay Rate', 'tooltip': 'Rate of decay for moment estimation.', 'type': 'float'},
            'decouple': {'title': 'Decouple', 'tooltip': 'Use AdamW style decoupled weight decay.', 'type': 'bool'},
            'differentiable': {'title': 'Differentiable', 'tooltip': 'Whether the optimization function is differentiable.', 'type': 'bool'},
            'eps': {'title': 'EPS', 'tooltip': 'A small value to prevent division by zero.', 'type': 'float'},
            'eps2': {'title': 'EPS 2', 'tooltip': 'Another small value for numeric stability.', 'type': 'float'},
            'foreach': {'title': 'ForEach', 'tooltip': 'Use a faster foreach implementation if available.', 'type': 'bool'},
            'fsdp_in_use': {'title': 'FSDP in Use', 'tooltip': 'Flag for using sharded parameters.', 'type': 'bool'},
            'fused': {'title': 'Fused', 'tooltip': 'Use a fused implementation if available.', 'type': 'bool'},
            'fused_back_pass': {'title': 'Fused Back Pass', 'tooltip': 'Fuses backprop pass with the optimizer step.', 'type': 'bool'},
            'growth_rate': {'title': 'Growth Rate', 'tooltip': 'Limit for D estimate growth rate.', 'type': 'float'},
            'initial_accumulator_value': {'title': 'Initial Accumulator Value', 'tooltip': 'Initial value for Adagrad.', 'type': 'float'},
            'initial_accumulator': {'title': 'Initial Accumulator', 'tooltip': 'Start value for moment estimates.', 'type': 'float'},
            'is_paged': {'title': 'Is Paged', 'tooltip': 'Use CPU paging for optimizer state.', 'type': 'bool'},
            'log_every': {'title': 'Log Every', 'tooltip': 'Intervals at which logging occurs.', 'type': 'int'},
            'lr_decay': {'title': 'LR Decay', 'tooltip': 'Rate at which LR decreases.', 'type': 'float'},
            'max_unorm': {'title': 'Max Unorm', 'tooltip': 'Max norm for gradient clipping.', 'type': 'float'},
            'maximize': {'title': 'Maximize', 'tooltip': 'Whether to maximize the objective.', 'type': 'bool'},
            'min_8bit_size': {'title': 'Min 8bit Size', 'tooltip': 'Minimum tensor size for 8-bit quantization.', 'type': 'int'},
            'momentum': {'title': 'optimizer_momentum', 'tooltip': 'Factor for accelerating SGD in relevant direction.', 'type': 'float'},
            'nesterov': {'title': 'Nesterov', 'tooltip': 'Enable Nesterov optimizer_momentum.', 'type': 'bool'},
            'no_prox': {'title': 'No Prox', 'tooltip': 'Disable prox updates if True.', 'type': 'bool'},
            'optim_bits': {'title': 'Optim Bits', 'tooltip': 'Number of bits used for optimization.', 'type': 'int'},
            'percentile_clipping': {'title': 'Percentile Clipping', 'tooltip': 'Clip gradient by percentile.', 'type': 'float'},
            'relative_step': {'title': 'Relative Step', 'tooltip': 'Use a relative step size.', 'type': 'bool'},
            'safeguard_warmup': {'title': 'Safeguard Warmup', 'tooltip': 'Avoid issues during warm-up.', 'type': 'bool'},
            'scale_parameter': {'title': 'Scale Parameter', 'tooltip': 'Scale parameter or not.', 'type': 'bool'},
            'stochastic_rounding': {'title': 'Stochastic Rounding', 'tooltip': 'Stochastic rounding for weight updates.', 'type': 'bool'},
            'use_bias_correction': {'title': 'Bias Correction', 'tooltip': 'Turn on Adam\'s bias correction.', 'type': 'bool'},
            'use_triton': {'title': 'Use Triton', 'tooltip': 'Whether Triton optimization is used.', 'type': 'bool'},
            'warmup_init': {'title': 'Warmup Initialization', 'tooltip': 'Whether to warm-up initialization.', 'type': 'bool'},
            'weight_decay': {'title': 'Weight Decay', 'tooltip': 'Regularization term for weights.', 'type': 'float'},
            'weight_lr_power': {'title': 'Weight LR Power', 'tooltip': 'Raise LR to this power for weighting.', 'type': 'float'},
            'decoupled_decay': {'title': 'Decoupled Decay', 'tooltip': 'Use decoupled weight decay (AdamW).', 'type': 'bool'},
            'fixed_decay': {'title': 'Fixed Decay', 'tooltip': 'Fixed weight decay scaling if decoupled.', 'type': 'bool'},
            'rectify': {'title': 'Rectify', 'tooltip': 'Perform the rectified update (RAdam).', 'type': 'bool'},
            'degenerated_to_sgd': {'title': 'Degenerated to SGD', 'tooltip': 'SGD update if gradient variance is high.', 'type': 'bool'},
            'k': {'title': 'K', 'tooltip': 'Number of vector projected per iteration.', 'type': 'int'},
            'xi': {'title': 'Xi', 'tooltip': 'Term used to avoid zero division in vector projections.', 'type': 'float'},
            'n_sma_threshold': {'title': 'N SMA Threshold', 'tooltip': 'Number of SMA threshold.', 'type': 'int'},
            'ams_bound': {'title': 'AMS Bound', 'tooltip': 'Use the AMSBound variant.', 'type': 'bool'},
            'r': {'title': 'R', 'tooltip': 'EMA factor.', 'type': 'float'},
            'adanorm': {'title': 'AdaNorm', 'tooltip': 'Whether to use the AdaNorm variant.', 'type': 'bool'},
            'adam_debias': {'title': 'Adam Debias', 'tooltip': 'Only correct the denominator, ignoring numerator inflation.', 'type': 'bool'},
            'slice_p': {'title': 'Slice parameters', 'tooltip': 'Reduce memory usage by partial vector updates.', 'type': 'int'},
            'cautious': {'title': 'Cautious', 'tooltip': 'Use the cautious variant if True.', 'type': 'bool'},
        }

        selected_optimizer = self.train_config.optimizer.optimizer
        if selected_optimizer not in OPTIMIZER_DEFAULT_PARAMETERS:
            return

        keys = list(OPTIMIZER_DEFAULT_PARAMETERS[selected_optimizer].keys())

        idx = 0
        for key in keys:
            if key not in KEY_DETAIL_MAP:
                continue
            info = KEY_DETAIL_MAP[key]
            title = info['title']
            tooltip = info['tooltip']
            field_type = info['type']

            row = (idx // 2) + 1
            col = 3 * (idx % 2)
            idx += 1

            components.label(master, row, col, title, tooltip=tooltip)
            if field_type == 'bool':
                components.switch(master, row, col + 1, self.optimizer_ui_state, key, command=self.update_user_pref)
            else:
                components.entry(master, row, col + 1, self.optimizer_ui_state, key, command=self.update_user_pref)

    def update_user_pref(self, *args):
        update_optimizer_config(self.train_config)

    def on_optimizer_change(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

        self.clear_dynamic_ui(self.frame)
        self.create_dynamic_ui(self.frame)

    def load_defaults(self, *args):
        optimizer_config = load_optimizer_defaults(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

    def on_window_close(self):
        self.close()
