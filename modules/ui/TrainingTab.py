# training_tab.py

from PySide6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QLineEdit, QComboBox, QCheckBox, QPushButton
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.DataType import DataType
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.GradientCheckpointingMethod import GradientCheckpointingMethod
from modules.util.enum.LearningRateScaler import LearningRateScaler
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.optimizer_util import change_optimizer
from modules.util.ui.UIState import UIState
from modules.util.ui import components
from modules.util.ui.CollapsibleWidget import CollapsibleWidget


from modules.ui.OffloadingWindow import OffloadingWindow
from modules.ui.OptimizerParamsWindow import OptimizerParamsWindow
from modules.ui.SchedulerParamsWindow import SchedulerParamsWindow
from modules.ui.TimestepDistributionWindow import TimestepDistributionWindow


class TrainingTab(QWidget):

    def __init__(self, train_config: TrainConfig, ui_state: UIState):
        super().__init__()

        self.train_config = train_config
        self.ui_state = ui_state

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_layout.addWidget(self.scroll_area)
        self.scroll_container = QWidget()
        self.scroll_area.setWidget(self.scroll_container)

        # 3-column grid layout
        self.grid_layout = QGridLayout(self.scroll_container)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setColumnStretch(2, 1)

        # For dynamic calls
        self.lr_scheduler_comp = None
        self.lr_scheduler_adv_comp = None

        self.refresh_ui()

    def refresh_ui(self):
        """
        Clears out existing layout items and rebuilds the UI based on model_type.
        """
        # Clear old items
        while self.grid_layout.count() > 0:
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Create 3 QFrames (or QWidgets) for columns
        col0 = QFrame(self.scroll_container)
        col0_layout = QVBoxLayout(col0)
        self.grid_layout.addWidget(col0, 0, 0)

        col1 = QFrame(self.scroll_container)
        col1_layout = QVBoxLayout(col1)
        self.grid_layout.addWidget(col1, 0, 1)

        col2 = QFrame(self.scroll_container)
        col2_layout = QVBoxLayout(col2)
        self.grid_layout.addWidget(col2, 0, 2)

        # Based on the model type, create frames in each column
        if self.train_config.model_type.is_stable_diffusion():
            self.__setup_stable_diffusion_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_stable_diffusion_3():
            self.__setup_stable_diffusion_3_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_stable_diffusion_xl():
            self.__setup_stable_diffusion_xl_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_wuerstchen():
            self.__setup_wuerstchen_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_pixart():
            self.__setup_pixart_alpha_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_flux():
            self.__setup_flux_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_sana():
            self.__setup_sana_ui(col0_layout, col1_layout, col2_layout)
        elif self.train_config.model_type.is_hunyuan_video():
            self.__setup_hunyuan_video_ui(col0_layout, col1_layout, col2_layout)

        col0_layout.addStretch()
        col1_layout.addStretch()
        col2_layout.addStretch()

    # -------------------------------------------------------------------------
    # The specialized UI setups for each model type
    # -------------------------------------------------------------------------
    def __setup_stable_diffusion_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_unet_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_stable_diffusion_3_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_2_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_3_frame(col0_layout, supports_include=True)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_transformer_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_stable_diffusion_xl_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout)
        self.__create_text_encoder_2_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_unet_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_wuerstchen_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_prior_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_pixart_alpha_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_prior_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout, supports_vb_loss=True)

    def __setup_flux_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_2_frame(col0_layout, supports_include=True)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_transformer_frame(col1_layout, supports_guidance_scale=True)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_sana_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_frame(col0_layout)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout)
        self.__create_prior_frame(col1_layout)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    def __setup_hunyuan_video_ui(self, col0_layout, col1_layout, col2_layout):
        self.__create_base_frame(col0_layout)
        self.__create_text_encoder_1_frame(col0_layout, supports_include=True)
        self.__create_text_encoder_2_frame(col0_layout, supports_include=True)
        self.__create_embedding_frame(col0_layout)

        self.__create_base2_frame(col1_layout, video_training_enabled=True)
        self.__create_transformer_frame(col1_layout, supports_guidance_scale=True)
        self.__create_noise_frame(col1_layout)

        self.__create_masked_frame(col2_layout)
        self.__create_loss_frame(col2_layout)

    # -----------------------------------------------------------------------
    # The sub-frame creation methods
    # -----------------------------------------------------------------------
    def __create_base_frame(self, layout):
        wrapper = CollapsibleWidget("(Optimizer)")

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame_layout = QGridLayout(frame)
        row = 0

        # optimizer
        components.label(frame, 0, 0, "Optimizer",
                         tooltip="The type of optimizer")
        components.options_adv(frame, 0, 1, [str(x) for x in list(Optimizer)], self.ui_state, "optimizer.optimizer",
                               command=self.__restore_optimizer_config, adv_command=self.__open_optimizer_params_window)

        # learning rate scheduler
        # Wackiness will ensue when reloading configs if we don't check and clear this first.
        if hasattr(self, "lr_scheduler_comp"):
            delattr(self, "lr_scheduler_comp")
            delattr(self, "lr_scheduler_adv_comp")
        components.label(frame, 1, 0, "Learning Rate Scheduler",
                         tooltip="Learning rate scheduler that automatically changes the learning rate during training")
        _, d = components.options_adv(frame, 1, 1, [str(x) for x in list(LearningRateScheduler)], self.ui_state,
                                      "learning_rate_scheduler", command=self.__restore_scheduler_config,
                                      adv_command=self.__open_scheduler_params_window)
        self.lr_scheduler_comp = d['component']
        self.lr_scheduler_adv_comp = d['button_component']
        # Initial call requires the presence of self.lr_scheduler_adv_comp.
        self.__restore_scheduler_config(self.ui_state.get_var("learning_rate_scheduler").get())

        # learning rate
        components.label(frame, 2, 0, "Learning Rate",
                         tooltip="The base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "learning_rate")

        # learning rate warmup steps
        components.label(frame, 3, 0, "Learning Rate Warmup Steps",
                         tooltip="The number of steps it takes to gradually increase the learning rate from 0 to the specified learning rate. Values >1 are interpeted as a fixed number of steps, values <=1 are intepreted as a percentage of the total training steps (ex. 0.2 = 20% of the total step count)")
        components.entry(frame, 3, 1, self.ui_state, "learning_rate_warmup_steps")

        # learning rate min factor
        components.label(frame, 4, 0, "Learning Rate Min Factor",
                         tooltip="Unit = float. Method = percentage. For a factor of 0.1, the final LR will be 10% of the initial LR. If the initial LR is 1e-4, the final LR will be 1e-5.")
        components.entry(frame, 4, 1, self.ui_state, "learning_rate_min_factor")

        # learning rate cycles
        components.label(frame, 5, 0, "Learning Rate Cycles",
                         tooltip="The number of learning rate cycles. This is only applicable if the learning rate scheduler supports cycles")
        components.entry(frame, 5, 1, self.ui_state, "learning_rate_cycles")

        # epochs
        components.label(frame, 6, 0, "Epochs",
                         tooltip="The number of epochs for a full training run")
        components.entry(frame, 6, 1, self.ui_state, "epochs")

        # batch size
        components.label(frame, 7, 0, "Batch Size",
                         tooltip="The batch size of one training step")
        components.entry(frame, 7, 1, self.ui_state, "batch_size")

        # accumulation steps
        components.label(frame, 8, 0, "Accumulation Steps",
                         tooltip="Number of accumulation steps. Increase this number to trade batch size for training speed")
        components.entry(frame, 8, 1, self.ui_state, "gradient_accumulation_steps")

        # Learning Rate Scaler
        components.label(frame, 9, 0, "Learning Rate Scaler",
                         tooltip="Selects the type of learning rate scaling to use during training. Functionally equated as: LR * SQRT(selection)")
        components.options(frame, 9, 1, [str(x) for x in list(LearningRateScaler)], self.ui_state,
                           "learning_rate_scaler")

        # clip grad norm
        components.label(frame, 10, 0, "Clip Grad Norm",
                         tooltip="Clips the gradient norm. Leave empty to disable gradient clipping.")
        components.entry(frame, 10, 1, self.ui_state, "clip_grad_norm")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_base2_frame(self, layout, video_training_enabled=False):
        wrapper = CollapsibleWidget("(Misc)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame_layout = QGridLayout(frame)
        row = 0

        # ema
        components.label(frame, row, 0, "EMA",
                         tooltip="EMA averages the training progress over many steps, better preserving different concepts in big datasets")
        components.options(frame, row, 1, [str(x) for x in list(EMAMode)], self.ui_state, "ema")
        row += 1

        # ema decay
        components.label(frame, row, 0, "EMA Decay",
                         tooltip="Decay parameter of the EMA model. Higher numbers will average more steps. For datasets of hundreds or thousands of images, set this to 0.9999. For smaller datasets, set it to 0.999 or even 0.998")
        components.entry(frame, row, 1, self.ui_state, "ema_decay")
        row += 1

        # ema update step interval
        components.label(frame, row, 0, "EMA Update Step Interval",
                         tooltip="Number of steps between EMA update steps")
        components.entry(frame, row, 1, self.ui_state, "ema_update_step_interval")
        row += 1

        # gradient checkpointing
        components.label(frame, row, 0, "Gradient checkpointing",
                         tooltip="Enables gradient checkpointing. This reduces memory usage, but increases training time")
        components.options_adv(frame, row, 1, [str(x) for x in list(GradientCheckpointingMethod)], self.ui_state,
                           "gradient_checkpointing", adv_command=self.__open_offloading_window)
        row += 1

        # gradient checkpointing layer offloading
        components.label(frame, row, 0, "Layer offload fraction",
                         tooltip="Enables offloading of individual layers during training to reduce VRAM usage. Increases training time and uses more RAM. Only available if checkpointing is set to CPU_OFFLOADED. values between 0 and 1, 0=disabled")
        components.entry(frame, row, 1, self.ui_state, "layer_offload_fraction")
        row += 1

        # train dtype
        components.label(frame, row, 0, "Train Data Type",
                         tooltip="The mixed precision data type used for training. This can increase training speed, but reduces precision")
        components.options_kv(frame, row, 1, [
            ("float32", DataType.FLOAT_32),
            ("float16", DataType.FLOAT_16),
            ("bfloat16", DataType.BFLOAT_16),
            ("tfloat32", DataType.TFLOAT_32),
        ], self.ui_state, "train_dtype")
        row += 1

        # fallback train dtype
        components.label(frame, row, 0, "Fallback Train Data Type",
                         tooltip="The mixed precision data type used for training stages that don't support float16 data types. This can increase training speed, but reduces precision")
        components.options_kv(frame, row, 1, [
            ("float32", DataType.FLOAT_32),
            ("bfloat16", DataType.BFLOAT_16),
        ], self.ui_state, "fallback_train_dtype")
        row += 1

        # autocast cache
        components.label(frame, row, 0, "Autocast Cache",
                         tooltip="Enables the autocast cache. Disabling this reduces memory usage, but increases training time")
        components.switch(frame, row, 1, self.ui_state, "enable_autocast_cache")
        row += 1

        # resolution
        components.label(frame, row, 0, "Resolution",
                         tooltip="The resolution used for training. Optionally specify multiple resolutions separated by a comma, or a single exact resolution in the format <width>x<height>")
        components.entry(frame, row, 1, self.ui_state, "resolution")
        row += 1

        # frames
        if video_training_enabled:
            components.label(frame, row, 0, "Frames",
                             tooltip="The number of frames used for training.")
            components.entry(frame, row, 1, self.ui_state, "frames")
            row += 1

        # force circular padding
        components.label(frame, row, 0, "Force Circular Padding",
                         tooltip="Enables circular padding for all conv layers to better train seamless images")
        components.switch(frame, row, 1, self.ui_state, "force_circular_padding")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_text_encoder_frame(self, layout):
        wrapper = CollapsibleWidget("(Text Encoder)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        row = 0

        # train text encoder
        components.label(frame, 0, 0, "Train Text Encoder",
                         tooltip="Enables training the text encoder model")
        components.switch(frame, 0, 1, self.ui_state, "text_encoder.train")

        # dropout
        components.label(frame, 1, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder conditioning")
        components.entry(frame, 1, 1, self.ui_state, "text_encoder.dropout_probability")

        # train text encoder epochs
        components.label(frame, 2, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder")
        components.time_entry(frame, 2, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)

        # text encoder learning rate
        components.label(frame, 3, 0, "Text Encoder Learning Rate",
                         tooltip="The learning rate of the text encoder. Overrides the base learning rate")
        components.entry(frame, 3, 1, self.ui_state, "text_encoder.learning_rate")

        # text encoder layer skip (clip skip)
        components.label(frame, 4, 0, "Clip Skip",
                         tooltip="The number of additional clip layers to skip. 0 = the model default")
        components.entry(frame, 4, 1, self.ui_state, "text_encoder_layer_skip")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_text_encoder_1_frame(self, layout, supports_include=False):
        wrapper = CollapsibleWidget("(TextEncoder 1)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        row = 0

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, "Include Text Encoder 1",
                             tooltip="Includes text encoder 1 in the training run")
            components.switch(frame, row, 1, self.ui_state, "text_encoder.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, "Train Text Encoder 1",
                         tooltip="Enables training the text encoder 1 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, "Train Text Encoder 1 Embedding",
                         tooltip="Enables training embeddings for the text encoder 1 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder 1 conditioning")
        components.entry(frame, row, 1, self.ui_state, "text_encoder.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 1")
        components.time_entry(frame, row, 1, self.ui_state, "text_encoder.stop_training_after",
                              "text_encoder.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, "Text Encoder 1 Learning Rate",
                         tooltip="The learning rate of the text encoder 1. Overrides the base learning rate")
        components.entry(frame, row, 1, self.ui_state, "text_encoder.learning_rate")
        row += 1

        # text encoder layer skip (clip skip)
        components.label(frame, row, 0, "Text Encoder 1 Clip Skip",
                         tooltip="The number of additional clip layers to skip. 0 = the model default")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_layer_skip")
        row += 1
        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_text_encoder_2_frame(self, layout, supports_include=False):
        wrapper = CollapsibleWidget("(TextEncoder 2)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        row = 0

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, "Include Text Encoder 2",
                             tooltip="Includes text encoder 2 in the training run")
            components.switch(frame, row, 1, self.ui_state, "text_encoder_2.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, "Train Text Encoder 2",
                         tooltip="Enables training the text encoder 2 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_2.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, "Train Text Encoder 2 Embedding",
                         tooltip="Enables training embeddings for the text encoder 2 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_2.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder 2 conditioning")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_2.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 2")
        components.time_entry(frame, row, 1, self.ui_state, "text_encoder_2.stop_training_after",
                              "text_encoder_2.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, "Text Encoder 2 Learning Rate",
                         tooltip="The learning rate of the text encoder 2. Overrides the base learning rate")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_2.learning_rate")
        row += 1

        # text encoder layer skip (clip skip)
        components.label(frame, row, 0, "Text Encoder 2 Clip Skip",
                         tooltip="The number of additional clip layers to skip. 0 = the model default")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_2_layer_skip")
        row += 1

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_text_encoder_3_frame(self, layout, supports_include=False):
        wrapper = CollapsibleWidget("(TextEncoder 3)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        row = 0

        if supports_include:
            # include text encoder
            components.label(frame, row, 0, "Include Text Encoder 3",
                             tooltip="Includes text encoder 3 in the training run")
            components.switch(frame, row, 1, self.ui_state, "text_encoder_3.include")
            row += 1

        # train text encoder
        components.label(frame, row, 0, "Train Text Encoder 3",
                         tooltip="Enables training the text encoder 3 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_3.train")
        row += 1

        # train text encoder embedding
        components.label(frame, row, 0, "Train Text Encoder 3 Embedding",
                         tooltip="Enables training embeddings for the text encoder 3 model")
        components.switch(frame, row, 1, self.ui_state, "text_encoder_3.train_embedding")
        row += 1

        # dropout
        components.label(frame, row, 0, "Dropout Probability",
                         tooltip="The Probability for dropping the text encoder 3 conditioning")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_3.dropout_probability")
        row += 1

        # train text encoder epochs
        components.label(frame, row, 0, "Stop Training After",
                         tooltip="When to stop training the text encoder 3")
        components.time_entry(frame, row, 1, self.ui_state, "text_encoder_3.stop_training_after",
                              "text_encoder_3.stop_training_after_unit", supports_time_units=False)
        row += 1

        # text encoder learning rate
        components.label(frame, row, 0, "Text Encoder 3 Learning Rate",
                         tooltip="The learning rate of the text encoder 3. Overrides the base learning rate")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_3.learning_rate")
        row += 1

        # text encoder layer skip (clip skip)
        components.label(frame, row, 0, "Text Encoder 3 Clip Skip",
                         tooltip="The number of additional clip layers to skip. 0 = the model default")
        components.entry(frame, row, 1, self.ui_state, "text_encoder_3_layer_skip")
        row += 1

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_embedding_frame(self, layout, supports_include: bool = False):
        wrapper = CollapsibleWidget("(Embedding)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)

        components.label(frame, 0, 0, "Embeddings Learning Rate",
                         tooltip="The learning rate of embeddings. Overrides the base learning rate")
        components.entry(frame, 0, 1, self.ui_state, "embedding_learning_rate")

        components.label(frame, 1, 0, "Preserve Embedding Norm",
                         tooltip="Rescales each trained embedding to the median embedding norm")
        components.switch(frame, 1, 1, self.ui_state, "preserve_embedding_norm")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_unet_frame(self, layout):
        wrapper = CollapsibleWidget("(UNet)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        # train unet
        components.label(frame, 0, 0, "Train UNet",
                         tooltip="Enables training the UNet model")
        components.switch(frame, 0, 1, self.ui_state, "unet.train")

        # train unet epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the UNet")
        components.time_entry(frame, 1, 1, self.ui_state, "unet.stop_training_after", "unet.stop_training_after_unit",
                              supports_time_units=False)

        # unet learning rate
        components.label(frame, 2, 0, "UNet Learning Rate",
                         tooltip="The learning rate of the UNet. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "unet.learning_rate")

        # rescale noise scheduler to zero terminal SNR
        components.label(frame, 3, 0, "Rescale Noise Scheduler",
                         tooltip="Rescales the noise scheduler to a zero terminal signal to noise ratio and switches the model to a v-prediction target")
        components.switch(frame, 3, 1, self.ui_state, "rescale_noise_scheduler_to_zero_terminal_snr")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_prior_frame(self, layout):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        # train prior
        components.label(frame, 0, 0, "Train Prior",
                         tooltip="Enables training the Prior model")
        components.switch(frame, 0, 1, self.ui_state, "prior.train")

        # train prior epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the Prior")
        components.time_entry(frame, 1, 1, self.ui_state, "prior.stop_training_after", "prior.stop_training_after_unit",
                              supports_time_units=False)

        # prior learning rate
        components.label(frame, 2, 0, "Prior Learning Rate",
                         tooltip="The learning rate of the Prior. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "prior.learning_rate")

        layout.addWidget(frame)

    def __create_transformer_frame(self, layout, supports_guidance_scale=False):
        wrapper = CollapsibleWidget("(Transformer)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        row = 0

        # train transformer
        components.label(frame, 0, 0, "Train Transformer",
                         tooltip="Enables training the Transformer model")
        components.switch(frame, 0, 1, self.ui_state, "prior.train")

        # train transformer epochs
        components.label(frame, 1, 0, "Stop Training After",
                         tooltip="When to stop training the Transformer")
        components.time_entry(frame, 1, 1, self.ui_state, "prior.stop_training_after", "prior.stop_training_after_unit",
                              supports_time_units=False)

        # transformer learning rate
        components.label(frame, 2, 0, "Transformer Learning Rate",
                         tooltip="The learning rate of the Transformer. Overrides the base learning rate")
        components.entry(frame, 2, 1, self.ui_state, "prior.learning_rate")

        # transformer learning rate
        components.label(frame, 3, 0, "Force Attention Mask",
                         tooltip="Force enables passing of a text embedding attention mask to the transformer. This can improve training on shorter captions.")
        components.switch(frame, 3, 1, self.ui_state, "prior.attention_mask")

        if supports_guidance_scale:
            # guidance scale
            components.label(frame, 4, 0, "Guidance Scale",
                             tooltip="The guidance scale of guidance distilled models passed to the transformer during training.")
            components.entry(frame, 4, 1, self.ui_state, "prior.guidance_scale")


        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    def __create_noise_frame(self, layout):
        wrapper = CollapsibleWidget("(Noise)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
                # offset noise weight
        components.label(frame, 0, 0, "Offset Noise Weight",
                         tooltip="The weight of offset noise added to each training step")
        components.entry(frame, 0, 1, self.ui_state, "offset_noise_weight")

        # perturbation noise weight
        components.label(frame, 1, 0, "Perturbation Noise Weight",
                         tooltip="The weight of perturbation noise added to each training step")
        components.entry(frame, 1, 1, self.ui_state, "perturbation_noise_weight")

        # timestep distribution
        components.label(frame, 2, 0, "Timestep Distribution",
                         tooltip="Selects the function to sample timesteps during training",
                         wide_tooltip=True)
        components.options_adv(frame, 2, 1, [str(x) for x in list(TimestepDistribution)], self.ui_state, "timestep_distribution",
                               adv_command=self.__open_timestep_distribution_window)

        # min noising strength
        components.label(frame, 3, 0, "Min Noising Strength",
                         tooltip="Specifies the minimum noising strength used during training. This can help to improve composition, but prevents finer details from being trained")
        components.entry(frame, 3, 1, self.ui_state, "min_noising_strength")

        # max noising strength
        components.label(frame, 4, 0, "Max Noising Strength",
                         tooltip="Specifies the maximum noising strength used during training. This can be useful to reduce overfitting, but also reduces the impact of training samples on the overall image composition")
        components.entry(frame, 4, 1, self.ui_state, "max_noising_strength")

        # noising weight
        components.label(frame, 5, 0, "Noising Weight",
                         tooltip="Controls the weight parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(frame, 5, 1, self.ui_state, "noising_weight")

        # noising bias
        components.label(frame, 6, 0, "Noising Bias",
                         tooltip="Controls the bias parameter of the timestep distribution function. Use the preview to see more details.")
        components.entry(frame, 6, 1, self.ui_state, "noising_bias")

        # timestep shift
        components.label(frame, 7, 0, "Timestep Shift",
                         tooltip="Shift the timestep distribution. Use the preview to see more details.")
        components.entry(frame, 7, 1, self.ui_state, "timestep_shift")

        # dynamic timestep shifting
        components.label(frame, 8, 0, "Dynamic Timestep Shifting",
                         tooltip="Dynamically shift the timestep distribution based on resolution. Use the preview to see more details.")
        components.switch(frame, 8, 1, self.ui_state, "dynamic_timestep_shifting")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)


    def __create_masked_frame(self, layout):
        wrapper = CollapsibleWidget("(Masking)")

        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout()
        frame.setLayout(fl)
        
        # Masked Training
        components.label(frame, 0, 0, "Masked Training",
                         tooltip="Masks the training samples to let the model focus on certain parts of the image. When enabled, one mask image is loaded for each training sample.")
        components.switch(frame, 0, 1, self.ui_state, "masked_training")

        # unmasked probability
        components.label(frame, 1, 0, "Unmasked Probability",
                         tooltip="When masked training is enabled, specifies the number of training steps done on unmasked samples")
        components.entry(frame, 1, 1, self.ui_state, "unmasked_probability")

        # unmasked weight
        components.label(frame, 2, 0, "Unmasked Weight",
                         tooltip="When masked training is enabled, specifies the loss weight of areas outside the masked region")
        components.entry(frame, 2, 1, self.ui_state, "unmasked_weight")

        # normalize masked area loss
        components.label(frame, 3, 0, "Normalize Masked Area Loss",
                         tooltip="When masked training is enabled, normalizes the loss for each sample based on the sizes of the masked region")
        components.switch(frame, 3, 1, self.ui_state, "normalize_masked_area_loss")

        wrapper.setWidget(frame)


        layout.addWidget(wrapper)

    def __create_loss_frame(self, layout, supports_vb_loss=False):
        wrapper = CollapsibleWidget("(Loss)")
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        fl = QGridLayout(frame)
        # MSE Strength
        components.label(frame, 0, 0, "MSE Strength",
                         tooltip="Mean Squared Error strength for custom loss settings. MAE + MSE Strengths generally should sum to 1.")
        components.entry(frame, 0, 1, self.ui_state, "mse_strength")

        # MAE Strength
        components.label(frame, 1, 0, "MAE Strength",
                         tooltip="Mean Absolute Error strength for custom loss settings. MAE + MSE Strengths generally should sum to 1.")
        components.entry(frame, 1, 1, self.ui_state, "mae_strength")

        # log-cosh Strength
        components.label(frame, 2, 0, "log-cosh Strength",
                         tooltip="Log - Hyperbolic cosine Error strength for custom loss settings.")
        components.entry(frame, 2, 1, self.ui_state, "log_cosh_strength")

        if supports_vb_loss:
            # VB Strength
            components.label(frame, 3, 0, "VB Strength",
                             tooltip="Variational lower-bound strength for custom loss settings. Should be set to 1 for variational diffusion models")
            components.entry(frame, 3, 1, self.ui_state, "vb_loss_strength")

        # Loss Weight function
        components.label(frame, 4, 0, "Loss Weight Function",
                         tooltip="Choice of loss weight function. Can help the model learn details more accurately.")
        components.options(frame, 4, 1, [str(x) for x in list(LossWeight)], self.ui_state, "loss_weight_fn")

        # Loss weight strength
        components.label(frame, 5, 0, "Gamma",
                         tooltip="Inverse strength of loss weighting. Range: 1-20, only applies to Min SNR and P2.")
        components.entry(frame, 5, 1, self.ui_state, "loss_weight_strength")

        # Loss Scaler
        components.label(frame, 6, 0, "Loss Scaler",
                         tooltip="Selects the type of loss scaling to use during training. Functionally equated as: Loss * selection")
        components.options(frame, 6, 1, [str(x) for x in list(LossScaler)], self.ui_state, "loss_scaler")

        wrapper.setWidget(frame)
        layout.addWidget(wrapper)

    # -----------------------------------------------------------------------
    # Called when user clicks advanced "..." buttons or when we need to open subwindows
    # -----------------------------------------------------------------------
    def __open_optimizer_params_window(self):
        window = OptimizerParamsWindow(self, self.train_config, self.ui_state)
        window.exec()
        

    def __open_scheduler_params_window(self):
        window = SchedulerParamsWindow(self, self.train_config, self.ui_state)
        window.exec()
        

    def __open_timestep_distribution_window(self):
        window = TimestepDistributionWindow(self, self.train_config, self.ui_state)
        window.exec()
        

    def __open_offloading_window(self):
        window = OffloadingWindow(self, self.train_config, self.ui_state)
        window.exec()
        

    # -----------------------------------------------------------------------
    # Restoration logic
    # -----------------------------------------------------------------------
    def __restore_optimizer_config(self, *args):
        optimizer_config = change_optimizer(self.train_config)
        self.ui_state.get_var("optimizer").update(optimizer_config)

    def __restore_scheduler_config(self, variable):
        if not self.lr_scheduler_adv_comp:
            return

        if variable == "CUSTOM":
            self.lr_scheduler_adv_comp.setEnabled(True)
        else:
            self.lr_scheduler_adv_comp.setEnabled(False)
