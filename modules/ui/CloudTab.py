
# implements the GUI hooks for training in the cloud

import webbrowser
from PySide6.QtWidgets import (
    QWidget, QScrollArea, QGridLayout, QVBoxLayout, QFrame,
    QLabel, QLineEdit, QCheckBox, QComboBox, QPushButton
)
from PySide6.QtCore import Qt

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync
from modules.util.enum.CloudType import CloudType
from modules.util.ui.UIState import UIState

class CloudTab(QScrollArea):

    def __init__(self, train_config: TrainConfig, ui_state: UIState, parent):
        super().__init__()

        self.train_config = train_config
        self.ui_state = ui_state
        self.parent = parent
        self.reattach = False

        container = QFrame()
        self.grid = QGridLayout(container)
        container.setLayout(self.grid)
        self.setWidget(container)


        self.setWidgetResizable(True)

        self.__build_ui()

    # Fills self.grid with widgets
    def __build_ui(self):
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(3, 1)
        self.grid.setColumnStretch(5, 1)

        # row -> 0
        lbl_enabled = QLabel("Enabled")
        lbl_enabled.setToolTip("Enable cloud training")
        self.grid.addWidget(lbl_enabled, 0, 0)

        self.enabled_switch = QCheckBox()
        self.enabled_switch.setChecked(bool(self.train_config.cloud.enabled))
        self.grid.addWidget(self.enabled_switch, 0, 1)

        # row -> 1
        lbl_type = QLabel("Type")
        lbl_type.setToolTip("Choose LINUX or RUNPOD, etc.")
        self.grid.addWidget(lbl_type, 1, 0)

        self.type_combo = QComboBox()
        # Suppose CloudType.RUNPOD=1, LINUX=2, etc. 
        self.type_combo.addItem("RUNPOD", "RUNPOD")
        self.type_combo.addItem("LINUX", "LINUX")
        # set current index from train_config.cloud.type
        self.grid.addWidget(self.type_combo, 1, 1)

        # row -> 2
        lbl_filesync = QLabel("File sync method")
        lbl_filesync.setToolTip("Choose NATIVE_SCP or FABRIC_SFTP.")
        self.grid.addWidget(lbl_filesync, 2, 0)

        self.filesync_combo = QComboBox()
        self.filesync_combo.addItem("NATIVE_SCP", "NATIVE_SCP")
        self.filesync_combo.addItem("FABRIC_SFTP", "FABRIC_SFTP")
        self.grid.addWidget(self.filesync_combo, 2, 1)

        # row -> 3
        lbl_api = QLabel("API key")
        lbl_api.setToolTip("Cloud service API key for RUNPOD. This value is stored separately.")
        self.grid.addWidget(lbl_api, 3, 0)

        self.api_line = QLineEdit()
        # set text if you want from self.train_config.secrets.cloud.api_key
        self.grid.addWidget(self.api_line, 3, 1)

        # row -> 4
        lbl_host = QLabel("Hostname")
        lbl_host.setToolTip("SSH server hostname or IP. Leave empty to auto-create or use Cloud ID.")
        self.grid.addWidget(lbl_host, 4, 0)

        self.host_line = QLineEdit()
        self.grid.addWidget(self.host_line, 4, 1)

        # row -> 5
        lbl_port = QLabel("Port")
        lbl_port.setToolTip("SSH server port. Leave empty if auto-creating or using Cloud ID.")
        self.grid.addWidget(lbl_port, 5, 0)

        self.port_line = QLineEdit()
        self.grid.addWidget(self.port_line, 5, 1)

        # row -> 6
        lbl_user = QLabel("User")
        lbl_user.setToolTip("SSH username. For RUNPOD, typically 'root'.")
        self.grid.addWidget(lbl_user, 6, 0)

        self.user_line = QLineEdit()
        self.grid.addWidget(self.user_line, 6, 1)

        # row -> 7
        lbl_cloudid = QLabel("Cloud id")
        lbl_cloudid.setToolTip("RUNPOD Cloud ID. Leave empty if auto-creating a new cloud.")
        self.grid.addWidget(lbl_cloudid, 7, 0)

        self.cloudid_line = QLineEdit()
        self.grid.addWidget(self.cloudid_line, 7, 1)

        # row -> 8
        lbl_tboard = QLabel("Tensorboard TCP tunnel")
        lbl_tboard.setToolTip("Make a TCP tunnel to a remote tensorboard instead of local.")
        self.grid.addWidget(lbl_tboard, 8, 0)

        self.tboard_switch = QCheckBox()
        self.grid.addWidget(self.tboard_switch, 8, 1)

        # row -> 1, col -> 2 => "Remote Directory"
        lbl_remotedir = QLabel("Remote Directory")
        lbl_remotedir.setToolTip("Directory on the cloud to upload/download files.")
        self.grid.addWidget(lbl_remotedir, 1, 2)

        self.remotedir_line = QLineEdit()
        self.grid.addWidget(self.remotedir_line, 1, 3)

        # row -> 2, col -> 2 => "OneTrainer Directory"
        lbl_onetrainerdir = QLabel("OneTrainer Directory")
        lbl_onetrainerdir.setToolTip("The directory for OneTrainer on the cloud.")
        self.grid.addWidget(lbl_onetrainerdir, 2, 2)

        self.onetrainer_line = QLineEdit()
        self.grid.addWidget(self.onetrainer_line, 2, 3)

        # row -> 3, col -> 2 => "Huggingface cache Directory"
        lbl_hfcache = QLabel("Huggingface cache Directory")
        lbl_hfcache.setToolTip("Huggingface models are downloaded here.")
        self.grid.addWidget(lbl_hfcache, 3, 2)

        self.hfcache_line = QLineEdit()
        self.grid.addWidget(self.hfcache_line, 3, 3)

        # row -> 4 => "Install OneTrainer" switch
        lbl_install = QLabel("Install OneTrainer")
        lbl_install.setToolTip("Automatically install from GitHub if missing.")
        self.grid.addWidget(lbl_install, 4, 2)

        self.install_switch = QCheckBox()
        self.grid.addWidget(self.install_switch, 4, 3)

        # row -> 5 => "Install command"
        lbl_installcmd = QLabel("Install command")
        lbl_installcmd.setToolTip("The command to install OneTrainer. For dev branches, etc.")
        self.grid.addWidget(lbl_installcmd, 5, 2)

        self.installcmd_line = QLineEdit()
        self.grid.addWidget(self.installcmd_line, 5, 3)

        # row -> 6 => "Update OneTrainer"
        lbl_update = QLabel("Update OneTrainer")
        lbl_update.setToolTip("Update OneTrainer if it exists on the cloud.")
        self.grid.addWidget(lbl_update, 6, 2)

        self.update_switch = QCheckBox()
        self.grid.addWidget(self.update_switch, 6, 3)

        # row -> 8 => "Detach remote trainer"
        lbl_detach = QLabel("Detach remote trainer")
        lbl_detach.setToolTip("Allows the trainer to keep running if connection is lost.")
        self.grid.addWidget(lbl_detach, 8, 2)

        self.detach_switch = QCheckBox()
        self.grid.addWidget(self.detach_switch, 8, 3)

        # row -> 9 => "Reattach id"
        lbl_reattachid = QLabel("Reattach id")
        lbl_reattachid.setToolTip("An id to reattach to a running trainer.")
        self.grid.addWidget(lbl_reattachid, 9, 2)

        reattach_frame = QFrame()
        reattach_layout = QGridLayout(reattach_frame)
        reattach_layout.setContentsMargins(0, 0, 0, 0)
        reattach_layout.setSpacing(5)

        self.reattach_line = QLineEdit()
        reattach_layout.addWidget(self.reattach_line, 0, 0)

        self.reattach_button = QPushButton("Reattach now")
        self.reattach_button.clicked.connect(self.__reattach)
        reattach_layout.addWidget(self.reattach_button, 0, 1)

        self.grid.addWidget(reattach_frame, 9, 3)

        # row -> 11 => "Download samples"
        lbl_dsamples = QLabel("Download samples")
        self.grid.addWidget(lbl_dsamples, 11, 2)
        self.dsamples_switch = QCheckBox()
        self.grid.addWidget(self.dsamples_switch, 11, 3)

        # row -> 12 => "Download output model"
        lbl_dmodel = QLabel("Download output model")
        self.grid.addWidget(lbl_dmodel, 12, 2)
        self.dmodel_switch = QCheckBox()
        self.grid.addWidget(self.dmodel_switch, 12, 3)

        # row -> 13 => "Download saved checkpoints"
        lbl_dsaves = QLabel("Download saved checkpoints")
        self.grid.addWidget(lbl_dsaves, 13, 2)
        self.dsaves_switch = QCheckBox()
        self.grid.addWidget(self.dsaves_switch, 13, 3)

        # row -> 14 => "Download backups"
        lbl_dbackups = QLabel("Download backups")
        self.grid.addWidget(lbl_dbackups, 14, 2)
        self.dbackups_switch = QCheckBox()
        self.grid.addWidget(self.dbackups_switch, 14, 3)

        # row -> 15 => "Download tensorboard logs"
        lbl_dtb = QLabel("Download tensorboard logs")
        self.grid.addWidget(lbl_dtb, 15, 2)
        self.dtb_switch = QCheckBox()
        self.grid.addWidget(self.dtb_switch, 15, 3)

        # row -> 16 => "Delete remote workspace"
        lbl_delws = QLabel("Delete remote workspace")
        self.grid.addWidget(lbl_delws, 16, 2)
        self.delws_switch = QCheckBox()
        self.grid.addWidget(self.delws_switch, 16, 3)

        # row -> 1 => "Create cloud via API" at col=4
        lbl_createcloud = QLabel("Create cloud via API")
        self.grid.addWidget(lbl_createcloud, 1, 4)

        create_frame = QFrame()
        cf_layout = QGridLayout(create_frame)
        create_switch = QCheckBox()
        # if self.train_config.cloud.create:
        #    create_switch.setChecked(True)
        cf_layout.addWidget(create_switch, 0, 0)

        create_button = QPushButton("Create cloud via website")
        create_button.clicked.connect(self.__create_cloud)
        cf_layout.addWidget(create_button, 0, 1)

        self.grid.addWidget(create_frame, 1, 5)

        # row -> 2 => "Cloud name"
        lbl_cname = QLabel("Cloud name")
        self.grid.addWidget(lbl_cname, 2, 4)
        self.cname_line = QLineEdit()
        self.grid.addWidget(self.cname_line, 2, 5)

        # row -> 3 => "Type" (sub_type)
        lbl_stype = QLabel("Type")
        self.grid.addWidget(lbl_stype, 3, 4)
        self.subtype_combo = QComboBox()
        self.subtype_combo.addItem("", "")
        self.subtype_combo.addItem("Community", "COMMUNITY")
        self.subtype_combo.addItem("Secure", "SECURE")
        self.grid.addWidget(self.subtype_combo, 3, 5)

        # row -> 4 => GPU
        lbl_gpu = QLabel("GPU")
        self.grid.addWidget(lbl_gpu, 4, 4)
        # an "advanced" combo?
        self.gpu_combo = QComboBox()
        self.grid.addWidget(self.gpu_combo, 4, 5)
        # plus a button if you want the "..." advanced approach or a direct call
        # in your code you had 'options_adv' that sets a command: __set_gpu_types
        # We'll do a simple button:
        self.gpu_button = QPushButton("Load GPUs")
        self.gpu_button.clicked.connect(self.__set_gpu_types)
        self.grid.addWidget(self.gpu_button, 4, 5)

        # row -> 5 => "Volume size"
        lbl_vol = QLabel("Volume size")
        self.grid.addWidget(lbl_vol, 5, 4)
        self.vol_line = QLineEdit()
        self.grid.addWidget(self.vol_line, 5, 5)

        # row -> 6 => "Min download"
        lbl_minDL = QLabel("Min download")
        self.grid.addWidget(lbl_minDL, 6, 4)
        self.minDL_line = QLineEdit()
        self.grid.addWidget(self.minDL_line, 6, 5)

        # row -> 7 => "Jupyter password"
        lbl_jpass = QLabel("Jupyter password")
        self.grid.addWidget(lbl_jpass, 7, 4)
        self.jpass_line = QLineEdit()
        self.grid.addWidget(self.jpass_line, 7, 5)

        # row -> 9 => "Action on finish"
        lbl_onfin = QLabel("Action on finish")
        self.grid.addWidget(lbl_onfin, 9, 4)
        self.onfin_combo = QComboBox()
        self.onfin_combo.addItem("None", "NONE")
        self.onfin_combo.addItem("Stop", "STOP")
        self.onfin_combo.addItem("Delete", "DELETE")
        self.grid.addWidget(self.onfin_combo, 9, 5)

        # row -> 10 => "Action on error"
        lbl_onerr = QLabel("Action on error")
        self.grid.addWidget(lbl_onerr, 10, 4)
        self.onerr_combo = QComboBox()
        self.onerr_combo.addItem("None", "NONE")
        self.onerr_combo.addItem("Stop", "STOP")
        self.onerr_combo.addItem("Delete", "DELETE")
        self.grid.addWidget(self.onerr_combo, 10, 5)

        # row -> 11 => "Action on detached finish"
        lbl_detfin = QLabel("Action on detached finish")
        self.grid.addWidget(lbl_detfin, 11, 4)
        self.detfin_combo = QComboBox()
        self.detfin_combo.addItem("None", "NONE")
        self.detfin_combo.addItem("Stop", "STOP")
        self.detfin_combo.addItem("Delete", "DELETE")
        self.grid.addWidget(self.detfin_combo, 11, 5)

        # row -> 12 => "Action on detached error"
        lbl_deterr = QLabel("Action on detached error")
        self.grid.addWidget(lbl_deterr, 12, 4)
        self.deterr_combo = QComboBox()
        self.deterr_combo.addItem("None", "NONE")
        self.deterr_combo.addItem("Stop", "STOP")
        self.deterr_combo.addItem("Delete", "DELETE")
        self.grid.addWidget(self.deterr_combo, 12, 5)

    # -------------------------------------------------------------------
    # Non-GUI util functions
    # -------------------------------------------------------------------
    def __set_gpu_types(self):
        if self.train_config.cloud.type == "RUNPOD":  # or CloudType.RUNPOD
            try:
                import runpod
                runpod.api_key = self.train_config.secrets.cloud.api_key
                gpus = runpod.get_gpus()
                # Clear old items
                self.gpu_combo.clear()
                for gpu in gpus:
                    self.gpu_combo.addItem(gpu['id'], gpu['id'])
            except Exception as e:
                print(f"Error fetching GPU types: {e}")

    def __reattach(self):
        self.reattach = True
        try:
            self.parent.start_training()
        finally:
            self.reattach = False

    def __create_cloud(self):
        if self.train_config.cloud.type == "RUNPOD":
            webbrowser.open("https://www.runpod.io/console/deploy?template=1a33vbssq9&type=gpu",
                            new=0, autoraise=False)
