import traceback
from importlib import reload
from functools import partial, wraps
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin
from maya import OpenMayaUI as omui

import pymel.core as pm
import pymel.core.nodetypes as nt

try:
    from PySide2.QtCore import *
    from PySide2.QtGui import *
    from PySide2.QtWidgets import *
except ImportError:
    pass

try:
    from PySide6.QtCore import *
    from PySide6.QtGui import *
    from PySide6.QtWidgets import *
except ImportError:
    pass

customMixinWindow = None


def maya_undo_stack(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = None
        try:
            pm.undoInfo(cn=func.__qualname__, openChunk=True)
            result = func(*args, **kwargs)

        except Exception as e:
            traceback.print_exc()
        finally:
            pm.undoInfo(closeChunk=True)
            return result

    return wrapper


class RobustWeightTransferUI(MayaQWidgetDockableMixin, QWidget):
    def __init__(self, parent=None):
        super(RobustWeightTransferUI, self).__init__(parent)
        self.setObjectName(self.__class__.__name__)
        self.setWindowTitle("Robust Weight Transfer")
        self.setFixedSize(QSize(400, 500))

        self.mainLayout = QVBoxLayout(self)
        self.setLayout(self.mainLayout)
        self.mainLayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # region source ui
        source_layout = QHBoxLayout()
        self.mainLayout.addLayout(source_layout)
        source_label = QLabel(self)
        source_layout.addWidget(source_label)
        source_label.setText("Source Mesh : ")
        source_label.setFixedWidth(90)

        source_line_edit = QLineEdit(self)
        source_layout.addWidget(source_line_edit)
        source_line_edit.setPlaceholderText("Select mesh to < Load > button")
        source_line_edit.setReadOnly(True)
        self.source_line_edit = source_line_edit

        source_btn = QPushButton(self)
        source_layout.addWidget(source_btn)
        source_btn.setText("Load")
        source_btn.setFixedWidth(80)
        # endregion

        # region target ui
        target_layout = QHBoxLayout()
        self.mainLayout.addLayout(target_layout)
        target_label = QLabel(self)
        target_layout.addWidget(target_label)
        target_label.setText("Target Mesh : ")
        target_label.setFixedWidth(90)

        target_line_edit = QLineEdit(self)
        target_layout.addWidget(target_line_edit)
        target_line_edit.setPlaceholderText("Select mesh to < Load > button")
        target_line_edit.setReadOnly(True)
        self.target_line_edit = target_line_edit

        target_btn = QPushButton(self)
        target_layout.addWidget(target_btn)
        target_btn.setText("Load")
        target_btn.setFixedWidth(80)
        # endregion

        separator_frame1 = QFrame(self)
        self.mainLayout.addWidget(separator_frame1)
        separator_frame1.setFrameShape(QFrame.Shape.HLine)

        # region Copy Option
        copy_weight_options_box = QGroupBox(self)
        self.mainLayout.addWidget(copy_weight_options_box)
        copy_weight_options_layout = QVBoxLayout()
        copy_weight_options_box.setLayout(copy_weight_options_layout)
        copy_weight_options_box.setTitle("Copy Options")

        distance_threshold_layout = QHBoxLayout()
        copy_weight_options_layout.addLayout(distance_threshold_layout)
        distance_threshold_label = QLabel(self)
        distance_threshold_layout.addWidget(distance_threshold_label)
        distance_threshold_label.setText("Distance Threshold (cm): ")
        distance_threshold_spin_box = QDoubleSpinBox(self)
        distance_threshold_layout.addWidget(distance_threshold_spin_box)
        distance_threshold_spin_box.setValue(0.1)
        distance_threshold_spin_box.setSingleStep(0.01)
        distance_threshold_spin_box.setMinimum(0.01)
        self.distance_threshold_spin = distance_threshold_spin_box

        angle_threshold_layout = QHBoxLayout()
        copy_weight_options_layout.addLayout(angle_threshold_layout)
        angle_threshold_label = QLabel(self)
        angle_threshold_layout.addWidget(angle_threshold_label)
        angle_threshold_label.setText("Angle Threshold (degrees): ")
        angle_threshold_spin_box = QDoubleSpinBox(self)
        angle_threshold_layout.addWidget(angle_threshold_spin_box)
        angle_threshold_spin_box.setValue(10)
        angle_threshold_spin_box.setSingleStep(1)
        angle_threshold_spin_box.setMinimum(0)
        angle_threshold_spin_box.setMaximum(180)
        self.angle_threshold_spin = angle_threshold_spin_box

        flip_vertex_normal_layout = QHBoxLayout()
        copy_weight_options_layout.addLayout(flip_vertex_normal_layout)
        flip_vertex_normal_label = QLabel()
        flip_vertex_normal_layout.addWidget(flip_vertex_normal_label)
        flip_vertex_normal_label.setText("Flip Vertex Normal: ")
        flip_vertex_normal_cb = QCheckBox(self)
        flip_vertex_normal_layout.addWidget(flip_vertex_normal_cb)
        flip_vertex_normal_cb.setChecked(True)
        self.flip_vertex_normal_cb = flip_vertex_normal_cb

        # endregion

        # region Smooth Option
        smooth_weight_options_box = QGroupBox(self)
        self.mainLayout.addWidget(smooth_weight_options_box)
        smooth_weight_options_layout = QVBoxLayout()
        smooth_weight_options_box.setLayout(smooth_weight_options_layout)
        smooth_weight_options_box.setTitle("Smooth Options")
        smooth_weight_options_box.setCheckable(True)
        smooth_weight_options_box.setChecked(True)
        self.smooth_weight_options = smooth_weight_options_box

        smooth_num_iter_step_layout = QHBoxLayout()
        smooth_weight_options_layout.addLayout(smooth_num_iter_step_layout)
        smooth_num_iter_step_label = QLabel(self)
        smooth_num_iter_step_layout.addWidget(smooth_num_iter_step_label)
        smooth_num_iter_step_label.setText("Num Iters Step : ")
        smooth_num_iter_step_spin_box = QSpinBox(self)
        smooth_num_iter_step_layout.addWidget(smooth_num_iter_step_spin_box)
        smooth_num_iter_step_spin_box.setValue(2)
        smooth_num_iter_step_spin_box.setSingleStep(1)
        smooth_num_iter_step_spin_box.setMinimum(1)
        smooth_num_iter_step_spin_box.setMaximum(100)
        self.smooth_num_iter_step_spin = smooth_num_iter_step_spin_box

        smooth_alpha_layout = QHBoxLayout()
        smooth_weight_options_layout.addLayout(smooth_alpha_layout)
        smooth_alpha_label = QLabel(self)
        smooth_alpha_layout.addWidget(smooth_alpha_label)
        smooth_alpha_label.setText("Alpha : ")
        smooth_alpha_spin_box = QDoubleSpinBox(self)
        smooth_alpha_layout.addWidget(smooth_alpha_spin_box)
        smooth_alpha_spin_box.setValue(0.1)
        smooth_alpha_spin_box.setSingleStep(0.01)
        smooth_alpha_spin_box.setMinimum(0.01)
        smooth_alpha_spin_box.setMaximum(1)
        self.smooth_alpha_spin = smooth_alpha_spin_box

        smooth_range_distance_layout = QHBoxLayout()
        smooth_weight_options_layout.addLayout(smooth_range_distance_layout)
        smooth_range_distance_label = QLabel(self)
        smooth_range_distance_layout.addWidget(smooth_range_distance_label)
        smooth_range_distance_label.setText("Surface Range Distance (cm): ")
        smooth_range_distance_spin_box = QDoubleSpinBox(self)
        smooth_range_distance_layout.addWidget(smooth_range_distance_spin_box)
        smooth_range_distance_spin_box.setValue(1)
        smooth_range_distance_spin_box.setSingleStep(0.01)
        smooth_range_distance_spin_box.setMinimum(0.01)
        self.smooth_range_distance_spin = smooth_range_distance_spin_box

        # endregion

        # region Limit Options
        influences_limit_options_box = QGroupBox(self)
        self.mainLayout.addWidget(influences_limit_options_box)
        influences_limit_options_layout = QVBoxLayout()
        influences_limit_options_box.setLayout(influences_limit_options_layout)
        influences_limit_options_box.setTitle("Influences Limit Options")
        influences_limit_options_box.setCheckable(True)
        influences_limit_options_box.setChecked(True)
        self.influences_limit_options = influences_limit_options_box

        limit_bone_num_layout = QHBoxLayout()
        influences_limit_options_layout.addLayout(limit_bone_num_layout)
        limit_bone_num_label = QLabel(self)
        limit_bone_num_layout.addWidget(limit_bone_num_label)
        limit_bone_num_label.setText("Bone Max Influences : ")
        limit_bone_num_spin_box = QSpinBox(self)
        limit_bone_num_layout.addWidget(limit_bone_num_spin_box)
        limit_bone_num_spin_box.setValue(4)
        limit_bone_num_spin_box.setSingleStep(1)
        limit_bone_num_spin_box.setMinimum(1)
        self.limit_bone_num_spin = limit_bone_num_spin_box

        limit_dilation_repeat_layout = QHBoxLayout()
        influences_limit_options_layout.addLayout(limit_dilation_repeat_layout)
        limit_dilation_repeat_label = QLabel(self)
        limit_dilation_repeat_layout.addWidget(limit_dilation_repeat_label)
        limit_dilation_repeat_label.setText("Dilation Repeat : ")
        limit_dilation_repeat_spin_box = QSpinBox(self)
        limit_dilation_repeat_layout.addWidget(limit_dilation_repeat_spin_box)
        limit_dilation_repeat_spin_box.setValue(5)
        limit_dilation_repeat_spin_box.setSingleStep(1)
        limit_dilation_repeat_spin_box.setMinimum(1)
        self.limit_dilation_repeat_spin = limit_dilation_repeat_spin_box

        # endregion

        separator_frame2 = QFrame(self)
        self.mainLayout.addWidget(separator_frame2)
        separator_frame2.setFrameShape(QFrame.Shape.HLine)

        run_btn = QPushButton(self)
        self.mainLayout.addWidget(run_btn)
        run_btn.setText("Run Task !!! ")

        # clicked connect
        source_btn.clicked.connect(partial(self.__on_clicked_load, self.source_line_edit))
        target_btn.clicked.connect(partial(self.__on_clicked_load, self.target_line_edit))
        run_btn.clicked.connect(self.__on_clicked_run_task)

    @staticmethod
    def dockable_widget_ui_script(restore=False):
        global customMixinWindow

        ''' When the control is restoring, the workspace control has already been created and
            all that needs to be done is restoring its UI.
        '''
        if restore == True:
            # Grab the created workspace control with the following.
            restoredControl = omui.MQtUtil.getCurrentParent()

        if customMixinWindow is None:
            # Create a custom mixin widget for the first time
            customMixinWindow = RobustWeightTransferUI()
            customMixinWindow.setObjectName(RobustWeightTransferUI.__name__)

        if restore == True:
            # Add custom mixin widget to the workspace control
            mixinPtr = omui.MQtUtil.findControl(customMixinWindow.objectName())
            omui.MQtUtil.addWidgetToMayaLayout(int(mixinPtr), int(restoredControl))
        else:
            # Create a workspace control for the mixin widget by passing all the needed parameters. See workspaceControl command documentation for all available flags.
            customMixinWindow.show(dockable=True, height=600, width=480,
                                   uiScript='DockableWidgetUIScript(restore=True)')

        return customMixinWindow

    def __on_clicked_load(self, widget):
        if not isinstance(widget, QLineEdit):
            raise TypeError("Widget must be of type QLineEdit")

        sel = pm.ls(sl=True, fl=True) or [None]
        if isinstance(sel[0], nt.Transform) or isinstance(sel[0], nt.Mesh):
            widget.setText(sel[0].name())

        return

    @maya_undo_stack
    def __on_clicked_run_task(self):
        """
        Generate the configuration from the ui, to run the task.
        :return:
        """

        from . import core
        reload(core)

        # region Generate Config
        copy_options = core.CopyOptions(
            distance_threshold=self.distance_threshold_spin.value(),
            angle_threshold=self.angle_threshold_spin.value(),
            flip_vertex_normal=self.flip_vertex_normal_cb.isChecked()
        )
        smooth_options = None
        if self.smooth_weight_options.isChecked():
            smooth_options = core.SmoothOptions(
                num_iter_step=self.smooth_num_iter_step_spin.value(),
                alpha=self.smooth_alpha_spin.value(),
                range_distance=self.smooth_range_distance_spin.value()
            )

        limit_options = None
        if self.influences_limit_options.isChecked():
            limit_options = core.LimitOptions(
                bone_num=self.limit_bone_num_spin.value(),
                dilation_repeat=self.limit_dilation_repeat_spin.value()
            )
        # endregion

        weight_transfer = core.WeightTransfer(
            self.source_line_edit.text(),
            self.target_line_edit.text(),
            copy_options=copy_options,
            smooth_options=smooth_options,
            limit_options=limit_options
        )
        weight_transfer.run()
        return


def show():
    """
    setup to show RobustWeightTransferUI
    :return:
    """
    global customMixinWindow
    _window = f"{RobustWeightTransferUI.__name__}WorkspaceControl"
    if pm.window(_window, exists=True):
        pm.deleteUI(_window)
        customMixinWindow = None
    RobustWeightTransferUI.dockable_widget_ui_script()

    return
