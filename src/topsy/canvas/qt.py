from __future__ import annotations


import PySide6 # noqa: F401 (need to import to select the qt backend)
from PySide6 import QtWidgets, QtGui, QtCore
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider

from wgpu.gui.qt import WgpuCanvas, call_later
from . import VisualizerCanvasBase
from ..drawreason import DrawReason
from ..recorder import VisualizationRecorder

import os
import time
import logging
import matplotlib as mpl

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..visualizer import Visualizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _get_icon(name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return QtGui.QIcon(os.path.join(this_dir, "icons", name))


class MyLineEdit(QtWidgets.QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timer = QtCore.QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.selectAll)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._timer.start(0)

class RecordingSettingsDialog(QtWidgets.QDialog):

    def __init__(self, *args):
        super().__init__(*args)
        self.setWindowTitle("Recording settings")
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # checkbox for smoothing:
        self._smooth_checkbox = QtWidgets.QCheckBox("Smooth timestream camera movements")
        self._smooth_checkbox.setChecked(True)
        self._layout.addWidget(self._smooth_checkbox)

        # leave some space:
        self._layout.addSpacing(10)

        # checkbox for including vmin/vmax:
        self._vmin_vmax_checkbox = QtWidgets.QCheckBox("Set vmin/vmax from timestream")
        self._vmin_vmax_checkbox.setChecked(True)
        self._layout.addWidget(self._vmin_vmax_checkbox)

        # checkbox for changing quantity:
        self._quantity_checkbox = QtWidgets.QCheckBox("Set quantity from timestream")
        self._quantity_checkbox.setChecked(True)
        self._layout.addWidget(self._quantity_checkbox)

        self._layout.addSpacing(10)

        # checkbox for showing colorbar:
        self._colorbar_checkbox = QtWidgets.QCheckBox("Show colorbar")
        self._colorbar_checkbox.setChecked(True)
        self._layout.addWidget(self._colorbar_checkbox)

        # checkbox for showing scalebar:
        self._scalebar_checkbox = QtWidgets.QCheckBox("Show scalebar")
        self._scalebar_checkbox.setChecked(True)
        self._layout.addWidget(self._scalebar_checkbox)

        self._layout.addSpacing(10)


        # select resolution from dropdown, with options half HD, HD, 4K
        self._resolution_dropdown = QtWidgets.QComboBox()
        self._resolution_dropdown.addItems(["Half HD (960x540)", "HD (1920x1080)", "4K (3840x2160)"])
        self._resolution_dropdown.setCurrentIndex(1)

        # select fps from dropdown, with options 24, 30, 60
        self._fps_dropdown = QtWidgets.QComboBox()
        self._fps_dropdown.addItems(["24 fps", "30 fps", "60 fps"])
        self._fps_dropdown.setCurrentIndex(1)

        # put resolution/fps next to each other horizontally:
        self._resolution_fps_layout = QtWidgets.QHBoxLayout()
        self._resolution_fps_layout.addWidget(self._resolution_dropdown)
        self._resolution_fps_layout.addWidget(self._fps_dropdown)
        self._layout.addLayout(self._resolution_fps_layout)

        self._layout.addSpacing(10)

        # cancel and save.. buttons:
        self._cancel_save_layout = QtWidgets.QHBoxLayout()
        self._cancel_button = QtWidgets.QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        self._save_button = QtWidgets.QPushButton("Save")
        # save button should be default:
        self._save_button.setDefault(True)
        self._save_button.clicked.connect(self.accept)
        self._cancel_save_layout.addWidget(self._cancel_button)
        self._cancel_save_layout.addWidget(self._save_button)
        self._layout.addLayout(self._cancel_save_layout)

        # show as a sheet on macos:
        #self.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.setWindowFlags(QtCore.Qt.WindowType.Sheet)

    @property
    def fps(self):
        return float(self._fps_dropdown.currentText().split()[0])

    @property
    def resolution(self):
        import re
        # use regexp
        # e.g. the string 'blah (123x456)' should map to tuple (123,456)
        match = re.match(r".*\((\d+)x(\d+)\)", self._resolution_dropdown.currentText())
        return int(match.group(1)), int(match.group(2))

    @property
    def smooth(self):
        return self._smooth_checkbox.isChecked()

    @property
    def set_vmin_vmax(self):
        return self._vmin_vmax_checkbox.isChecked()

    @property
    def set_quantity(self):
        return self._quantity_checkbox.isChecked()

    @property
    def show_colorbar(self):
        return self._colorbar_checkbox.isChecked()

    @property
    def show_scalebar(self):
        return self._scalebar_checkbox.isChecked()


class RGBMapControls(QtWidgets.QDialog):
    def __init__(self, parent: WgpuCanvas):
        super().__init__()
        self._parent = parent
        self._visualizer = parent._visualizer


        self.setWindowTitle("RGB map controls")

        # set default width of window to 400 pixels:
        self.resize(400, 0)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self._rgb_icon = _get_icon("rgb.png")
        self._open_action = QtGui.QAction(self._rgb_icon, "RGB controls", self)
        self._open_action.triggered.connect(self.open)

        self._mag_range = QLabeledDoubleRangeSlider()
        self._mag_range.setWindowTitle("Test")
        self._mag_range.setRange(15, 35)
        self._mag_range.setValue((15,32))
        self._mag_range.valueChanged.connect(self._mag_range_changed)
        self._mag_label = QtWidgets.QLabel("mag/arcsec^2")
        self._mag_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self._gamma_label = QtWidgets.QLabel("gamma")
        self._gamma_slider = QLabeledDoubleSlider()
        self._gamma_slider.setRange(0.5, 3.0)
        self._gamma_slider.setValue(1.0)
        self._gamma_slider.valueChanged.connect(self._gamma_changed)


        self._layout.addWidget(self._mag_range)
        self._layout.addWidget(self._mag_label)
        self._layout.addSpacing(10)
        self._layout.addWidget(self._gamma_label)
        self._layout.addWidget(self._gamma_slider)

        self.setWindowFlags(QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint)


    def open(self):
        self._colormap = self._visualizer._colormap
        self._mag_range.setValue((self._colormap.min_mag, self._colormap.max_mag))
        self._gamma_slider.setValue(self._colormap.gamma)

        action_rect = self._parent._toolbar.actionGeometry(self._open_action)
        popoverPosition = self._parent._toolbar.mapToGlobal(action_rect.topLeft())
        super().show()
        self.move(popoverPosition - QtCore.QPoint(self.width()//2, self.height()))




    def _mag_range_changed(self):
        self._colormap.min_mag, self._colormap.max_mag = self._mag_range.value()

    def _gamma_changed(self):
        self._colormap.gamma = self._gamma_slider.value()

    def add_to_toolbar(self, toolbar):
        toolbar.addAction(self._open_action)



class VisualizationRecorderWithQtProgressbar(VisualizationRecorder):

    def __init__(self, visualizer: Visualizer, parent_widget: QtWidgets.QWidget):
        super().__init__(visualizer)
        self._parent_widget = parent_widget

    def _progress_iterator(self, ntot):
        progress_bar = QtWidgets.QProgressDialog("Rendering to mp4...", "Stop", 0, ntot, self._parent_widget)
        progress_bar.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress_bar.forceShow()

        last_update = 0

        loop = QtCore.QEventLoop()

        try:
            for i in range(ntot):
                # updating the progress bar triggers a render in the main window, which
                # in turn is quite slow (because it can trigger software rendering
                # of resizable elements like the colorbar). So only update every half second or so.
                if time.time() - last_update > 0.5:
                    last_update = time.time()
                    progress_bar.setValue(i)

                    with self._visualizer.prevent_sph_rendering():
                        loop.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

                    if progress_bar.wasCanceled():
                        break
                yield i

        finally:
            progress_bar.close()

class ColorMapControls:
    _default_quantity_name = "Projected density"

    def __init__(self, canvas: WgpuCanvas):
        self._canvas = canvas
        self._visualizer = canvas._visualizer

        self._colormap_menu = QtWidgets.QComboBox()
        self._colormap_menu.addItems(mpl.colormaps.keys())
        self._colormap_menu.setCurrentText(self._visualizer.colormap_name)
        self._colormap_menu.currentTextChanged.connect(self._colormap_menu_changed_action)

        self._quantity_menu = QtWidgets.QComboBox()
        self._quantity_menu.addItem(self._default_quantity_name)
        self._quantity_menu.setEditable(True)

        self._quantity_menu.setLineEdit(MyLineEdit())

        # at this moment, the data loader hasn't been initialized yet, so we can't
        # use it to populate the menu. This needs a callback:
        def populate_quantity_menu():
            self._quantity_menu.addItems(self._visualizer.data_loader.get_quantity_names())
            self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
            self._quantity_menu.currentIndexChanged.connect(self._quantity_menu_changed_action)
            self._quantity_menu.lineEdit().editingFinished.connect(self._quantity_menu_changed_action)
            self._quantity_menu.adjustSize()
            self._canvas.setWindowTitle("topsy: " + self._visualizer.data_loader.get_filename())

        self._canvas.call_later(0, populate_quantity_menu)

    def add_to_toolbar(self, toolbar):
        toolbar.addWidget(self._colormap_menu)
        toolbar.addWidget(self._quantity_menu)

    def _colormap_menu_changed_action(self):
        logger.info("Colormap changed to %s", self._colormap_menu.currentText())
        self._visualizer.colormap_name = self._colormap_menu.currentText()

    def _quantity_menu_changed_action(self):
        logger.info("Quantity changed to %s", self._quantity_menu.currentText())
        if self._quantity_menu.currentText() == self._default_quantity_name:
            self._visualizer.quantity_name = None
        else:
            try:
                self._visualizer.quantity_name = self._quantity_menu.currentText()
            except ValueError as e:
                message = QtWidgets.QMessageBox(self._canvas)
                message.setWindowTitle("Invalid quantity")
                message.setText(str(e))
                message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                message.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
                self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
                message.exec()


class VisualizerCanvas(VisualizerCanvasBase, WgpuCanvas):

    _all_instances = []
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._all_instances.append(self)
        self.hide()

        self._toolbar = QtWidgets.QToolBar()
        self._toolbar.setIconSize(QtCore.QSize(16, 16))

        # setup toolbar to show text and icons
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._load_icons()

        self._record_action = QtGui.QAction(self._record_icon, "Record", self)
        self._record_action.triggered.connect(self.on_click_record)

        self._save_action = QtGui.QAction(self._save_icon, "Snapshot", self)
        self._save_action.triggered.connect(self.on_click_save)

        self._save_movie_action = QtGui.QAction(self._save_movie_icon, "Save mp4", self)
        self._save_movie_action.triggered.connect(self.on_click_save_movie)
        self._save_movie_action.setDisabled(True)

        self._save_script_action = QtGui.QAction(self._export_icon, "Save timestream", self)
        self._save_script_action.triggered.connect(self.on_click_save_script)
        self._save_script_action.setDisabled(True)

        self._load_script_action = QtGui.QAction(self._import_icon, "Load timestream", self)
        self._load_script_action.triggered.connect(self.on_click_load_script)

        self._link_action = QtGui.QAction(self._unlinked_icon, "Link to other windows", self)
        self._link_action.setIconText("Link")
        self._link_action.triggered.connect(self.on_click_link)



        self._toolbar.addAction(self._load_script_action)
        self._toolbar.addAction(self._save_script_action)
        self._toolbar.addAction(self._record_action)
        self._toolbar.addAction(self._save_movie_action)

        self._toolbar.addSeparator()
        self._toolbar.addAction(self._save_action)
        self._toolbar.addSeparator()

        if not self._visualizer._hdr and not self._visualizer._rgb:
            self._colormap_controls = ColorMapControls(self)
            self._colormap_controls.add_to_toolbar(self._toolbar)
            self._toolbar.addSeparator()
        elif self._visualizer._rgb:
            self._colormap_controls = RGBMapControls(self)
            self._colormap_controls.add_to_toolbar(self._toolbar)
            self._toolbar.addSeparator()

        self._toolbar.addAction(self._link_action)
        self._recorder = None



        # now replace the wgpu layout with our own
        layout = self.layout()
        layout.removeWidget(self._subwidget)

        our_layout = PySide6.QtWidgets.QVBoxLayout()
        our_layout.addWidget(self._subwidget)
        our_layout.addWidget(self._toolbar)
        our_layout.setContentsMargins(0, 0, 0, 0)
        our_layout.setSpacing(0)

        self._toolbar.adjustSize()

        self._toolbar_update_timer = QtCore.QTimer(self)
        self._toolbar_update_timer.timeout.connect(self._update_toolbar)
        self._toolbar_update_timer.start(100)

        layout.addLayout(our_layout)

    def __del__(self):
        try:
            self._all_instances.remove(self)
        except ValueError:
            pass
        super().__del__()

    def _load_icons(self):
        self._record_icon = _get_icon("record.png")
        self._stop_icon = _get_icon("stop.png")
        self._save_icon = _get_icon("camera.png")
        self._linked_icon = _get_icon("linked.png")
        self._unlinked_icon = _get_icon("unlinked.png")
        self._save_movie_icon = _get_icon("movie.png")
        self._import_icon = _get_icon("load_script.png")
        self._export_icon = _get_icon("save_script.png")

    def on_click_record(self):

        if self._recorder is None or not self._recorder.recording:
            logger.info("Starting recorder")
            self._recorder = VisualizationRecorderWithQtProgressbar(self._visualizer, self)
            self._recorder.record()
            self._record_action.setIconText("Stop")
            self._record_action.setIcon(self._stop_icon)
        else:
            logger.info("Stopping recorder")
            self._recorder.stop()
            self._record_action.setIconText("Record")
            self._record_action.setIcon(self._record_icon)

    def on_click_save_movie(self):
        # show the options dialog first:
        dialog = RecordingSettingsDialog(self)
        dialog.exec()
        if dialog.result() == QtWidgets.QDialog.DialogCode.Accepted:
            fd = QtWidgets.QFileDialog(self)
            fname, _ = fd.getSaveFileName(self, "Save video", "", "MP4 (*.mp4)")
            if fname:
                logger.info("Saving video to %s", fname)
                self._recorder.save_mp4(fname, show_colorbar=dialog.show_colorbar,
                                        show_scalebar=dialog.show_scalebar,
                                        fps=dialog.fps,
                                        resolution=dialog.resolution,
                                        smooth=dialog.smooth,
                                        set_vmin_vmax=dialog.set_vmin_vmax,
                                        set_quantity=dialog.set_quantity)
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fname))

    def on_click_save(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getSaveFileName(self, "Save snapshot", "", "PNG (*.png);; PDF (*.pdf)")
        if fname:
            logger.info("Saving snapshot to %s", fname)
            self._visualizer.save(fname)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fname))

    def on_click_save_script(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getSaveFileName(self, "Save camera movements", "", "Python Pickle (*.pickle)")
        if fname:
            logger.info("Saving timestream to %s", fname)
            self._recorder.save_timestream(fname)

    def on_click_load_script(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getOpenFileName(self, "Load camera movements", "", "Python Pickle (*.pickle)")
        if fname:
            logger.info("Loading timestream from %s", fname)
            self._recorder = VisualizationRecorderWithQtProgressbar(self._visualizer, self)
            self._recorder.load_timestream(fname)


    def on_click_link(self):
        if self._visualizer.is_synchronizing():
            logger.info("Stop synchronizing")
            self._visualizer.stop_synchronizing()
        else:
            logger.info("Start synchronizing")
            from .. import view_synchronizer
            synchronizer = view_synchronizer.ViewSynchronizer()
            for instance in self._all_instances:
                synchronizer.add_view(instance._visualizer)

    def _update_toolbar(self):
        if self._recorder is not None or len(self._all_instances)<2:
            self._link_action.setDisabled(True)
        else:
            self._link_action.setDisabled(False)
            if self._visualizer.is_synchronizing():
                self._link_action.setIcon(self._linked_icon)
                self._link_action.setIconText("Unlink")
            else:
                self._link_action.setIcon(self._unlinked_icon)
                self._link_action.setIconText("Link")
        if self._recorder is not None and not self._recorder.recording:
            self._save_movie_action.setDisabled(False)
            self._save_script_action.setDisabled(False)
        else:
            self._save_movie_action.setDisabled(True)
            self._save_script_action.setDisabled(True)




    def request_draw(self, function=None):
        # As a side effect, wgpu gui layer stores our function call, to enable it to be
        # repainted later. But we want to distinguish such repaints and handle them
        # differently, so we need to replace the function with our own
        def function_wrapper():
            function()
            self._subwidget.draw_frame = lambda: self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)

        super().request_draw(function_wrapper)

    @classmethod
    def call_later(cls, delay, fn, *args):
        call_later(delay, fn, *args)
