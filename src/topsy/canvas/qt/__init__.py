from __future__ import annotations


import PySide6 # noqa: F401 (need to import to select the qt backend)
from PySide6 import QtWidgets, QtGui, QtCore

from rendercanvas.qt import RenderCanvas, loop

from .colormap import RGBColorControls, ColorMapControls, BivariateColorMapControls
from .lineedit import MyLineEdit
from .recording import RecordingSettingsDialog, VisualizationRecorderWithQtProgressbar
from .. import VisualizerCanvasBase
from ...drawreason import DrawReason

import os
import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _get_icon(name):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return QtGui.QIcon(os.path.join(this_dir, "icons", name))


class VisualizerCanvas(VisualizerCanvasBase, RenderCanvas):

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




        self._cmap_icon = _get_icon("rgb.png")
        self._open_cmap = QtGui.QAction(self._cmap_icon, "Color", self)



        self._toolbar.addAction(self._load_script_action)
        self._toolbar.addAction(self._save_script_action)
        self._toolbar.addAction(self._record_action)
        self._toolbar.addAction(self._save_movie_action)

        self._toolbar.addSeparator()
        self._toolbar.addAction(self._save_action)
        self._toolbar.addSeparator()

        self._toolbar.addAction(self._open_cmap)

        if not self._visualizer._rgb:
            cmap_menu = QtWidgets.QComboBox(self._toolbar)
            cmap_menu.addItems(["Univariate", "Bivariate"])
            if self._visualizer._bivariate:
                cmap_menu.setCurrentIndex(1)
            else:
                cmap_menu.setCurrentIndex(0)
            cmap_menu.currentIndexChanged.connect(self._on_change_cmap_type)
            self._toolbar.addWidget(cmap_menu)

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
        self.call_later(0, self._prepare_colormap_pane)

    def _on_change_cmap_type(self, index):
        match index:
            case 0:
                self._visualizer._bivariate = False
            case 1:
                self._visualizer._bivariate = True

        self._visualizer._reinitialize_colormap_and_bar()
        self._prepare_colormap_pane()

    def _prepare_colormap_pane(self):
        if hasattr(self, '_cmap_connection'):
            self._open_cmap.disconnect(self._cmap_connection)
        if not self._visualizer._hdr and not self._visualizer._rgb:
            if self._visualizer._bivariate:
                self._colormap_controls = BivariateColorMapControls(self)
            else:
                self._colormap_controls = ColorMapControls(self)
        elif self._visualizer._rgb:
            self._colormap_controls = RGBColorControls(self)

        self._cmap_connection = self._open_cmap.triggered.connect(self._colormap_controls.open)


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
        fname, _ = fd.getSaveFileName(self, "Save snapshot", "", "PNG (*.png);; PDF (*.pdf);; numpy (*.npy)")
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
            from ... import view_synchronizer
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
        call_count = 0
        def function_wrapper():
            nonlocal call_count
            if call_count == 0:
                function()
            else:
                # we have been cached!
                self._visualizer.draw(DrawReason.PRESENTATION_CHANGE)
            call_count += 1

        super().request_draw(function_wrapper)

    @classmethod
    def call_later(cls, delay, fn, *args):
        loop.call_later(delay, fn, *args)
