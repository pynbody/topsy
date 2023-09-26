from __future__ import annotations


import PySide6 # noqa: F401 (need to import to select the qt backend)
from PySide6 import QtWidgets, QtGui, QtCore

from wgpu.gui.qt import WgpuCanvas, call_later
from . import VisualizerCanvasBase
from ..drawreason import DrawReason
from ..recorder import VisualizationRecorder

import os
import time
import logging
import matplotlib as mpl

from typing import TYPE_CHECKING

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


class VisualizationRecorderWithQtProgressbar(VisualizationRecorder):

    def __init__(self, visualizer: Visualizer, parent_widget: QtWidgets.QWidget):
        super().__init__(visualizer)
        self._parent_widget = parent_widget

    def _progress_iterator(self, ntot):
        progress_bar = QtWidgets.QProgressDialog("Rendering to mp4...", "Stop", 0, ntot, self._parent_widget)
        progress_bar.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        last_update = 0

        loop = QtCore.QEventLoop()

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

        progress_bar.close()
        del progress_bar

class VisualizerCanvas(VisualizerCanvasBase, WgpuCanvas):
    _default_quantity_name = "Projected density"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hide()

        self._toolbar = QtWidgets.QToolBar()
        self._toolbar.setIconSize(QtCore.QSize(16, 16))

        # setup toolbar to show text and icons
        self._toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)


        self._record_icon = _get_icon("record.png")
        self._stop_icon = _get_icon("stop.png")
        self._save_icon = _get_icon("camera.png")

        self._record_action = QtGui.QAction(self._record_icon, "Record", self)
        self._record_action.setIconText("Record")
        self._record_action.triggered.connect(self.on_click_record)

        self._save_action = QtGui.QAction(self._save_icon, "Snapshot", self)
        self._save_action.setIconText("Snapshot")
        self._save_action.triggered.connect(self.on_click_save)

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
            logger.info(f"Quantities: {self._visualizer.data_loader.get_quantity_names()}")
            self._quantity_menu.addItems( self._visualizer.data_loader.get_quantity_names())
            self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
            self._quantity_menu.currentIndexChanged.connect(self._quantity_menu_changed_action)
            self._quantity_menu.lineEdit().editingFinished.connect(self._quantity_menu_changed_action)
            self._quantity_menu.adjustSize()

        self.call_later(0, populate_quantity_menu)


        self._toolbar.addAction(self._record_action)
        self._toolbar.addAction(self._save_action)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._colormap_menu)
        self._toolbar.addWidget(self._quantity_menu)
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

        layout.addLayout(our_layout)

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
                message = QtWidgets.QMessageBox(self)
                message.setWindowTitle("Invalid quantity")
                message.setText(str(e))
                message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                message.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
                self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
                message.exec()


    def on_click_record(self):

        if self._recorder is None:
            logger.info("Starting recorder")
            self._recorder = VisualizationRecorderWithQtProgressbar(self._visualizer, self)
            self._recorder.record()
            self._record_action.setIconText("Finish and export to mp4")
            self._record_action.setIcon(self._stop_icon)
        else:
            logger.info("Stopping recorder")
            self._recorder.stop()
            self._record_action.setIconText("Record")
            self._record_action.setIcon(self._record_icon)
            rec = self._recorder
            self._recorder = None

            fd = QtWidgets.QFileDialog(self)
            fname, _ = fd.getSaveFileName(self, "Save video", "", "MP4 (*.mp4)")
            if fname:
                logger.info("Saving video to %s", fname)
                rec.save_mp4(fname)

    def on_click_save(self):
        fd = QtWidgets.QFileDialog(self)
        fname, _ = fd.getSaveFileName(self, "Save snapshot", "", "PNG (*.png);; PDF (*.pdf)")
        if fname:
            logger.info("Saving snapshot to %s", fname)
            self._visualizer.save(fname)



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
