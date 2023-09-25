import PySide6 # noqa: F401 (need to import to select the qt backend)
from PySide6 import QtWidgets, QtGui, QtCore

from wgpu.gui.qt import WgpuCanvas, call_later
from . import VisualizerCanvasBase
from ..drawreason import DrawReason

import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VisualizerCanvas(VisualizerCanvasBase, WgpuCanvas):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hide()

        self._toolbar = QtWidgets.QToolBar()
        self._toolbar.setFloatable(True)
        self._toolbar.setMovable(True)

        self._record_action = QtGui.QAction("Record", self)
        self._record_action.triggered.connect(self.on_click_record)
        self._toolbar.addAction(self._record_action)
        self._recorder = None



        # now replace the wgpu layout with our own
        layout = self.layout()
        layout.removeWidget(self._subwidget)

        our_layout = PySide6.QtWidgets.QVBoxLayout()
        our_layout.addWidget(self._subwidget)
        our_layout.addWidget(self._toolbar)
        our_layout.setContentsMargins(0, 0, 0, 0)

        self._toolbar.adjustSize()

        layout.addLayout(our_layout)


    def on_click_record(self):

        if self._recorder is None:
            logger.info("Starting recorder")
            from ..recorder import VisualizationRecorder
            class QtVisualizationRecorder(VisualizationRecorder):

                @classmethod
                def _progress_iterator(cls, ntot):
                    progress_bar = QtWidgets.QProgressDialog("Rendering to mp4...", "Stop", 0, ntot, self)
                    progress_bar.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

                    last_update = 0

                    for i in range(ntot):
                        # updating the progress bar triggers a render in the main window, which
                        # in turn is quite slow. So only update every second or so.
                        if time.time() - last_update > 1.0:
                            last_update = time.time()
                            progress_bar.setValue(i)
                            if progress_bar.wasCanceled():
                                break
                        yield i
                    progress_bar.close()


            self._recorder = QtVisualizationRecorder(self._visualizer)
            self._recorder.record()
            self._record_action.setText("Stop")
        else:
            logger.info("Stopping recorder")
            self._recorder.stop()
            self._record_action.setText("Record")
            rec = self._recorder
            self._recorder = None

            fd = QtWidgets.QFileDialog(self)
            #fd.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
            fname, _ = fd.getSaveFileName(self, "Save video", "", "MP4 (*.mp4)")
            if fname:
                logger.info("Saving video to %s", fname)
                rec.save_mp4(fname)



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
