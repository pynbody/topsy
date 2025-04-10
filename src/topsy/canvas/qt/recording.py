from __future__ import annotations

import time

from PySide6 import QtWidgets, QtCore

from ...recorder import VisualizationRecorder

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...visualizer import Visualizer


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
