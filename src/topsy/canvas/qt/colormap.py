from __future__ import annotations

import matplotlib as mpl
from PySide6 import QtWidgets, QtGui, QtCore
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
from wgpu.gui.qt import WgpuCanvas

from .lineedit import MyLineEdit

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RGBMapControls(QtWidgets.QDialog):
    def __init__(self, parent: WgpuCanvas):
        from . import _get_icon
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
        self._mag_range.setWindowTitle("Star rendering map")
        self._mag_range.setRange(5, 35)
        self._mag_range.setValue((15,32))
        self._mag_range.valueChanged.connect(self._mag_range_changed)
        self._mag_label = QtWidgets.QLabel("mag/arcsec^2")
        self._mag_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self._gamma_label = QtWidgets.QLabel("gamma")
        self._gamma_slider = QLabeledDoubleSlider()
        self._gamma_slider.setRange(0.25, 8.0)
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
