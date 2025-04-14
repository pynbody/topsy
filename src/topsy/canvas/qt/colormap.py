from __future__ import annotations

from typing import Any

import matplotlib as mpl
from PySide6 import QtWidgets, QtCore
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
from rendercanvas import BaseRenderCanvas

from .lineedit import MyLineEdit

import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MapControlsBase(QtWidgets.QDialog):
    def __init__(self, parent: BaseRenderCanvas):
        self._parent = parent
        super().__init__()

        self.setWindowTitle("Color controls")
        self.setWindowFlags(QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint)
        self.resize(400, 0)

    def open(self):
        action_rect = self._parent._toolbar.actionGeometry(self._parent._open_cmap) # EEK!
        popoverPosition = self._parent._toolbar.mapToGlobal(action_rect.topLeft())
        super().show()
        self.move(popoverPosition - QtCore.QPoint(self.width()//2, self.height()))



class RGBMapControls(MapControlsBase):
    def __init__(self, parent: BaseRenderCanvas):
        super().__init__(parent)

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

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




    def open(self):
        self._colormap = self._parent._visualizer._colormap # EEK!
        self._mag_range.setValue((self._colormap.min_mag, self._colormap.max_mag))
        self._gamma_slider.setValue(self._colormap.gamma)

        super().open()

    def _mag_range_changed(self):
        self._colormap.min_mag, self._colormap.max_mag = self._mag_range.value()

    def _gamma_changed(self):
        self._colormap.gamma = self._gamma_slider.value()

class QLabeledDoubleRangeSliderWithAutoscale(QLabeledDoubleRangeSlider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._scale_exponent = 0
        super().__init__(*args, **kwargs)

    def _scale_float(self, value: float) -> float:
        return value / 10**self._scale_exponent

    def _unscale_float(self, value: float) -> float:
        return value * 10**self._scale_exponent

    def _repr_value_to_scale_exponent(self, value: float) -> int:
        if value == 0.0:
            return 0
        try:
            exponent = math.floor(math.log10(abs(value)))
        except ValueError:
            return 0
        if exponent < -2 or exponent > 2:
            return exponent
        else:
            return 0

    def setRange(self, vmin: float, vmax: float) -> None:
        if vmin == 0.0 and vmax == 0.0:
            repr_val = 1.0
        elif vmin==0.0:
            repr_val = vmax
        elif vmax==0.0:
            repr_val = vmin
        else:
            repr_val = max(abs(vmin), abs(vmax))

        self._scale_exponent = self._repr_value_to_scale_exponent(repr_val)
        scaled_min = self._scale_float(vmin)
        scaled_max = self._scale_float(vmax)

        super().setRange(scaled_min, scaled_max)

    def setValue(self, value: tuple[float, float]) -> None:
        scaled_value = (self._scale_float(value[0]), self._scale_float(value[1]))
        super().setValue(scaled_value)

    def value(self) -> tuple[float, float]:
        scaled_value = super().value()
        return (self._unscale_float(scaled_value[0]), self._unscale_float(scaled_value[1]))



class ColorMapControls(MapControlsBase):
    _default_quantity_name = "Projected density"

    def __init__(self, canvas: BaseRenderCanvas):
        super().__init__(canvas)

        self._visualizer = canvas._visualizer # EEK!

        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        self._menu_layout = QtWidgets.QHBoxLayout()
        self._colormap_menu = QtWidgets.QComboBox()
        self._colormap_menu.addItems(mpl.colormaps.keys())
        self._colormap_menu.currentTextChanged.connect(self._colormap_menu_changed_action)

        self._quantity_menu = QtWidgets.QComboBox()
        self._quantity_menu.setEditable(True)
        self._quantity_menu.setLineEdit(MyLineEdit())
        self._quantity_menu.lineEdit().editingFinished.connect(self._quantity_menu_changed_action)
        self._quantity_menu.currentIndexChanged.connect(self._quantity_menu_changed_action)

        self._first_update = True
        self._disable_updates = True

        self._log_checkbox = QtWidgets.QCheckBox("Log scale")
        self._log_checkbox.stateChanged.connect(self._log_checkbox_changed)

        self._menu_layout.addWidget(self._colormap_menu)
        self._menu_layout.addWidget(self._quantity_menu)
        self._menu_layout.addWidget(self._log_checkbox)

        self._range_layout = QtWidgets.QHBoxLayout()
        self._slider = QLabeledDoubleRangeSliderWithAutoscale()
        self._slider.setRange(0, 100)
        self._slider.setValue((10,50))
        self._slider.valueChanged.connect(self._slider_changed)

        self._auto_button = QtWidgets.QPushButton("Auto")
        self._auto_button.setStyleSheet("color: black;") # unclear why this is necessary
        self._auto_button.pressed.connect(self._auto)


        self._range_layout.addWidget(self._slider)
        self._range_layout.addWidget(self._auto_button)


        self._layout.addLayout(self._menu_layout)
        self._layout.addLayout(self._range_layout)


    def open(self):
        self._update_ui()
        super().open()

    def _auto(self):
        self._visualizer._colormap.autorange_vmin_vmax()
        self._update_ui()

    def _update_ui(self):
        self._disable_updates = True
        try:
            self._colormap_menu.setCurrentText(self._visualizer.colormap_name)
            if self._first_update:
                self._quantity_menu.addItem(self._default_quantity_name)
                self._quantity_menu.addItems(sorted(self._visualizer.data_loader.get_quantity_names(), key = lambda s: s.lower()))
                self._first_update = False
            self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
            self._quantity_menu.adjustSize()
            self._slider.setRange(*self._visualizer._colormap.get_ui_range())
            self._log_checkbox.setChecked(self._visualizer.log_scale)
            self._slider.setValue((self._visualizer.vmin, self._visualizer.vmax))
        finally:
            self._disable_updates = False

    def _colormap_menu_changed_action(self):
        if self._disable_updates:
            return
        logger.info("Colormap changed to %s", self._colormap_menu.currentText())
        self._visualizer.colormap_name = self._colormap_menu.currentText()
        self._update_ui()

    def _log_checkbox_changed(self):
        if self._disable_updates:
            return
        if self._visualizer.log_scale == self._log_checkbox.isChecked():
            return
        logger.info("Log scale changed to %s", self._log_checkbox.isChecked())

        self._visualizer.log_scale = self._log_checkbox.isChecked()
        self._visualizer.vmin, self._visualizer.vmax = self._visualizer._colormap.get_ui_range()
        self._update_ui()

    def _slider_changed(self):
        if self._disable_updates:
            return
        self._visualizer.vmin, self._visualizer.vmax = self._slider.value()

    def _quantity_menu_changed_action(self):
        if self._disable_updates:
            return

        if self._quantity_menu.currentText() == self._default_quantity_name:
            new_quantity_name = None
        else:
            new_quantity_name = self._quantity_menu.currentText()

        if self._visualizer.quantity_name == new_quantity_name:
            return

        logger.info("Quantity changed to %s", new_quantity_name)

        try:
            self._visualizer.quantity_name = new_quantity_name
        except ValueError as e:
            self._qt_errorbox(e)

        self._visualizer.render_sph()
        self._auto()

    def _qt_errorbox(self, e):
        message = QtWidgets.QMessageBox(self)
        message.setWindowTitle("Invalid quantity")
        message.setText(str(e))
        message.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        message.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        self._quantity_menu.setCurrentText(self._visualizer.quantity_name or self._default_quantity_name)
        message.exec()
