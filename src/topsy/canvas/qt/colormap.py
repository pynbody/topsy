from __future__ import annotations

from typing import Any, Dict, Union, Callable

from PySide6 import QtWidgets, QtCore
from superqt import QLabeledDoubleRangeSlider, QLabeledDoubleSlider
from rendercanvas import BaseRenderCanvas

from .lineedit import MyLineEdit

from ...colormap.ui import LayoutSpec, GenericController, ControlSpec, UnifiedColorMapController

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



class ColorMapControls(QtWidgets.QDialog):
    def __init__(self, canvas: BaseRenderCanvas):
        super().__init__(canvas)
        self.setWindowTitle("Color controls")
        self.setWindowFlags(QtCore.Qt.WindowType.Popup
                            | QtCore.Qt.WindowType.FramelessWindowHint)

        self.controller: GenericController = UnifiedColorMapController(canvas._visualizer, self._refresh_ui)

        # build UI
        self._widgets: Dict[str, QtWidgets.QWidget] = {}
        root_spec = self.controller.get_layout()
        self._layout = self._build_layout(root_spec)
        self.setLayout(self._layout)

    def open(self):
        # position next to toolbar
        self.controller.refresh_ui()
        super().show()
        self._update_screen_size_and_position()

    def _update_screen_size_and_position(self):
        self.resize(400, 0)
        self.updateGeometry()
        action_rect = self.parent()._toolbar.actionGeometry(
            self.parent()._open_cmap
        )
        pos = self.parent()._toolbar.mapToGlobal(action_rect.topLeft())
        self.move(pos - QtCore.QPoint(self.width()//2, self.height()))

    def _build_layout(self, spec: LayoutSpec) -> QtWidgets.QLayout:
        if spec.type == "vbox":
            layout = QtWidgets.QVBoxLayout()
        else:
            layout = QtWidgets.QHBoxLayout()

        for child in spec.children:
            if isinstance(child, ControlSpec):
                label_layout = self._make_widget(child)
                self._widgets[child.name] = label_layout

                if child.label is not None and child.type != "button" and child.type != "checkbox":
                    w_inner = label_layout
                    label_layout = QtWidgets.QHBoxLayout()
                    label_layout.addWidget(QtWidgets.QLabel(child.label))
                    label_layout.addWidget(w_inner)
                    label_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                    layout.addLayout(label_layout)
                else:
                    layout.addWidget(label_layout)
            else:
                layout.addLayout(self._build_layout(child))
        return layout

    def _make_widget(self, spec: ControlSpec) -> QtWidgets.QWidget:
        if spec.type == "combo" or spec.type == "combo-edit":
            w = QtWidgets.QComboBox()
            w.setEditable(spec.name == "quantity")

            w.addItems(spec.options or [])
            w.setCurrentText(spec.value)
            edited_callback = lambda: self._on_changed(spec.callback, w.currentText())
            w.currentIndexChanged.connect(edited_callback)
            if spec.type == "combo-edit":
                w.setLineEdit(MyLineEdit())
                w.lineEdit().editingFinished.connect(edited_callback)

        elif spec.type == "checkbox":
            w = QtWidgets.QCheckBox(spec.label or "")
            w.setChecked(bool(spec.value))
            w.stateChanged.connect(
                lambda st, cb=spec.callback: self._on_changed(cb, bool(st))
            )
        elif spec.type == "range_slider":
            w = QLabeledDoubleRangeSliderWithAutoscale()
            w.setRange(*(spec.range or (0.0, 1.0)))
            w.setValue(tuple(spec.value))
            w.valueChanged.connect(
                lambda _, cb=spec.callback, widget=w:
                    self._on_changed(cb, widget.value())
            )
        elif spec.type == "slider":
            w = QLabeledDoubleSlider()
            w.setRange(*(spec.range or (0.0, 1.0)))
            w.setValue(spec.value)
            w.valueChanged.connect(
                lambda _, cb=spec.callback, widget=w:
                    self._on_changed(cb, widget.value())
            )
        elif spec.type == "button":
            w = QtWidgets.QPushButton(spec.label or "")
            w.setStyleSheet("color: black;")  # unclear why this is necessary
            w.pressed.connect(lambda cb=spec.callback: self._on_changed(cb, None))
        elif spec.type == "color_picker":
            w = QtWidgets.QPushButton()
            w.setText("")
            w.setStyleSheet(f"background-color: {spec.value};")
            original_color = spec.value
            def pick_color():
                dialog = QtWidgets.QColorDialog(self)
                dialog.setCurrentColor(spec.value)
                dialog.setWindowTitle(spec.name)
                dialog.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
                def on_color_changed(color):
                    if color.isValid():
                        w.setStyleSheet(f"background-color: {color.name()};")
                        self._on_changed(spec.callback, color.name())
                dialog.currentColorChanged.connect(on_color_changed)
                if not dialog.exec():
                    w.setStyleSheet(f"background-color: {original_color};")
                    self._on_changed(spec.callback, original_color)

            w.clicked.connect(pick_color)
        else:
            w = QtWidgets.QLabel(f"Unknown control {spec.name}")

        return w

    def _on_changed(self, callback: Callable[[Any], None], value: Any):
        callback(value)

    @classmethod
    def _clear_layout(cls, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                cls._clear_layout(child_layout)
                child_layout.setParent(None)
        
    def _rebuild_ui(self, root: LayoutSpec):
        self._clear_layout(self._layout)
        self._widgets = {}
        QtWidgets.QWidget().setLayout(self._layout)
        self._layout = self._build_layout(root)
        self.setLayout(self._layout)
        self._update_screen_size_and_position()

    def _update_ui(self, root: LayoutSpec):
        if isinstance(root, ControlSpec):
            w = self._widgets.get(root.name)
            if not w:
                return
            if root.type == "combo" or root.type == 'combo-edit':
                w.blockSignals(True)
                w.setCurrentText(root.value)
                w.blockSignals(False)
            elif root.type == "checkbox":
                w.blockSignals(True)
                w.setChecked(root.value)
                w.blockSignals(False)
            elif root.type == "range_slider":
                w.blockSignals(True)
                w.setRange(*(root.range or (0, 1)))
                w.setValue(tuple(root.value))
                w.blockSignals(False)
            elif root.type == "slider":
                w.blockSignals(True)
                w.setRange(*(root.range or (0, 1)))
                w.setValue(root.value)
                w.blockSignals(False)
        else:
            for c in root.children:
                self._update_ui(c)

    def _refresh_ui(self, root: LayoutSpec, new_widgets: bool):
        if new_widgets:
            self._rebuild_ui(root)
        else:
            self._update_ui(root)

