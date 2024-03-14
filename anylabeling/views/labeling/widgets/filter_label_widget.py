from PyQt5.QtWidgets import QWidget, QHBoxLayout, QComboBox,QCheckBox
from PyQt5.QtCore import Qt

class LabelFilterComboBox(QWidget):
    def __init__(self, parent=None, items=[]):
        super(LabelFilterComboBox, self).__init__(parent)
        self.items = items
        self.combo_box = QComboBox()
        self.combo_box.addItems(self.items)
        self.combo_box.currentIndexChanged.connect(
            parent.combo_selection_changed
        )
        self.checkbox = QCheckBox("Track", self)
        layout = QHBoxLayout()
        layout.addWidget(self.combo_box)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def update_items(self, items):
        self.items = items
        self.combo_box.clear()
        self.combo_box.addItems(self.items)

    def update_track_id(self, track_id):
        self.checkbox.setText(f"Track: {track_id}")