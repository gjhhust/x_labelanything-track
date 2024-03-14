import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel

class NewTrackWindow(QtWidgets.QDialog):
    def __init__(self, class_name_list):
        super(NewTrackWindow, self).__init__()
        

        self.class_name_list = class_name_list
        self.result = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        for class_name in self.class_name_list:
            button = QPushButton(class_name)
            button.clicked.connect(lambda ch, cn=class_name: self.on_button_click(cn))
            layout.addWidget(button)

        self.setLayout(layout)

    def on_button_click(self, class_name):
        if class_name in self.track_id_dict:
            next_number = self.track_id_dict[class_name][-1] + 1
        else:
            next_number = 1

        self.result = "{}_{}".format(class_name, next_number)
        self.accept()
    
    def gengrate_track(self, track_ids_list):
        track_id_dict = {}
        for track_id in track_ids_list:
            parts = track_id.split('_')
            group_id = int(parts.pop())
            label = '_'.join(parts)
            if label not in  track_id_dict:
                track_id_dict[label] = []
            track_id_dict[label].append(group_id)
        return track_id_dict

    def exec_(self, track_ids_list):
        self.track_id_dict = self.gengrate_track(track_ids_list)

        for k,v in self.track_id_dict.items():
            if len(v) > 1:
                self.track_id_dict[k] = sorted(self.track_id_dict[k])
        result = super(NewTrackWindow, self).exec_()
        return result