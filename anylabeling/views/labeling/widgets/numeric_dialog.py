from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt,QEvent
import os

class NumberedInputDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(NumberedInputDialog, self).__init__(parent)

        self.input_fields = []
        self.result_values = []
        self.current_input_field = None  # 用于记录当前选中的输入框

        # Set the title for the dialog
        self.setWindowTitle("数字快捷键追踪id设置")

        # Create a QVBoxLayout to hold the input fields and additional elements
        self.layout = QtWidgets.QVBoxLayout(self)
        
        # 创建悬浮窗口
        self.tooltip = QDialog(self)
        self.tooltip.setWindowTitle("提示")

        # 设置始终置顶标志
        self.tooltip.setWindowFlags(self.tooltip.windowFlags() | Qt.WindowStaysOnTopHint)

        tooltip_layout = QVBoxLayout(self.tooltip)
        self.tooltip_label = QLabel(self.tooltip)
        tooltip_layout.addWidget(self.tooltip_label)


        # 创建十个带数字标签的输入框
        for i in range(1, 11):
            if i == 10:
                i = 0
            label = QtWidgets.QLabel(f"{i}:", self)
            input_field = QtWidgets.QLineEdit(self)
            self.input_fields.append(input_field)

            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(label)
            layout.addWidget(input_field)
            self.layout.addLayout(layout)
            # 连接textChanged信号，实时更新悬浮窗口的内容
            input_field.textChanged.connect(self.show_tooltip)
            # 安装事件过滤器
            input_field.installEventFilter(self)

        # Add explanatory text at the bottom
        explanation_label = QtWidgets.QLabel(
            "设置好后，在追踪模式下按对应数字按键可以直接将标注加入对应track，如果需要修改快捷键按L键", self
        )
        self.layout.addWidget(explanation_label)

        # 创建确定和取消按钮
        ok_button = QtWidgets.QPushButton("OK", self)
        ok_button.clicked.connect(self.accept)

        cancel_button = QtWidgets.QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        # 创建删除按钮
        delete_button = QtWidgets.QPushButton("Delete", self)
        delete_button.clicked.connect(self.delete_current_input_field)

        # 将删除按钮添加到按钮布局中 
        button_layout.addWidget(delete_button)

        # Add the button layout to the main layout
        self.lable_button_layout = QtWidgets.QGridLayout()
        self.lable_button_cnt = 0

        self.layout.addLayout(button_layout)
        self.layout.addLayout(self.lable_button_layout)

        self.result_values1 = []
        self.result_values2 = [
            "move_car",
            "static_car",
            "move_people",
            "static_people",
            "move_bicycle",
            "static_bicycle",
            "static_peoplebicycle",
            "bicycleCrowd",
            "peopleCrowd",
            "ignore",
        ]

        # 将窗口设置为悬浮窗口
        self.setWindowFlags(Qt.Window | Qt.Tool)


    def update_save_dir(self,dir):
        self.value1_path = os.path.join(dir,"values", "values1.txt")
        self.value2_path = os.path.join(dir,"values", "values2.txt")

        self.result_values1 = self.read_values(self.value1_path)
        self.result_values2 = self.read_values(self.value2_path)

    def save_values(self):
        with open(self.value1_path, 'w') as file:
            for value in self.result_values1:
                file.write(value + '\n')

        with open(self.value2_path, 'w') as file:
            for value in self.result_values2:
                file.write(value + '\n')

    def read_values(self,path):
        # 判断文件是否存在
        os.makedirs(os.path.dirname(path),exist_ok=True)
        if os.path.exists(path):
            print(f"read path: {path}")
            with open(path, 'r') as file:
                return [line.strip() for line in file.readlines()]
        else:
            print(f"create path: {path}")
            with open(path, 'w') as file:
                pass
            return []
        
    def delete_current_input_field(self):
        if self.current_input_field:
            current_index = self.input_fields.index(self.current_input_field)

            # 清空当前输入框的内容
            self.current_input_field.clear()

            # 将后续的输入框前移
            for i in range(current_index, len(self.input_fields) - 1):
                self.input_fields[i].setText(self.input_fields[i + 1].text())

            # 清空最后一个输入框
            self.input_fields[-1].clear()
            self.input_fields[-1].setFocus()

    def eventFilter(self, obj, event):
        # 事件过滤器，处理焦点事件
        if event.type() == QEvent.FocusIn and obj in self.input_fields:
            self.current_input_field = obj
        return super().eventFilter(obj, event)
    
    def create_buttons(self):
        # 创建按钮并连接到槽函数
        for label_key in self.labels.keys():
            # Check if a button with the same label_key already exists
            existing_button = self.findChild(QtWidgets.QPushButton, label_key)

            if existing_button:
                continue
            else:
                button = QtWidgets.QPushButton(label_key, self)
                # 使用setObjectName为按钮设置唯一标识符
                button.setObjectName(label_key)
                button.clicked.connect(lambda _, key=label_key: self.on_button_clicked(key))

            # 将按钮添加到布局中
            col = self.lable_button_cnt % 3
            row = self.lable_button_cnt // 3
            self.lable_button_layout.addWidget(button,row, col)
            self.lable_button_cnt += 1
            

    def on_button_clicked(self, key):
        if self.current_input_field and self.current_input_field in self.input_fields:
            group_id = self.labels[key][-1] + 1 if self.labels[key][-1] else 1
            self.labels[key].append(group_id)
            self.current_input_field.setText(key + "_" + str(group_id))
            
        # 恢复焦点到之前选中的输入框
        if self.current_input_field and self.current_input_field in self.input_fields:
            self.current_input_field.setFocus()

    def select_shape(self, label, group_id):
        if self.current_input_field and self.current_input_field in self.input_fields and group_id is not None:
            self.labels[label].append(group_id)
            self.current_input_field.setText(label + "_" + str(group_id))
            
        # 恢复焦点到之前选中的输入框
        if self.current_input_field and self.current_input_field in self.input_fields:
            self.current_input_field.setFocus()

    def update_lables(self,labels):
        self.labels = {}
        for item in labels:
            if item.shape().label not in self.labels:
                self.labels[item.shape().label] = [None]
            if item.shape().group_id is not None:
                self.labels[item.shape().label].append(item.shape().group_id)

        for k,v in self.labels.items():
            if len(v) > 1:
                self.labels[k] = sorted(self.labels[k][1:])

    def show_tooltip(self):
        # 获取输入的内容
        values = [field.text() for field in self.input_fields]

        # 在悬浮窗口中显示内容
        tooltip_text = "\n".join([f"{i}: {value}" for i, value in enumerate(values, start=1)])
        self.tooltip_label.setText(tooltip_text)
        self.tooltip.setGeometry(self.geometry().bottomRight().x(), self.geometry().bottomRight().y(), 200, 100)
        self.tooltip.show()


    def exec_(self, use_first_set,label_list):
        self.update_lables(label_list)

        if use_first_set:
            # 填充track结果到输入框
            for input_field in self.input_fields:
                input_field.clear()
            for field, value in zip(self.input_fields, self.result_values1):
                field.setText(value)
        else:
            # 填充第二套结果到输入框
            for input_field in self.input_fields:
                input_field.clear()
            for field, value in zip(self.input_fields, self.result_values2):
                field.setText(value)

        self.create_buttons()

        result = super(NumberedInputDialog, self).exec_()

        if result == QtWidgets.QDialog.Accepted:
            # 获取输入框的内容
            self.result_values = [field.text() for field in self.input_fields]
            if use_first_set:
                self.result_values1 = [field.text() for field in self.input_fields]
            else:
                self.result_values2 = [field.text() for field in self.input_fields]

            self.save_values()

        return result

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    input_dialog = NumberedInputDialog()
    result = input_dialog.exec_(False)

    if result == QtWidgets.QDialog.Accepted:
        print("Result Values:", input_dialog.result_values)

    app.exec_()
