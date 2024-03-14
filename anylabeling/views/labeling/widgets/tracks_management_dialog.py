from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
class TracksManagementDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TracksManagementDialog, self).__init__(parent)

        self.start_frame_input = QtWidgets.QLineEdit(self)
        self.end_frame_input = QtWidgets.QLineEdit(self)

        # Set the title for the dialog
        self.setWindowTitle("输入视频帧范围")

        # Create a QVBoxLayout to hold the input fields and additional elements
        self.layout = QtWidgets.QVBoxLayout(self)

        # 创建提示信息
        self.info_label = QtWidgets.QLabel("输入视频帧范围:", self)

        # 创建帧范围输入框
        start_label = QtWidgets.QLabel("开始帧号:", self)
        end_label = QtWidgets.QLabel("结束帧号:", self)

        start_layout = QtWidgets.QHBoxLayout()
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_frame_input)

        end_layout = QtWidgets.QHBoxLayout()
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_frame_input)

        # 添加提示信息和输入框到主布局
        self.layout.addWidget(self.info_label)
        self.layout.addLayout(start_layout)
        self.layout.addLayout(end_layout)

        # 创建确定和取消按钮
        ok_button = QtWidgets.QPushButton("导出tack预览视频", self)
        ok_button.clicked.connect(self.accept)

        cancel_button = QtWidgets.QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        # 添加按钮布局到主布局
        self.layout.addLayout(button_layout)

        # 创建Tracks列表
        self.tracks_list = QtWidgets.QListWidget(self)
        self.tracks_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.layout.addWidget(self.tracks_list)

        # 添加Tracks操作按钮
        tracks_button_layout = QtWidgets.QHBoxLayout()
        
        get_button = QtWidgets.QPushButton("聚合视频tracks", self)
        get_button.clicked.connect(self.get_all_track)

        edit_button = QtWidgets.QPushButton("编辑选定的Track", self)
        edit_button.clicked.connect(self.edit_selected_track)
        
        interpolate_button = QtWidgets.QPushButton("插值选定的Track", self)
        interpolate_button.clicked.connect(self.interpolate_selected_tracks)
        
        check_button = QtWidgets.QPushButton("检查选定的Track", self)
        check_button.clicked.connect(self.check_selected_tracks)
        
        merge_button = QtWidgets.QPushButton("合并选定的Tracks", self)
        merge_button.clicked.connect(self.merge_selected_tracks)

        # export_video_button = QtWidgets.QPushButton("导出tack预览视频", self)
        # export_video_button.clicked.connect(self.export_video)

        tracks_button_layout.addWidget(get_button)
        tracks_button_layout.addWidget(edit_button)
        tracks_button_layout.addWidget(interpolate_button)
        tracks_button_layout.addWidget(check_button)
        tracks_button_layout.addWidget(merge_button)
        # tracks_button_layout.addWidget(export_video_button)
        
        self.layout.addLayout(tracks_button_layout)

        self.result_start_frame = None
        self.result_end_frame = None
        self.get_tracks = None

    def exec_(self, max_len):
        result = super(TracksManagementDialog, self).exec_()
        self.info_label.setText(f"输入视频帧范围:0 - {max_len-1}")
        if result == QtWidgets.QDialog.Accepted:
            # 获取输入框的内容
            # self.result_start_frame = int(self.start_frame_input.text())
            # self.result_end_frame = int(self.end_frame_input.text())
            self.result_start_frame = 0
            self.result_end_frame = max_len
            bool_file, file_name = self.export_video()
        # self.tracks_dict = tracks_dict
        # # 更新Tracks列表
        # self.tracks_list.clear()
        # self.tracks_list.addItems(self.tracks_dict.keys())

        return bool_file, file_name
    
    def get_all_track(self):
        self.get_tracks()

    def export_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "保存预览视频", "", "视频文件 (*.mp4);;所有文件 (*)", options=options)

        if file_name:
            # 用户选择了文件名
            print("选择的文件路径:", file_name)
            return True, file_name
            # 在这里你可以将视频导出到选择的文件路径
            # 例如，你可以使用OpenCV等库来创建并保存视频
            # 这里只是一个示例，具体的实现取决于你使用的库和导出视频的逻辑
        else:
            # 用户取消了选择
            print("用户取消了选择")
            return False, None

    def edit_selected_track(self):
        selected_items = self.tracks_list.selectedItems()
        for item in selected_items:
            track_id = item.text()
            # 获取选定Track的frame_range等信息
            track_info = self.tracks_dict.get(track_id, {})
            # 调用修改track_id和重用名的函数

    def interpolate_selected_tracks(self):
        selected_items = self.tracks_list.selectedItems()
        for item in selected_items:
            track_id = item.text()
            # 获取选定Track的frame_range等信息
            track_info = self.tracks_dict.get(track_id, {})
            # 调用插值函数

    def check_selected_tracks(self):
        selected_items = self.tracks_list.selectedItems()
        for item in selected_items:
            track_id = item.text()
            # 获取选定Track的frame_range等信息
            track_info = self.tracks_dict.get(track_id, {})
            # 调用检查函数

    def merge_selected_tracks(self):
        selected_items = self.tracks_list.selectedItems()
        selected_tracks = [item.text() for item in selected_items]
        # 调用合并函数，可以弹出一个新的对话框让用户输入合并后的track_id等信息
        # ...
