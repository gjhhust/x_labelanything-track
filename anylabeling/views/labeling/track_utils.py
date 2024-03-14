'''
增加按照轨迹操作的模式和行为，（track_id从group id得到，且每一帧是唯一的）
'''
import os
from anylabeling.services.auto_labeling.types import AutoLabelingMode
from copy import deepcopy
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import os,cv2,json
import glob
from .utils import (find_perspective_transform, apply_bboxes_xyxy,apply_bbox, 
                    apply_bbox_xyxy, 
                    fix_bbox_xyxy,save_difference_image)
import os.path as osp
from collections import deque
from PyQt5 import QtCore, QtGui, QtWidgets
import random,shutil
from datetime import datetime


import cv2,threading
import multiprocessing
import os.path as osp
import random
from PyQt5.QtWidgets import  QWidget, QVBoxLayout, QPushButton, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt

ALL_RECAL = "ALL_RECAL"
CAL_PREDICTION = "CAL_PREDICTION"
CAL_INTERPOLATION = "CAL_INTERPOLATION"
NO_CAL = "NO_CAL"

try:
    from metaseg import SegManualMaskPredictor
    has_seg_manual_mask_predictor = True
    print("can use seg")
except ImportError:
    has_seg_manual_mask_predictor = False
    print("not use seg")

##########################计算匹配矩阵的线程############################################   
# 多进程计算图像匹配
def process_homoMatrixes_worker(args):
    image_list_dict, ref_frame, target_frame = args

    ref_img = cv2.imread(image_list_dict[ref_frame])
    tag_img = cv2.imread(image_list_dict[target_frame])
    matrix = find_perspective_transform(ref_img, tag_img)

    return matrix,target_frame,ref_frame

class CalHomoMatrixWorkerThread(QThread):
    progress_update = pyqtSignal(int)
    result_ready = pyqtSignal(str)

    def __init__(self, matrix_dict, image_list_dict, frame_pairs, save=True, num_processes=4):
        super().__init__()
        self.matrix_dict = matrix_dict
        self.image_list_dict = image_list_dict
        self.frame_pairs = frame_pairs
        self.save = save
        self.num_processes = num_processes
        self.cancelled = False

    def run(self):
        try:
            self.process_homoMatrixes()
            self.result_ready.emit("Calculation complete!")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                None,
                "Error",
                f"{e}",
                QtWidgets.QMessageBox.Ok,
            )
            

    def process_homoMatrixes(self):
        total_elements = len(self.frame_pairs)
        dir_name = osp.dirname(next(iter(self.image_list_dict.values())))
        dict_path = osp.join(dir_name, "values", "homoMatrix.npz")

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            args_list = [
                (self.image_list_dict, ref, target)
                for ref, target in self.frame_pairs
            ]

            for idx, (matrix, target_frame, ref_frame) in enumerate(pool.imap(process_homoMatrixes_worker, args_list), start=1):
                if self.cancelled:
                    break

                self.matrix_dict[target_frame][ref_frame] = matrix

                if self.save:
                    save_homodict(self.matrix_dict, dict_path)

                progress = int((idx / total_elements) * 100)
                self.progress_update.emit(progress)

                # 添加取消检查
                self.msleep(100)  # 避免线程过于密集检查取消状态
                if self.isCancelled():
                    break

    def cancel(self):
        self.cancelled = True

    def isCancelled(self):
        return self.cancelled
# main
# frame_pairs = [(ref1, target1), (ref2, target2), ...]  # 你的帧对列表
# matrix_dict = process_homoMatrixes(matrix_dict, image_list_dict, frame_pairs)
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QPushButton, QMessageBox

class ProgressWindow(QWidget):
    def __init__(self, worker_thread, parent=None):
        super().__init__(parent)
        self.setWindowTitle('计算进度')
        self.setGeometry(500, 500, 300, 150)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout(self)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.cancel_button = QPushButton('取消', self)
        self.cancel_button.clicked.connect(worker_thread.cancel)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def set_progress_value(self, value):
        self.progress_bar.setValue(value)

def backup_file(file_path):
    if os.path.exists(file_path):
        # 获取文件名和扩展名
        file_name, file_extension = os.path.splitext(file_path)
        
        # 获取当前日期
        current_date = datetime.now().strftime("%Y%m%d")
        
        cont = 0

        # 生成备份文件名
        backup_file_name = f"{file_name}_{current_date}{file_extension}"
        
        # 检查文件是否存在，如果存在则增加备份计数器
        while os.path.exists(backup_file_name):
            cont += 1
            backup_file_name = f"{file_name}_{current_date}_{cont}{file_extension}"

        # 备份文件
        shutil.copy2(file_path, backup_file_name)
        print(f"Backup created: {backup_file_name}")
    else:
        print("File not found.")

def save_homodict(matrix_dict, path):
    # 将字典中所有的数字键转换为字符串键
    matrix_dict_str_keys = {str(key): value for key, value in matrix_dict.items()}
    for key, value in matrix_dict_str_keys.items():
        matrix_dict_str_keys[key] = {str(k): v for k, v in value.items()}

    # 保存字典到文件
    np.savez(path, matrix_dict=matrix_dict_str_keys)

def read_homodict(path):
    if osp.exists(path):
        # 从文件中加载字典
        loaded_data = np.load(path, allow_pickle=True)

        # 获取加载的字典
        loaded_matrix_dict = loaded_data['matrix_dict'].item()

        # 将字符串键还原为数字键
        loaded_matrix_dict_original_keys = {int(key): value for key, value in loaded_matrix_dict.items()}
        for key, value in loaded_matrix_dict_original_keys.items():
            loaded_matrix_dict_original_keys[key] = {int(k): v for k, v in value.items()}
        return loaded_matrix_dict_original_keys
    return {}

def check_homoMatrixes(matrix_dict, image_list_dict,ref_frame, target_frame, save=True):
    if target_frame not in matrix_dict:
        matrix_dict[target_frame] = {}

    if ref_frame not in matrix_dict[target_frame]:
        ref_img = cv2.imread(image_list_dict[ref_frame])
        tag_img = cv2.imread(image_list_dict[target_frame])
        matrix = find_perspective_transform(ref_img, tag_img)
        matrix_dict[target_frame][ref_frame] = matrix

        dir_name = osp.dirname(image_list_dict[ref_frame])
        if save:
            dict_path = osp.join(dir_name, "values", "homoMatrix.npz")
            save_homodict(matrix_dict, dict_path)

        if random.random() < 0.05:
            h, w = ref_img.shape[:2]
            transformed_frame = cv2.warpPerspective(ref_img, matrix, (w, h))
            save_path = osp.join(dir_name, "values", f"homo_{target_frame}-{ref_frame}.png")
            save_difference_image(tag_img, transformed_frame, save_path) 
            print(f"save homo image in {save_path}")
    return matrix_dict


def count_png_files(directory_path):
    # 使用glob获取目录下所有后缀为.png的文件路径列表
    png_files = glob.glob(os.path.join(directory_path, '*.png'))

    # 获取文件数量
    num_png_files = len(png_files)

    return num_png_files

def is_bbox_out_of_bounds(bbox, image_width, image_height):
    """
    判断边界框是否超出图片边界
    """
    return bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= image_width or bbox[3] >= image_height



def bbox_area(bbox):
    # bbox 格式为 [x_min, y_min, x_max, y_max]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    return area
def bbox_ratio(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    # 计算最长边和最短边的比例
    if width > height:
        aspect_ratio = width / (height+0.001)
    else:
        aspect_ratio = height / (width+0.001)
    return aspect_ratio

def adjust_bbox_to_bounds(bbox, image_width, image_height, mean_area, max_ratio):
    """
    将边界框调整为图片边界内，如果调整后的面积超过原面积的1/2或者面积小于16，则返回None
    """
    x1, y1, x2, y2 = bbox

    # 截断边界框，使其保持在图片边界内
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    bbox = np.array([x1, y1, x2, y2],dtype=np.float32)

    # 计算原边界框面积和调整后边界框面积
    adjusted_area = bbox_area(bbox)

    aspect_ratio = bbox_ratio(bbox)

    # 如果调整后的面积超过原面积的1/2或者面积小于16，则返回None
    if adjusted_area < mean_area / 2 or adjusted_area < 36 or adjusted_area > mean_area*1.5 or aspect_ratio > 1.6*max_ratio:
        return None
    else:
        return np.array([x1, y1, x2, y2],dtype=np.float32)


class Track_object():
    def __init__(self, track_id, frame_number=None, shape_ann=None):
        """
        从某帧的注释中创建轨迹
        frame_number: int 帧号
        shape_ann；{
            points: list[n,2] 该注释围成的点，逆时针
            shape_type: str bbox注释则为rectangle
            label:str 该标注的分类
            group_id: int 该标注的分类内分配的独特id
            label_group_id组合成track_id
        }
        """
        if frame_number is not None:
            if not isinstance(shape_ann, dict):
                shape_ann = self.get_dict_ann(shape_ann)
            shape_ann = self.precess_ann(shape_ann)
            self.label = shape_ann["label"]
            self.group_id = shape_ann["group_id"]
            self.track_anns = {}
            self.repeate_anns = {}
            self.track_id = track_id
            self.track_anns[frame_number] = shape_ann
        else:
            self.track_anns = {}
            self.repeate_anns = {}
            self.track_id = track_id
            parts = track_id.split('_')
            self.group_id = int(parts.pop())
            self.label = '_'.join(parts)
            
        
    def get_dict_ann(self, shape):
        def format_shape(s):
            data = s.other_data.copy()
            info = {
                "label": s.label,
                "points": [(p.x(), p.y()) for p in s.points],
                "group_id": s.group_id,
                "description": "" if s.description is None else s.description,
                "difficult": s.difficult,
                "shape_type": s.shape_type,
                "flags": s.flags,
                "attributes": s.attributes,
            }
            if s.shape_type == "rotation":
                info["direction"] = s.direction
            data.update(info)

            return data
        return format_shape(shape)
                
    def precess_ann(self, ann):
        if ann["shape_type"] == "rectangle":
            ann["bbox"] = ann["points"][0] + ann["points"][2]
        if "inter" not in ann["description"]:
            ann["description"] = "ann" #来自ann或者interpolation
        return ann
    
    def get_ann(self, frame_number):
        return self.track_anns[frame_number]

    def get_inter_ann(self, frame_number):
        if frame_number in self.track_anns:
            ann = self.track_anns[frame_number]
            if "inter" in ann["description"]:
                return ann.copy()
            else: 
                return None
        else:
            return None

    def get_all_frame_pairs(self, matrix_dict, frame_range):
        if len(self.track_anns)==0:
            return []
        
        max_image_index = frame_range[-1]
        # 得到有标注的范围
        frame_range[0] = max(frame_range[0], min(self.track_anns.keys()))
        frame_range[1] = min(max(self.track_anns.keys()), frame_range[1]) + 1
        # #有标注的范围扩充10帧，因为标注间隔10，中间可能才是当前track的开始或者结尾
        # frame_range[0] = max(0, frame_range[0]-10)
        # frame_range[1] = min(max_image_index-1, frame_range[1]+10)

        frame_numbers = sorted(self.track_anns.keys())

        if len(frame_numbers)<2:
            return []
        
        pairs = []
        for frame_number in range(frame_range[0],frame_range[1]):
            ann_cur = self.track_anns.get(frame_number, {})
            description = ann_cur.get("description", "")
            if len(ann_cur)>0 and "inter" not in description: #无需插值
                # Frame already has annotation, no interpolation needed
                pass
            else:
                # Find the nearest frames with annotations
                prev_frame = max(filter(lambda f: f < frame_number, frame_numbers), default=None)
                next_frame = min(filter(lambda f: f > frame_number, frame_numbers), default=None)

                if next_frame - prev_frame > 15: #前后两帧差距过大则不插值
                    continue
                if frame_number not in matrix_dict:
                    matrix_dict[frame_number] = {}
                if prev_frame not in matrix_dict[frame_number]:
                    pairs.append((prev_frame, frame_number))
                if next_frame not in matrix_dict[frame_number]:
                    pairs.append((next_frame, frame_number))
        return pairs

    def linear_interpolation(self, frame_range, track_anns, width, height, Seg = None, Seg_predict = None):
        frame_numbers = sorted([key for key, value in track_anns.items() if 'inter' not in value.get('description', '')])

        if len(frame_numbers)<2:
            return track_anns
         
        for frame_number in frame_range:
            try:
                ann_cur = track_anns.get(frame_number, {})
                description = ann_cur.get("description", "")
                if self.cal_mode == NO_CAL:#存在的全部不用计算
                    condition = len(ann_cur)>0
                elif self.cal_mode == ALL_RECAL:#全部重新计算
                    condition = len(ann_cur)>0 and ("inter" not in description)
                elif self.cal_mode == CAL_PREDICTION: #重新计算预测框
                    condition = len(ann_cur)>0 and ("predict_inter" not in description)
                elif self.cal_mode == CAL_INTERPOLATION: #重新计算插值框
                    condition = len(ann_cur)>0 and ("ann_inter" not in description)
                
                if condition:
                    # Frame already has annotation, no interpolation needed
                    pass
                else:
                    prev_frame = max(filter(lambda f: f < frame_number, frame_numbers), default=None)
                    next_frame = min(filter(lambda f: f > frame_number, frame_numbers), default=None)
                    cur_ann = track_anns[frame_numbers[0]].copy()

                    # 情况1：frame_number小于所有标注帧
                    if prev_frame is None:
                        # 取最接近的两帧，基于它们的相对位置估算当前帧的边界框
                        prev_frame, next_frame = sorted(track_anns.keys())[:2]
                        
                        if next_frame - prev_frame > 15: #前后两帧差距过大则不插值
                            continue

                        # 假设track_anns是包含每一帧注释的字典
                        prev_ann = np.array(track_anns[prev_frame]['bbox'], dtype=np.float32)
                        next_ann = np.array(track_anns[next_frame]['bbox'], dtype=np.float32)

                        # self.homoMatrixes = check_homoMatrixes(self.homoMatrixes, self.image_list_dict, prev_frame, frame_number)
                        # self.homoMatrixes = check_homoMatrixes(self.homoMatrixes, self.image_list_dict, next_frame, frame_number)
                        # transformed_prev_ann = apply_bbox_xyxy(self.homoMatrixes[frame_number][prev_frame], prev_ann)
                        # transformed_next_ann = apply_bbox_xyxy(self.homoMatrixes[frame_number][next_frame], next_ann)
                        transformed_prev_ann = prev_ann
                        transformed_next_ann = next_ann

                        # 基于nearst1和nearst2之间的差异估算速度
                        velocity = (transformed_next_ann - transformed_prev_ann) / (next_frame - prev_frame)

                        # 基于速度估算当前帧的边界框
                        interpolated_bbox = transformed_prev_ann + velocity * (frame_number - prev_frame)

                        interpolated_bbox = adjust_bbox_to_bounds(interpolated_bbox, width, height, self.mean_area,self.max_ratio)
                        if interpolated_bbox is None:
                            break

                        if Seg_predict is not None:
                            interpolated_bbox = fix_bbox_xyxy(Seg_predict, interpolated_bbox, cv2.imread(self.image_list_dict[frame_number]))
                            cur_ann['description'] = "sam_predict_inter"
                        else:
                            cur_ann['description'] = "predict_inter"
                        
                    # 情况2：frame_number大于所有标注帧
                    elif next_frame is None:
                        # 取最接近的两帧，基于它们的相对位置估算当前帧的边界框
                        prev_frame, next_frame = sorted(track_anns.keys())[-2:]

                        if next_frame - prev_frame > 15: #前后两帧差距过大则不插值
                            continue

                        prev_ann = np.array(track_anns[prev_frame]['bbox'], dtype=np.float32)
                        next_ann = np.array(track_anns[next_frame]['bbox'], dtype=np.float32)

                        # self.homoMatrixes = check_homoMatrixes(self.homoMatrixes, self.image_list_dict, prev_frame, frame_number)
                        # self.homoMatrixes = check_homoMatrixes(self.homoMatrixes, self.image_list_dict, next_frame, frame_number)
                        # transformed_prev_ann = apply_bbox_xyxy(self.homoMatrixes[frame_number][prev_frame], prev_ann)
                        # transformed_next_ann = apply_bbox_xyxy(self.homoMatrixes[frame_number][next_frame], next_ann)
                        transformed_prev_ann = prev_ann
                        transformed_next_ann = next_ann

                        # 基于nearst1和nearst2之间的差异估算速度
                        velocity = (transformed_next_ann - transformed_prev_ann) / (next_frame - prev_frame)

                        # 基于速度估算当前帧的边界框
                        interpolated_bbox = transformed_next_ann + velocity * (frame_number - next_frame)

                        interpolated_bbox = adjust_bbox_to_bounds(interpolated_bbox, width, height, self.mean_area,self.max_ratio)
                        if interpolated_bbox is None:
                            break
                        
                        if Seg_predict is not None:
                            interpolated_bbox = fix_bbox_xyxy(Seg_predict, interpolated_bbox, cv2.imread(self.image_list_dict[frame_number]))
                            cur_ann['description'] = "sam_predict_inter"
                        else:
                            cur_ann['description'] = "predict_inter"

                    elif prev_frame is not None and next_frame is not None:

                        if next_frame - prev_frame > 15: #前后两帧差距过大则不插值
                            continue
                        self.homoMatrixes = check_homoMatrixes(self.homoMatrixes, self.image_list_dict, prev_frame, frame_number)
                        self.homoMatrixes = check_homoMatrixes(self.homoMatrixes, self.image_list_dict, next_frame, frame_number)

                        # Perform linear interpolation
                        prev_ann = track_anns[prev_frame]['bbox']
                        next_ann = track_anns[next_frame]['bbox']

                        transformed_prev_ann = apply_bbox_xyxy(self.homoMatrixes[frame_number][prev_frame], prev_ann)
                        transformed_next_ann = apply_bbox_xyxy(self.homoMatrixes[frame_number][next_frame], next_ann)

                        alpha = (frame_number - prev_frame) / (next_frame - prev_frame)
                        interpolated_bbox = [
                            float(prev + alpha * (next - prev)) for prev, next in zip(transformed_prev_ann, transformed_next_ann)
                        ]

                        if Seg is not None:
                            interpolated_bbox = fix_bbox_xyxy(Seg, interpolated_bbox, cv2.imread(self.image_list_dict[frame_number]))
                        cur_ann['description'] = "ann_inter"
                    
                    cur_ann['bbox'] = interpolated_bbox
                    cur_ann['points'] = bbox_xyxy_to_points(interpolated_bbox)
                    track_anns[frame_number] = cur_ann
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    None,
                    "Error",
                    f"{e}",
                    QtWidgets.QMessageBox.Ok,
                )


        return track_anns

    def gaussian_smooth_interpolation(self,frame_range, track_anns, sigma=1):
        interpolated_anns = {}
        frame_numbers = sorted(track_anns.keys())

        for frame_number in range(frame_range[0],frame_range[1]):
            if frame_number in track_anns:
                # Frame already has annotation, no interpolation needed
                interpolated_anns[frame_number] = track_anns[frame_number]
            else:
                # Find the nearest frames with annotations
                nearest_frames = sorted(frame_numbers, key=lambda f: abs(f - frame_number))[:2]
                if len(nearest_frames) == 2:
                    # Perform Gaussian smoothing
                    nearest_anns = [track_anns[f]['bbox'] for f in nearest_frames]
                    smoothed_bbox = [
                        val for val in gaussian_filter1d(nearest_anns, sigma=sigma, axis=0)[0]
                    ]

                    cur_ann = nearest_anns[0].copy()
                    cur_ann['bbox'] = smoothed_bbox
                    cur_ann['description'] = "ann_inter"

                    interpolated_anns[frame_number] = cur_ann

        return interpolated_anns

    def get_object_mean_info(self):  
        # 获取标注框的特征
        areas = []
        ratio = []
        for value in self.track_anns.values():
            if "inter" not in value["description"]:
                areas.append(bbox_area(value["bbox"]))
                ratio.append(bbox_ratio(value["bbox"]))

        max_number = min(4, len(ratio)-1)

        if len(areas) == 0:
            QtWidgets.QMessageBox.warning(
                        None,
                        "Error",
                        f"track:【{self.track_id}】 no ann by person, please check this track_id",
                        QtWidgets.QMessageBox.Ok,
                    )

        return sum(areas) / len(areas), max(sorted(ratio)[-max_number:])
    
    def interpolate(self,Seg,Seg_predict, homoMatrixes, image_list_dict, frame_range, predict_number=0, cal_mode = ALL_RECAL):#linear gaussian
        """
        homoMatrixes是图像变换矩阵
        对轨迹进行插值
        """
        self.cal_mode = cal_mode
        self.homoMatrixes = homoMatrixes
        self.image_list_dict = image_list_dict
        if len(self.track_anns)==0:
            return homoMatrixes
        
        self.mean_area, self.max_ratio = self.get_object_mean_info()
        min_image_index = frame_range[0]
        max_image_index = frame_range[-1]
        # 得到有标注的范围
        frame_range[0] = max(frame_range[0], min(self.track_anns.keys()))
        frame_range[1] = min(max(self.track_anns.keys()), frame_range[1]) + 1
        # #有标注的范围扩充10帧，因为标注间隔10，中间可能才是当前track的开始或者结尾
        # frame_range[0] = max(0, frame_range[0]-10)
        # frame_range[1] = min(max_image_index-1, frame_range[1]+10)

        image = cv2.imread(next(iter(self.image_list_dict.values())))
        height, width, channels = image.shape

        self.track_anns = self.linear_interpolation(range(frame_range[0],frame_range[1]), self.track_anns, width, height, Seg, None)
        # 推理第一帧之前10帧
        self.track_anns = self.linear_interpolation(range(frame_range[0]-1, max(min_image_index, frame_range[0]-predict_number), -1), self.track_anns, width, height, Seg, Seg_predict)
        # 推理最后一帧之后10帧
        self.track_anns = self.linear_interpolation(range(frame_range[1], min(max_image_index-1, frame_range[1]+predict_number)), self.track_anns, width, height, Seg, Seg_predict)

        return self.homoMatrixes

    def add(self,frame_number, ann):
        """
        将某个标注加入当前track
        """
        repeate = False
        if not isinstance(ann, dict):
            ann.label = self.label
            ann.group_id = self.group_id
            ann_dict = self.get_dict_ann(ann)
        else:
            ann["label"] = self.label
            ann["group_id"] = self.group_id
            ann_dict = ann
        if frame_number in self.track_anns:
            repeate = True
            if frame_number not in self.repeate_anns:
                self.repeate_anns[frame_number] = []
            self.repeate_anns[frame_number].append(self.track_anns[frame_number].copy())

        self.track_anns[frame_number] = self.precess_ann(ann_dict)
        return self.label, self.group_id, repeate

    def process_repeate(self,frame_number):
        if frame_number in self.repeate_anns:
            repeate_ann = list(self.repeate_anns[frame_number])
            del self.repeate_anns[frame_number]
            return repeate_ann

    def delete(self):
        """
        将某个标注删除当前track
        """    
        pass

    def check_repeate(self,frame_number):
        """
        检查重复
        """
        pass

def create_new_track(track_lists, label_str):
    """
    创建label_str内不重复的新track，加入track_lists,并返回track_id
    """
    pass


import colorsys

def number_to_color(number):
    # Map the number to the range [0, 20]
    normalized_number = number % 20
    
    # Map the normalized number to the range [0, 1]
    normalized_number /= 20.0
    
    # Map the normalized number to HSL color
    hue = normalized_number * 360.0
    saturation = 100.0
    lightness = 50.0
    
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(hue / 360.0, lightness / 100.0, saturation / 100.0)
    
    # Scale RGB values to the range [0, 255]
    rgb = tuple(int(value * 255) for value in rgb)
    
    return rgb

def bbox_xyxy_to_points(bbox):#xyxy
        points = []
        points.append([bbox[0],bbox[1]])
        points.append([bbox[2],bbox[1]])
        points.append([bbox[2],bbox[3]])
        points.append([bbox[0],bbox[3]])
        return points

def bbox_xywh_to_points(bbox):#xywh
        x1,y1,w,h = bbox
        x2,y2 = x1+w,y1+h
        points = []
        points.append([x1,y1])
        points.append([x2,y1])
        points.append([x2,y2])
        points.append([x1,y2])
        return points

def coco_ann_to_shape(coco_ann):
    track_id = coco_ann["track_id"]
    parts = track_id.split('_')
    group_id = int(parts.pop())
    label = '_'.join(parts)
    shape_ann = {
        "label": label,
        "points": bbox_xywh_to_points(coco_ann["bbox"]),
        "group_id": group_id,
        "description":"" if coco_ann["description"] is None else coco_ann["description"],
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "other_data": {}
    }
    return shape_ann



from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QInputDialog, QPushButton

from PyQt5.QtWidgets import QInputDialog, QCheckBox, QComboBox, QVBoxLayout, QLineEdit
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

class InputIntervalDialog(QtWidgets.QDialog):
    def __init__(self, has_seg_manual_mask_predictor):
        super().__init__()
        
        self.init_ui(has_seg_manual_mask_predictor)

    def init_ui(self, has_seg_manual_mask_predictor):
        
        # Create a QLineEdit for numeric input
        self.interval_input = QLineEdit(self)
        # Set a validator to allow only integers in the range [1, 9]
        self.interval_input.setValidator(QtGui.QIntValidator(1, 9))
        self.interval_input.setText("1")
        self.interval_input.setToolTip("输入插值间隔帧，数字越大插值显示越快\n推荐首次输入3-6，快速浏览插值后的图像正确与否，确认无误后输入1使得每一帧都插值\n（输入范围1-9）")

        # Create a QLineEdit for numeric input
        self.predict_input = QLineEdit(self)
        # Set a validator to allow only integers in the range [0, 9]
        self.predict_input.setValidator(QtGui.QIntValidator(0, 9))
        self.predict_input.setText("5")
        self.predict_input.setToolTip("将每条轨迹向前向后延拓n帧补全漏标，0则不延拓")


        # 添加插值Sam修复的勾选框
        self.sam_repair_checkbox = QCheckBox("插值Sam修复")
        self.sam_repair_checkbox.setChecked(False)
        self.sam_repair_checkbox.setToolTip("中间插值得到的bbox使用sam进行修复bbox")

        # 添加预测Sam修复的勾选框
        self.prediction_repair_checkbox = QCheckBox("预测Sam修复")
        self.prediction_repair_checkbox.setChecked(True)
        self.prediction_repair_checkbox.setToolTip("标注的bbox之前和之后可能还存在未标注的需要预测和sam修复")

        if not has_seg_manual_mask_predictor:
            # 如果has_seg_manual_mask_predictor为False，禁用这两个勾选框
            self.sam_repair_checkbox.setChecked(False)
            self.sam_repair_checkbox.setEnabled(False)  # 禁用勾选框

            self.prediction_repair_checkbox.setChecked(False)  # 这里可以选择是否将其设置为False
            self.prediction_repair_checkbox.setEnabled(False)  # 禁用勾选框

            self.predict_input.setText("0")
            self.predict_input.setEnabled(False)  # 禁用延拓

        # 添加选择列表
        self.choose_list_combobox = QComboBox()
        self.choose_list_combobox.addItem("所有非标注框重新计算")
        self.choose_list_combobox.addItem("只计算预测框")
        self.choose_list_combobox.addItem("只计算插值框")
        self.choose_list_combobox.addItem("不计算")
        # 设置默认选项为 "所有非标注框重新计算"
        default_option = "所有非标注框重新计算"
        index = self.choose_list_combobox.findText(default_option)
        if index != -1:
            self.choose_list_combobox.setCurrentIndex(index)
        self.choose_list_combobox.setToolTip("选择如何处理非标注框")

        # 创建带有确定和取消按钮的按钮框
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # 为按钮创建水平布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(button_box)

        # 为整个对话框创建垂直布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.interval_input)
        main_layout.addWidget(self.predict_input)
        main_layout.addWidget(self.sam_repair_checkbox)
        main_layout.addWidget(self.prediction_repair_checkbox)
        main_layout.addWidget(self.choose_list_combobox)
        main_layout.addLayout(button_layout)  # 将按钮布局添加到主布局中

        # 为QInputDialog设置布局
        self.setLayout(main_layout)

    def get_interval(self):
        interval_text = self.interval_input.text()
        try:
            interval_value = int(interval_text)
            return interval_value
        except ValueError:
            # Handle the case where the input is not a valid integer
            return 0
        
    def get_predict_input(self):
        predict_input = self.predict_input.text()
        try:
            predict_value = int(predict_input)
            return predict_value
        except ValueError:
            # Handle the case where the input is not a valid integer
            return 0
        
    def is_sam_repair_selected(self):
        # 获取插值Sam修复的勾选状态
        return self.sam_repair_checkbox.isChecked()

    def is_prediction_repair_selected(self):
        # 获取预测Sam修复的勾选状态
        return self.prediction_repair_checkbox.isChecked()

    def get_choose_list_option(self):
        # 获取选择列表中的选项
        current_option = self.choose_list_combobox.currentText()

        # 判断当前选项并返回相应的宏定义
        if current_option == "所有非标注框重新计算":
            return ALL_RECAL
        elif current_option == "只计算预测框":
            return CAL_PREDICTION
        elif current_option == "只计算插值框":
            return CAL_INTERPOLATION
        elif current_option == "不计算":
            return NO_CAL
        else:
            return NO_CAL