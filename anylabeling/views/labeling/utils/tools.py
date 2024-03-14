import PIL.Image
import io
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import QPointF
import random
import math

from PyQt5.QtCore import QPointF
import math
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QProgressDialog,
    QLabel,
)
def get_numpy_imagedata(image_data):
    numpy_image = np.array(PIL.Image.open(io.BytesIO(image_data)))
    return numpy_image


def sort_points_by_distance(label_list):
    if not label_list or not all(isinstance(label.shape().points[0], QPointF) for label in label_list):
        raise ValueError("Input should be a non-empty list of labels with QPointF points.")

    sorted_list = sorted(label_list, key=lambda label: distance(label.shape().points[0], QPointF(0, 0)))

    origin = QPointF(0, 0)
    sum_sorted_list = [sorted_list.pop(0)]

    progress_dialog = QProgressDialog(
            "Generate firstframe. Please wait...",
            "Cancel",
            0,
            len(sorted_list),
        )
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setWindowTitle("Sorted Progress")
    i = 0
    while sorted_list:
        closest_label = min(filter(lambda label: label not in sum_sorted_list,
                                    sorted_list),
                            key=lambda label: distance(sum_sorted_list[-1].shape().points[0],
                                                      label.shape().points[0]))
        
        sum_sorted_list.append(closest_label)
        sorted_list.remove(closest_label)

        progress_dialog.setValue(i)
        if progress_dialog.wasCanceled():
            break
        
    progress_dialog.close()

    return sum_sorted_list

from PyQt5.QtCore import QPointF
import math
from sklearn.cluster import KMeans

def distance(point1, point2):
    return math.sqrt((point1.x() - point2.x())**2 + (point1.y() - point2.y())**2)

def k_means_clustering(labels, k):
    if not labels or not all(isinstance(label.shape().points[0], QPointF) for label in labels):
        raise ValueError("输入应为包含 QPonitF 点的标签列表且不能为空.")

    # 使用不同的变量名
    points = [(label.shape().points[0].x(), label.shape().points[0].y()) for label in labels]

    kmeans = KMeans(n_clusters=k, random_state=42).fit(points)

    clustered_labels = [[] for _ in range(k)]
    for label, cluster_label in zip(labels, kmeans.labels_):
        clustered_labels[cluster_label].append(label)

    return clustered_labels

def k_means_clustering(labels, k):
    if not labels or not all(isinstance(label.shape().points[0], QPointF) for label in labels):
        raise ValueError("输入应为包含 QPointF 点的标签列表且不能为空.")

    # 使用不同的变量名
    points = [(label.shape().points[0].x(), label.shape().points[0].y()) for label in labels]

    kmeans = KMeans(n_clusters=k, random_state=42).fit(points)

    clustered_labels_kmeans_lists = [[] for _ in range(k)]
    cluster_centers_radii = []  # 存储每个簇的中心点和大致半径信息

    for i, label in enumerate(labels):
        cluster_label = kmeans.labels_[i]
        clustered_labels_kmeans_lists[cluster_label].append(label)

    for cluster_label in range(k):
        cluster_points = [(label.shape().points[0].x(), label.shape().points[0].y()) for label in clustered_labels_kmeans_lists[cluster_label]]
        cluster_center = QPointF(sum(x for x, y in cluster_points) / len(cluster_points),
                                 sum(y for x, y in cluster_points) / len(cluster_points))

        # 计算大致半径
        max_distance = max(distance(QPointF(x, y), cluster_center) for x, y in cluster_points)
        cluster_centers_radii.append((cluster_center, max_distance))

    # 根据中心点的 x 和 y 坐标进行排序
    sorted_indices = sorted(range(len(cluster_centers_radii)), key=lambda i: (cluster_centers_radii[i][0].x(), cluster_centers_radii[i][0].y()))

    sorted_clustered_labels_kmeans_lists = [clustered_labels_kmeans_lists[i] for i in sorted_indices]

    return sorted_clustered_labels_kmeans_lists

def prim_algorithm(label_list):
    if not label_list or not all(isinstance(label.shape().points[0], QPointF) for label in label_list):
        raise ValueError("Input should be a non-empty list of labels with QPointF points.")

    origin = QPointF(0, 0)

    start_label = min(label_list, key=lambda label: distance(label.shape().points[0], origin))

    unvisited_labels = set(label_list)
    visited_labels = set([start_label])
    result_labels = [start_label]

    while unvisited_labels:
        min_distance = float('inf')
        closest_label = None

        for visited_label in visited_labels:
            for unvisited_label in unvisited_labels:
                dist = distance(visited_label.shape().points[0], unvisited_label.shape().points[0])
                if dist < min_distance:
                    min_distance = dist
                    closest_label = unvisited_label

        if closest_label:
            visited_labels.add(closest_label)
            unvisited_labels.remove(closest_label)
            result_labels.append(closest_label)

    return result_labels

def sort_points_by_clusters(label_list, k=4):
    if not label_list or not all(isinstance(label.shape().points[0], QPointF) for label in label_list):
        raise ValueError("Input should be a non-empty list of labels with QPointF points.")

    # 进行 KMeans 聚类
    clustered_labels_kmeans_lists = k_means_clustering(label_list, k)

    # # 对每个聚类应用 Prim 算法，整体排序
    # sorted_clusters_kmeans_lists = prim_algorithm(clustered_labels_kmeans_lists)

    # 对每个聚类内部应用 Prim 算法
    sorted_list = []
    for cluster_labels in clustered_labels_kmeans_lists:
        sorted_list.extend(prim_algorithm(cluster_labels))

    return sorted_list

def generate_random_points(num_points, range_x=(0, 10), range_y=(0, 10)):
    return [QPointF(random.uniform(range_x[0], range_x[1]), random.uniform(range_y[0], range_y[1])) for _ in range(num_points)]

def visualize_sorted_points(sorted_list):
    fig, ax = plt.subplots()

    for i, label in enumerate(sorted_list):
        # 根据索引选择颜色
        color = plt.cm.rainbow(i / len(sorted_list))
        
        for point in label.shape().points():
            ax.scatter(point.x(), point.y(), color=color)

            # 在每个点中间标注索引
            ax.annotate(str(i), (point.x(), point.y()), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    ax.set_aspect('equal', adjustable='box')
    plt.show()

from PyQt5.QtCore import QRectF

def calculate_iou(rect1: QRectF, rect2: QRectF) -> float:
    # 计算两个矩形的交集
    intersection = rect1.intersected(rect2)
    
    # 如果交集为空，则IOU为0
    if intersection.isEmpty():
        return 0.0
    
    # 计算交集的面积
    intersection_area = intersection.width() * intersection.height()
    
    # 计算并集的面积
    union_area = rect1.width() * rect1.height() + rect2.width() * rect2.height() - intersection_area
    
    # 计算IOU
    iou = intersection_area / union_area
    
    return iou

# class Label:
#     def __init__(self, points):
#         self._shape = Shape(points)

#     def shape(self):
#         return self._shape

# class Shape:
#     def __init__(self, points):
#         self._points = points

#     def points(self):
#         return self._points
    
if __name__ == "__main__":
    num_points_per_cluster = 5  # 每个聚类中的点数
    num_clusters = 3  # 聚类数

    label_list = []
    for i in range(num_clusters):
        cluster_center = QPointF(random.uniform(0, 10), random.uniform(0, 10))
        cluster_points = generate_random_points(num_points_per_cluster, range_x=(cluster_center.x() - 1, cluster_center.x() + 1), range_y=(cluster_center.y() - 1, cluster_center.y() + 1))
        label_list.append(Label(cluster_points))

    sorted_list = sort_points_by_clusters(label_list, k=num_clusters)
    print("排序后的列表:")
    for label in sorted_list:
        print(label.shape().points()[0])

    visualize_sorted_points(sorted_list)