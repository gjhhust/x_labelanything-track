import cv2
import numpy as np
import time
import torch
def load_annotations(frame_number):
    # Replace this function with your actual annotation loading logic
    if frame_number == 315:
        return {
            'car': [2215.4926757812495, 708.697021484375, 49.75439453125, 99.71868896484375],
            'people_24': [1893.7525634765625, 1283.7935791015623, 15.84130859375, 23.271484375],
            'people_33': [708.559326171875, 1351.549072265625, 25.91741943359375, 31.4149169921875]
        }
    elif frame_number == 325:
        return {
            'car': [2381.9331054687505, 792.7058715820312, 59.4111328125, 103.10101318359375],
            'people_24': [2015.8708496093748, 1455.98388671875, 15.4462890625, 27.2646484375],
            'people_33': [808.75341796875, 1408.220947265625, 26.37506103515625, 28.154052734375]
        }


def find_global_motion_transform(frame1, frame2):
    """
    使用OpenCV的全局运动匹配计算frame1到frame2的变换矩阵

    参数:
    - frame1: 第一帧图像
    - frame2: 第二帧图像

    返回:
    - transform_matrix: 变换矩阵
    - transformed_frame1: 经变换后的frame1图像
    """
    # 转换为灰度图像
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 使用全局运动匹配
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-5)
    _, transform_matrix = cv2.findTransformECC(gray_frame2, gray_frame1, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)

    # 对frame1进行变换
    h, w = frame1.shape[:2]
    transformed_frame1 = cv2.warpAffine(frame1, transform_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return transform_matrix, transformed_frame1

def find_perspective_transform(frame1, frame2):
    """
    使用SIFT特征和FLANN匹配器计算frame1到frame2的透视变换矩阵

    参数:
    - frame1: 第一帧图像
    - frame2: 第二帧图像

    返回:
    - perspective_matrix: 透视变换矩阵
    - transformed_frame1: 经透视变换后的frame1图像
    """
    # 使用SIFT特征检测器
    sift = cv2.SIFT_create()

    # 在两个图像上检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(frame1, None)
    kp2, des2 = sift.detectAndCompute(frame2, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=20)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 选择良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算透视变换矩阵
    perspective_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # # 对frame1进行透视变换
    # h, w = frame1.shape[:2]
    # transformed_frame1 = cv2.warpPerspective(frame1, perspective_matrix, (w, h))

    # return perspective_matrix, transformed_frame1
    return perspective_matrix

def save_difference_image(frame2, transformed_frame1, output_path):
    # 计算两个变换后图像的差异
    diff_image = cv2.absdiff(transformed_frame1, frame2)

    # 将差异图保存到文件
    cv2.imwrite(output_path, diff_image)

def linear_interpolate_bbox(start_bbox, end_bbox, alpha):
    interpolated_bbox = []
    for start_coord, end_coord in zip(start_bbox, end_bbox):
        interpolated_coord = (1 - alpha) * start_coord + alpha * end_coord
        interpolated_bbox.append(interpolated_coord)
    return interpolated_bbox

def apply_perspective_transform_to_bbox(perspective_matrix, bbox):
    bbox_array = np.array(bbox, dtype=np.float32).reshape(-1, 2)
    transformed_bbox = cv2.perspectiveTransform(bbox_array[None, :, :], perspective_matrix)[0]
    return transformed_bbox.flatten().tolist()

def apply_bboxes_xyxy(bboxes, M):
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)   # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T
        
# import sys
# sys.path.append('yolov5') 
# from models.experimental import attempt_load
# from utils.general import check_img_size, check_requirements, xyxy2xywh, non_max_suppression

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 计算交集的坐标
    intersection_x1 = max(x1, x2)
    intersection_y1 = max(y1, y2)
    intersection_x2 = min(x1 + w1, x2 + w2)
    intersection_y2 = min(y1 + h1, y2 + h2)
    
    # 计算交集的面积
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    
    # 计算并集的面积
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou
def calculate_iou_xyxy(bbox1, bbox2):
    # 提取矩形框的坐标
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 计算交叉区域的坐标
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)
    
    # 计算交叉区域和并集区域的面积
    area_intersection = w_intersection * h_intersection
    area_union = w1 * h1 + w2 * h2 - area_intersection
    
    # 计算交并比
    iou = area_intersection / area_union if area_union > 0 else 0.0
    
    return iou

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] # 排除最外层的连通图

def fix_bbox(Seg, bbox_orige, image, extend_boder_rate = -0.1):
    # bbox_orige is xywh
    x_boder = bbox_orige[2]*extend_boder_rate
    y_boder = bbox_orige[3]*extend_boder_rate
    bbox_orige_xyxy = [bbox_orige[0]-x_boder, bbox_orige[1]-y_boder, bbox_orige[0]+bbox_orige[2]+x_boder, bbox_orige[1]+bbox_orige[3]+y_boder]
    bbox_orige_xyxy = [int(coord) for coord in bbox_orige_xyxy]
    masks = Seg.image_predict(
        source=image.copy(),
        model_type="vit_l", # vit_l, vit_h, vit_b
        input_point=np.array([[bbox_orige_xyxy[0]+bbox_orige[2]//2, bbox_orige_xyxy[1]+bbox_orige[3]//3], 
                              [bbox_orige_xyxy[2]-bbox_orige[2]//2, bbox_orige_xyxy[3]-bbox_orige[2]//3]]),
        input_label=[0, 1],
        input_box= bbox_orige_xyxy, # or [[100, 100, 200, 200], [100, 100, 200, 200]]
        multimask_output=False,
        random_color=False,
        show=False,
        output_path="sam.png",
        save=True,
    )
    h, w = masks.shape[-2:]
    fix_bbox = mask_find_bboxs(masks[0].astype(np.uint8))[0][:4]
    iou = calculate_iou(fix_bbox, bbox_orige)
    print(f"iou:{iou}")
    if iou < 0.7:
        return bbox_orige
    sam_image = cv2.imread("sam.png")
    cv2.rectangle(sam_image, (fix_bbox[0], fix_bbox[1]), (fix_bbox[0] + fix_bbox[2], fix_bbox[1] + fix_bbox[3]), (0, 0, 255), 1)
    cv2.rectangle(sam_image, (bbox_orige_xyxy[0], bbox_orige_xyxy[1]), (bbox_orige_xyxy[2], bbox_orige_xyxy[3]), (255, 0, 0), 1)
    cv2.imwrite("sam.png",sam_image)
    return fix_bbox

def fix_bbox_xyxy(Seg, bbox_orige, image, extend_boder_rate = -0.1):
    # bbox_orige is xyxy
    bbox_w,bbox_h = bbox_orige[2]-bbox_orige[0], bbox_orige[3]-bbox_orige[1]
    x_boder = bbox_w*extend_boder_rate
    y_boder = bbox_h*extend_boder_rate
    bbox_orige_xyxy = [bbox_orige[0]-x_boder, bbox_orige[1]-y_boder, bbox_orige[2]+x_boder, bbox_orige[3]+y_boder]
    bbox_orige_xyxy = [int(coord) for coord in bbox_orige_xyxy]
    masks = Seg.image_predict(
        source=image.copy(),
        model_type="vit_l", # vit_l, vit_h, vit_b
        input_point=np.array([[bbox_orige_xyxy[0]+bbox_w//2, bbox_orige_xyxy[1]+bbox_h//3], 
                              [bbox_orige_xyxy[2]-bbox_w//2, bbox_orige_xyxy[3]-bbox_h//3]]),
        input_label=[0, 1],
        input_box= bbox_orige_xyxy, # or [[100, 100, 200, 200], [100, 100, 200, 200]]
        multimask_output=False,
        random_color=False,
        show=False,
        output_path="sam.png",
        save=False,
    )

    try:
        fix_bbox = mask_find_bboxs(masks[0].astype(np.uint8))[0][:4]
        fix_bbox = [fix_bbox[0],fix_bbox[1],fix_bbox[0]+fix_bbox[2],fix_bbox[1]+fix_bbox[3]]
        iou = calculate_iou_xyxy(fix_bbox, bbox_orige)
        print(f"iou:{iou}")
        if iou < 0.85:
            return bbox_orige
        # sam_image = cv2.imread("sam.png")
        # cv2.rectangle(sam_image, (fix_bbox[0], fix_bbox[1]), (fix_bbox[0] + fix_bbox[2], fix_bbox[1] + fix_bbox[3]), (0, 0, 255), 1)
        # cv2.rectangle(sam_image, (bbox_orige_xyxy[0], bbox_orige_xyxy[1]), (bbox_orige_xyxy[2], bbox_orige_xyxy[3]), (255, 0, 0), 1)
        # cv2.imwrite("sam.png",sam_image)
        return fix_bbox
    except:
        return bbox_orige
    

import cv2
import numpy as np

def crop_and_resize_image(image_, bbox, target_size=640, extend_border_rate=0.5):
    x_border = bbox[2] * extend_border_rate
    y_border = bbox[3] * extend_border_rate
    bbox = [bbox[0] - x_border, bbox[1] - y_border, bbox[2] + x_border, bbox[3] + y_border]
    bbox = [int(coord) for coord in bbox]

    # 读取图像
    image = image_.copy()

    # 获取bbox的坐标信息
    x, y, w, h = bbox

    # 裁剪图像
    cropped_image = image[y:y+h, x:x+w]

    # 获取原图像的宽高
    original_height, original_width = cropped_image.shape[:2]

    # 计算缩放比例
    scale_factor = target_size / max(original_height, original_width)

    # 调整图像大小
    resized_image = cv2.resize(cropped_image, None, fx=scale_factor, fy=scale_factor)

    # 获取调整后的图像尺寸
    new_height, new_width = resized_image.shape[:2]

    # 将短边调整到距离64的倍数最近的数字
    pad_height = int(np.ceil(new_height / 64) * 64 - new_height)
    pad_width = int(np.ceil(new_width / 64) * 64 - new_width)

    # 使用pad图像右侧填充得到最终的结果图
    final_image = cv2.copyMakeBorder(resized_image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[114, 114, 114])

    return final_image, scale_factor, [x, y]

def inverse_crop_and_resize(bbox, scale_factor, original_cord):
    x, y, w, h = bbox

    # 逆缩放
    x /= scale_factor
    y /= scale_factor
    w /= scale_factor
    h /= scale_factor

    # 逆裁剪
    x += original_cord[0]
    y += original_cord[1]

    return [int(coord) for coord in [x, y, w, h]]

def run_image(model, img_):
    img = torch.from_numpy(img_)
    img = img.float().permute(2, 0, 1) # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=True)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)[0]
    # 找到第五维度数字最大的一行的索引
    max_row_index = torch.argmax(pred[:, 4])
    # 取出第五维度数字最大的一行
    xyxy = pred[max_row_index][:4].view(-1).int().tolist()

    xywh = [xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]]
    return xywh 

def fix_human(model, image_orige, crop_bbox):
    # 调用裁剪函数
    cropped_resized_image, scale_factor, original_cord = crop_and_resize_image(image_orige.copy(), crop_bbox)

    pred_bbox = run_image(model, cropped_resized_image)

    cv2.rectangle(cropped_resized_image, (pred_bbox[0], pred_bbox[1]), (pred_bbox[0]+pred_bbox[2],pred_bbox[1]+ pred_bbox[3]), (255, 0, 0), 1)
    cv2.imwrite("fix_human_crop.png",cropped_resized_image)

    # 调用逆函数获取原始坐标
    fix_human_bbox = inverse_crop_and_resize(pred_bbox, scale_factor, original_cord)

    show = image_orige.copy()
    crop_bbox = [int(coord) for coord in crop_bbox]
    cv2.rectangle(show, (fix_human_bbox[0], fix_human_bbox[1]), (fix_human_bbox[0] + fix_human_bbox[2], fix_human_bbox[1] + fix_human_bbox[3]), (0, 0, 255), 1)
    cv2.rectangle(show, (crop_bbox[0], crop_bbox[1]), (crop_bbox[0]+crop_bbox[2],crop_bbox[1]+ crop_bbox[3]), (255, 0, 0), 1)
    cv2.imwrite("fix_human.png",show)
    
    return fix_human_bbox
    



def apply_bbox(M, bbox):
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        bboxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)   # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        new_bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T
        return [new_bbox[0][0], new_bbox[0][1], new_bbox[0][2]-new_bbox[0][0], new_bbox[0][3]-new_bbox[0][1]]

def apply_bbox_xyxy(M, bbox):
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        bboxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)   # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        new_bbox = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T
        return new_bbox[0]


def draw_save_ann(frame, ann, save_path):
    # 绘制标注到帧上
    saveframe = frame.copy()
    for obj, bbox in ann.items():
        bbox = [int(coord) for coord in bbox]
        cv2.rectangle(saveframe, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1)
        cv2.putText(saveframe, obj, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 保存插值后的帧
    cv2.imwrite(save_path, saveframe)


def interpolate_and_draw(start_frame, end_frame, target_frame, start_annotations, end_annotations, start_fn, end_fn, target_frame_number, output_path):
    # 计算首尾帧到目标帧的透视变换
    perspective_matrix_start_to_target, trans_start_frame = find_perspective_transform(start_frame, target_frame)
    perspective_matrix_end_to_target, trans_end_frame = find_perspective_transform(end_frame, target_frame)

    # 获取目标帧在首尾帧之间的相对位置
    alpha = (target_frame_number - start_fn) / (end_fn - start_fn)

    # 应用透视变换到标注框
    transformed_annotations_start = {}
    transformed_annotations_end = {}

    Seg = SegManualMaskPredictor()
    # yolov5_model = attempt_load("crowdhuman_yolov5m.pt", map_location="cpu")  # load FP32 model
    # yolov5_model(torch.zeros(1, 3, 640, 640).type_as(next(yolov5_model.parameters())))  # run once

    for obj in start_annotations.keys():
        transformed_bbox_start = apply_bbox(perspective_matrix_start_to_target, start_annotations[obj])
        transformed_annotations_start[obj] = transformed_bbox_start

        transformed_bbox_end = apply_bbox(perspective_matrix_end_to_target, end_annotations[obj])
        transformed_annotations_end[obj] = transformed_bbox_end

    # 计算目标帧的标注
    final_annotations = {}
    final_annotations_line = {}
    final_annotations_fix = {}
    for obj in start_annotations.keys():
        interpolated_bbox = linear_interpolate_bbox(transformed_annotations_start[obj], transformed_annotations_end[obj], alpha)
        final_annotations[obj] = interpolated_bbox
        interpolated_bbox_fix = fix_bbox(Seg, interpolated_bbox, target_frame)
        # interpolated_bbox_fix = fix_human(yolov5_model, target_frame, interpolated_bbox)
        final_annotations_fix[obj] = interpolated_bbox_fix
        interpolated_bbox = linear_interpolate_bbox(start_annotations[obj], end_annotations[obj], alpha)
        final_annotations_line[obj] = interpolated_bbox

    draw_save_ann(target_frame, final_annotations, output_path.format(target_frame_number))
    draw_save_ann(target_frame, final_annotations_line, output_path.format(f"{target_frame_number}_line"))
    draw_save_ann(target_frame, final_annotations_fix, output_path.format(f"{target_frame_number}_fix"))


    draw_save_ann(trans_start_frame, transformed_annotations_start, "output/start_trans.png")
    draw_save_ann(trans_end_frame, transformed_annotations_end, "output/end_trans.png")

if __name__ == "__main__":

    # 示例用法：
    start_fn = 315
    end_fn = 325
    target_frame_number = 322
    start_frame = cv2.imread(f"images/{start_fn}.png")  # 请替换为实际的图像路径
    end_frame = cv2.imread(f"images/{end_fn}.png")  # 请替换为实际的图像路径
    target_frame = cv2.imread(f"images/{target_frame_number}.png")
    # #查看变换矩阵是否正确
    # perspective_matrix, transformed_frame1 = find_perspective_transform(frame1, frame2)
    # save_difference_image(frame2, transformed_frame1, "difference_image.png") 

    start_annotations = load_annotations(315)
    end_annotations = load_annotations(325)


    output_path = "output/interpolated_frame{}.png"  # Replace with the desired output path

    interpolate_and_draw(start_frame, end_frame, target_frame, start_annotations, end_annotations, start_fn = start_fn, end_fn = end_fn, 
                        target_frame_number = target_frame_number, output_path = output_path)

    # 在这里，perspective_matrix包含frame1到frame2的透视变换矩阵
    # transformed_frame1是frame1经透视变换后的图像
