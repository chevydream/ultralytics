import sys, cv2, os, shutil, time, torch
from ultralytics import YOLO
from tkinter import filedialog
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile


# 画边框和标签
def my_box_label(im, box, label='', idx=0, color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = 3

    if isinstance(box, torch.Tensor):
        box = box.tolist()

    pt1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, pt1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        p1 = (int(box[0]+idx*w), int(box[1]))
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def cal_iou(box1, box2):
    """
    :param box1: = [left1, top1, right1, bottom1]
    :param box2: = [left2, top2, right2, bottom2]
    :return:
    """
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2

    # 计算每个矩形的面积
    s1 = (right1 - left1) * (bottom1 - top1)  # b1的面积
    s2 = (right2 - left2) * (bottom2 - top2)  # b2的面积

    # 计算相交矩形
    left = max(left1, left2)
    top = max(top1, top2)
    right = min(right1, right2)
    bottom = min(bottom1, bottom2)

    # 相交框的w,h
    w = max(0, right - left)
    h = max(0, bottom - top)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)

    return iou


# xywhn 格式计算
# xywhn 标记格式：中心坐标，宽度和高度
# xyxyn 标记格式：左上坐标，右下坐标
# 后缀n 表示已经归一化处理
def bbox_iou_xywhn(box1, box2, set_threshold=None, threshold=0.75):
    #如果box1,box2是一维数组,就复制成2维数组
    if box1.ndim == 1:
        box1 = np.tile(box1[np.newaxis, :], (2, 1))
    if box2.ndim == 1:
        box2 = np.tile(box2[np.newaxis, :], (2, 1))
    # 将box1和box2转换回xyxy格式
    box1_xy = box1[..., :2] - (box1[..., 2:] / 2.)
    box1_wh = box1[..., 2:]
    box1_x1y1 = np.concatenate((box1_xy, box1_xy + box1_wh), axis=-1)

    box2_xy = box2[..., :2] - (box2[..., 2:] / 2.)
    box2_wh = box2[..., 2:]
    box2_x1y1 = np.concatenate((box2_xy, box2_xy + box2_wh), axis=-1)

    # 计算交集的左上角和右下角点
    xy1 = np.maximum(box1_x1y1[:, None, :2], box2_x1y1[..., :2])
    xy2 = np.minimum(box1_x1y1[:, None, 2:], box2_x1y1[..., 2:])
    # 计算交集区域的宽和高
    wh = np.maximum(xy2 - xy1, 0)
    # 计算交集区域的面积
    inter_area = wh[..., 0] * wh[..., 1]
    # 计算并集区域的面积
    box1_area = box1_wh.prod(axis=-1)
    box2_area = box2_wh.prod(axis=-1)
    union_area = box1_area[:, None] + box2_area - inter_area
    # 计算IoU并返回结果
    iou = inter_area / union_area
    # 大于阈值的置成1，小于阈值的置成0
    if set_threshold:
        iou[iou >= threshold] = 1
        iou[iou < threshold] = 0
    return iou

# xywhn 格式计算
# xywhn 标记格式：中心坐标，宽度和高度
# xyxyn 标记格式：左上坐标，右下坐标
def bbox_iou_xywh(box1, box2):
    flag = 0

    # 如果box1,box2是一维数组,就复制成2维数组
    if box1.ndim == 1:
        box1 = np.tile(box1[np.newaxis, :], (2, 1))
        flag += 1
    if box2.ndim == 1:
        box2 = np.tile(box2[np.newaxis, :], (2, 1))
        flag += 1

    # 将box1和box2转换回xyxy格式
    box1_xy = box1[..., :2] - (box1[..., 2:] / 2.)
    box1_wh = box1[..., 2:]
    box1_x1y1 = np.concatenate((box1_xy, box1_xy + box1_wh), axis=-1)
    box2_xy = box2[..., :2] - (box2[..., 2:] / 2.)
    box2_wh = box2[..., 2:]
    box2_x1y1 = np.concatenate((box2_xy, box2_xy + box2_wh), axis=-1)

    # 计算交集的左上角和右下角点
    xy1 = np.maximum(box1_x1y1[:, None, :2], box2_x1y1[..., :2])
    xy2 = np.minimum(box1_x1y1[:, None, 2:], box2_x1y1[..., 2:])
    # 计算交集区域的宽和高
    wh = np.maximum(xy2 - xy1, 0)
    # 计算交集区域的面积
    inter_area = wh[..., 0] * wh[..., 1]
    # 计算并集区域的面积
    box1_area = box1_wh.prod(axis=-1)
    box2_area = box2_wh.prod(axis=-1)
    union_area = box1_area[:, None] + box2_area - inter_area
    # 计算IoU并返回结果
    iou = inter_area / union_area

    if flag == 2:
        iou = iou[0][0]

    return iou